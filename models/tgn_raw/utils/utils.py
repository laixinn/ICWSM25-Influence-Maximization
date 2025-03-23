import time, multiprocessing

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import dgl


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3, n_classes=1):
    super().__init__()
    self.n_classes = n_classes
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, n_classes)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)

class TRandEdgeSampler(object):
  def __init__(self, src_list, dst_list, timestamp, seed=None):
    self.seed = None

    self.t_to_src = {t: [] for t in timestamp}
    self.t_to_dst = {t: [] for t in timestamp}
    for src, dst, t in zip(src_list, dst_list, timestamp):
      self.t_to_src[t].append(src)
      self.t_to_dst[t].append(dst)
    self.t_to_src = {k: np.array(list(set(v))) for k, v in self.t_to_src.items()}
    self.t_to_dst = {k: np.array(list(set(v))) for k, v in self.t_to_dst.items()}

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, timestamps, dtype=np.int64):
    if type(timestamps) == torch.Tensor:
      timestamps = timestamps.cpu().numpy()
    src_samples = np.zeros(len(timestamps))
    dst_samples = np.zeros(len(timestamps))
    t_elements = np.unique(timestamps)
    t_elements.sort()
    for t in t_elements:
      t_mask = timestamps==t
      t_size = sum(t_mask)
      t_src, t_dst = self.sample_t(t_size, t)
      src_samples[t_mask] = t_src
      dst_samples[t_mask] = t_dst
    return src_samples.astype(dtype), dst_samples.astype(dtype)

  def sample_t(self, size, t):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.t_to_src[t]), size)
      dst_index = np.random.randint(0, len(self.t_to_dst[t]), size)
    else:
      src_index = self.random_state.randint(0, len(self.t_to_src[t]), size)
      dst_index = self.random_state.randint(0, len(self.t_to_dst[t]), size)

    return self.t_to_src[t][src_index.tolist()], self.t_to_dst[t][dst_index.tolist()]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

'''test features'''

class NeighborFinder_archived:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      if type(source_node) == torch.Tensor:
        source_node = source_node.cpu().numpy()
      if type(timestamp) == torch.Tensor:
        timestamp = timestamp.cpu().numpy()
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

  def get_temporal_neighbor_parallel(self, source_nodes, timestamps, n_neighbors=20, num_worker=2):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    # # map
    # if num_worker is None:
    #   num_worker = multiprocessing.cpu_count()
    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool(processes=num_worker)
    #
    # partition = len(source_nodes)//num_worker
    # data_list = [[source_nodes[w*partition: (w+1)*partition], timestamps[w*partition: (w+1)*partition], n_neighbors] for w in range(num_worker)]
    # results = pool.map(self.get_temporal_neighbor, data_list)
    #
    # pool.close()
    # pool.join()

    # map
    ctx = torch.multiprocessing.get_context("spawn")
    if num_worker is None:
      num_worker = torch.multiprocessing.cpu_count()
    pool = ctx.Pool(num_worker)

    pool_list = []
    partition = len(source_nodes) // num_worker
    for w in range(num_worker):
      w_src = source_nodes[w * partition: (w + 1) * partition]
      w_ts = timestamps[w * partition: (w + 1) * partition]
      res = pool.apply_async(self.get_temporal_neighbor, args=(w_src, w_ts, n_neighbors))
      pool_list.append(res)

    pool.close()
    pool.join()

    # reduce
    for i, item in enumerate(pool_list):
      _neighbors, _edge_idxs, _edge_times = item.get()
      neighbors[i*partition: (i+1)*partition] = _neighbors
      edge_idxs[i * partition: (i + 1) * partition] = _edge_idxs
      edge_times[i * partition: (i + 1) * partition] = _edge_times

    return neighbors, edge_idxs, edge_times


class TensorRandEdgeSampler(object):
  def __init__(self, src_list, dst_list, timestamp, seed=None, device='cuda'):
    self.src = src_list
    self.dst = dst_list
    self.ts = timestamp
    self.device = device

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, timestamps, dtype=torch.int64):
    if type(timestamps) == np.ndarray:
      timestamps = torch.from_numpy(timestamps).to(self.device)
    src_samples = torch.zeros(len(timestamps), dtype=dtype).to(self.device)
    dst_samples = torch.zeros(len(timestamps), dtype=dtype).to(self.device)
    t_elements = torch.unique(timestamps)
    t_elements, _ = t_elements.sort()
    for t in t_elements:
      t_mask = timestamps==t
      t_size = sum(t_mask)
      t1 = time.time()
      t_src, t_dst = self.sample_t(t_size, t)
      print(f"sample_t takes {time.time()-t1}")
      src_samples[t_mask] = t_src
      dst_samples[t_mask] = t_dst
    return src_samples, dst_samples

  def sample_t(self, size, t):
    t = t.cpu().numpy()
    size = size.cpu().numpy()
    t_src = torch.unique(torch.from_numpy(self.src[self.ts == t])).to(self.device)
    t_dst = torch.unique(torch.from_numpy(self.dst[self.ts == t])).to(self.device)
    if self.seed is None:
      src_index = np.random.randint(0, len(t_src), size)
      dst_index = np.random.randint(0, len(t_dst), size)
    else:
      src_index = self.random_state.randint(0, len(t_src), size)
      dst_index = self.random_state.randint(0, len(t_dst), size)
    src_index = torch.from_numpy(src_index).to(self.device)
    dst_index = torch.from_numpy(dst_index).to(self.device)

    return t_src[src_index], t_dst[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)

class TensorDict():
  def __init__(self, pydict, device='cuda'):
    # csr manner
    self.device = device
    # Convert dictionary to tensors
    keys_tensor = torch.tensor(list(pydict.keys()), dtype=torch.int64, device=device)
    # values_tensor = torch.tensor(list(pydict.values()), dtype=torch.int64)
    values_tensor = pad_sequence(
      [torch.from_numpy(item).long().to(device) for item in pydict.values()],
      batch_first=True)

    # Sort keys tensor for efficient search
    sorted_keys, sorted_indices = torch.sort(keys_tensor)

    self.sorted_keys = sorted_keys.to(device)
    self.sorted_indices = sorted_indices.to(device)
    self.values = values_tensor.to(device)

  def __getitem__(self, sources):
    # Searchsorted to find indices for given IDs
    indices = torch.searchsorted(self.sorted_keys, sources)
    # Map indices using sorted_indices
    mapped_values = self.values[self.sorted_indices[indices]]
    return mapped_values

  # def __setitem__(self, sources, value):
  #   indices = torch.searchsorted(self.sorted_keys, sources)
  #   self.values[self.sorted_indices[indices]] = value

class TensorNeighborFinder_v0:
  def __init__(self, adj_list, uniform=False, seed=None, device='cuda'):
    self.device = device
    node_to_neighbors = {}
    node_to_edge_idxs = {}
    node_to_edge_timestamps = {}

    for i, neighbors in enumerate(adj_list):
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      node_to_neighbors[i] = np.array([x[0] for x in sorted_neighhbors])
      node_to_edge_idxs[i] = np.array([x[1] for x in sorted_neighhbors])
      node_to_edge_timestamps[i] = np.array([x[2] for x in sorted_neighhbors])

    self.node_to_neighbors = TensorDict(node_to_neighbors, device=device)
    self.node_to_edge_idxs = TensorDict(node_to_edge_idxs, device=device)
    self.node_to_edge_timestamps = TensorDict(node_to_edge_timestamps, device=device)

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    source_neighbors = self.node_to_neighbors[source_nodes]
    source_edge_idxs = self.node_to_edge_idxs[source_nodes]
    source_edge_times = self.node_to_edge_timestamps[source_nodes]
    # t_index = torch.searchsorted(source_edge_times, timestamps.unsqueeze(1))
    #
    # index_tensor = torch.arange(source_neighbors.shape[1]).unsqueeze(0).expand(source_neighbors.shape[0],
    #                                                                            source_neighbors.shape[1]).to(self.device)
    # t_index = torch.lt(source_edge_times, t_index)
    t_index = torch.lt(source_edge_times, timestamps.unsqueeze(1))
    source_neighbors = source_neighbors[t_index]
    source_edge_idxs = source_edge_idxs[t_index]
    source_edge_times = source_edge_times[t_index]

    sampled_idx = source_neighbors.multinomial(n_neighbors)
    neighbors = source_neighbors[sampled_idx]
    edge_times = source_edge_times[sampled_idx]
    edge_idxs = source_edge_idxs[sampled_idx]

    edge_times, pos = torch.sort(edge_times, dim=1)
    neighbors = neighbors[pos]
    edge_idxs = edge_idxs[pos]

    return neighbors, edge_idxs, edge_times

def get_tensor_neighbor_finder_deprecated(data, uniform, max_node_idx=None, device='cuda'):
  unique_timestamps = np.unique(data.timestamps)
  unique_timestamps.sort()
  t_graphs = {}
  t_dicts = {}
  for t in unique_timestamps:
    t_index = data.timestamps == t
    t_src, t_dst, t_eidx = data.sources[t_index], data.destinations[t_index], data.edge_idxs[t_index]
    t_unique_node = np.unique(np.stack([t_src, t_dst]))
    node2index = dict({v: k for k, v in enumerate(t_unique_node)})
    n2i_vec = np.vectorize(lambda x: node2index[x])
    t_src_idx = torch.from_numpy(n2i_vec(t_src))
    t_dst_idx = torch.from_numpy(n2i_vec(t_dst))
    tg = dgl.graph((t_src_idx, t_dst_idx))
    tg.ndata['_ID'] = torch.from_numpy(t_unique_node)
    tg.edata['eidx'] = torch.from_numpy(t_eidx)
    tg = dgl.to_simple(dgl.add_reverse_edges(tg, copy_ndata=True, copy_edata=True), copy_ndata=True, copy_edata=True)

    t_graphs[t] = tg.to(device)
    t_dicts[t] = node2index

  return TensorNeighborFinder(t_graphs, t_dicts, uniform=uniform, device=device)

class TensorNeighborFinder_deprecated:
  def __init__(self, graphs, node2index, uniform=False, seed=None, device='cuda'):
    self.graphs = graphs
    # self.node2index = node2index
    self.device = device

    # tensorize dict
    self.tensor_dict = {}
    for t, t_dict in node2index.item():
      self.tensor_dict[t] = TensorDict(t_dict)

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = torch.zeros((len(source_nodes), tmp_n_neighbors)).int().to(self.device) # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = torch.zeros((len(source_nodes), tmp_n_neighbors)).float().to(self.device)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = torch.zeros((len(source_nodes), tmp_n_neighbors)).int().to(self.device)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    unique_timestamp, _ = torch.unique(timestamps).sort()
    for t in unique_timestamp:
      t_index = timestamps == t
      t1 = time.time()
      src_node = torch.Tensor([self.node2index[t.cpu().tolist()][item] for item in source_nodes[t_index].cpu().tolist()]).long().to(self.device)
      t2 = time.time()
      src_node = self.tensor_dict[t][source_nodes]
      print(f"origin takes {t2-t1}, new takes {time.time()-t2}")
      for gtime, tgraph in self.graphs.items():
        if t <= gtime: break
        sampled_graph = dgl.sampling.sample_neighbors(tgraph, src_node, n_neighbors, replace=True)

        # adj_list = [sampled_nodes[i].tolist() for i in range(len(sampled_nodes))]

        neighbors[t_index] = sampled_graph.ndata['_ID']
        edge_times[t_index] = gtime
        edge_idxs[t_index] = sampled_graph.edata['eidx']

    return neighbors, edge_idxs, edge_times