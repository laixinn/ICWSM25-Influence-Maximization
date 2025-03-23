import torch
from torch import nn
import numpy as np
import math, time

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return NotImplemented


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features, n_time_features,
                                            embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device

    # def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
    #                       use_time_proj=True):
    #   """Recursive implementation of curr_layers temporal graph attention layers.
    #
    #   src_idx_l [batch_size]: users / items input ids.
    #   cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    #   curr_layers [scalar]: number of temporal convolutional layers to stack.
    #   num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    #   """
    #
    #   assert (n_layers >= 0)
    #
    #   source_node_features = torch.from_numpy(self.node_features.loc[[(s, t) for s, t in zip(source_nodes, timestamps)]].values.astype(np.float32)).to(self.device)
    #
    #   source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    #   timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)
    #
    #   # query node always has the start time -> time span == 0
    #   source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
    #     timestamps_torch))
    #
    #   # source_node_features = self.node_features[source_nodes_torch, :]
    #
    #   if self.use_memory:
    #     source_node_features = memory[source_nodes, :] + source_node_features
    #
    #   if n_layers == 0:
    #     return source_node_features
    #   else:
    #
    #     source_node_conv_embeddings = self.compute_embedding(memory,
    #                                                          source_nodes,
    #                                                          timestamps,
    #                                                          n_layers=n_layers - 1,
    #                                                          n_neighbors=n_neighbors)
    #
    #     neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
    #       source_nodes,
    #       timestamps,
    #       n_neighbors=n_neighbors)
    #
    #     neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    #
    #     edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
    #
    #     edge_deltas = timestamps[:, np.newaxis] - edge_times
    #
    #     edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
    #
    #     neighbors = neighbors.flatten()
    #     neighbor_embeddings = self.compute_embedding(memory,
    #                                                  neighbors,
    #                                                  np.repeat(timestamps, n_neighbors),
    #                                                  n_layers=n_layers - 1,
    #                                                  n_neighbors=n_neighbors)
    #
    #     effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    #     neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
    #     edge_time_embeddings = self.time_encoder(edge_deltas_torch)
    #
    #     edge_features = self.edge_features[edge_idxs, :]
    #
    #     mask = neighbors_torch == 0
    #
    #     source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
    #                                       source_nodes_time_embedding,
    #                                       neighbor_embeddings,
    #                                       edge_time_embeddings,
    #                                       edge_features,
    #                                       mask)
    #
    #     return source_embedding

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        # # when np.repeat(timestamps, n_neighbors)
        # st_index = np.array([(s, t) for s, t in zip(source_nodes, timestamps)])
        # mask = np.array([(s, t) in self.node_features.index for s, t in zip(source_nodes, timestamps)]).astype(bool)
        # source_node_features = np.zeros([len(source_nodes), self.node_features.shape[1]])
        # source_node_features[mask] = self.node_features.loc[[(s, t) for s, t in st_index[mask]]].values
        # source_node_features = torch.from_numpy(source_node_features).long().to(self.device)

        st_index = np.stack([source_nodes, timestamps], axis=1)
        source_node_features = np.zeros([len(source_nodes), self.node_features.shape[1]])
        node_features = self.node_features.reset_index()
        unique_t = np.unique(st_index[:, 1])
        ut_res = []
        for ut in unique_t:
            ut_feat = node_features[node_features['timestamp'] == ut].set_index('role_id').drop(columns=['timestamp'])
            ut_s = st_index[st_index[:, 1] == ut][:, 0]
            ut_data = np.zeros([len(ut_s), ut_feat.shape[1]])
            ut_mask = np.in1d(ut_s, ut_feat.index.values)
            ut_data[ut_mask] = ut_feat.loc[ut_s[ut_mask]].values
            # assert (np.in1d(ut_s, ut_feat.index.values)).all(), f'missing features at: {ut}, {ut_s[~np.in1d(ut_s, ut_feat.index.values)]}'
            ut_res.append(ut_data)
        source_node_features = np.concatenate(ut_res)
        source_node_features = torch.from_numpy(source_node_features).long().to(self.device)

        # when edge_times
        # source_node_features = torch.from_numpy(self.node_features.loc[[(s, t) for s, t in zip(source_nodes, timestamps)]].values.astype(np.float32)).to(self.device)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        # source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            source_node_conv_embeddings = self.compute_embedding(memory,
                                                                 source_nodes,
                                                                 timestamps,
                                                                 n_layers=n_layers - 1,
                                                                 n_neighbors=n_neighbors)

            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            # assert (edge_times < timestamps[:, np.newaxis]).all()

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            edge_times = edge_times.flatten()
            neighbor_embeddings = self.compute_embedding(memory,
                                                         neighbors,
                                                         # edge_times,
                                                         np.repeat(timestamps, n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

            return source_embedding

    def compute_tensor_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                 use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        st_index = torch.stack([source_nodes, timestamps], axis=1).cpu().numpy()
        # mask = np.array([(s, t) in self.node_features.index for s, t in st_index]).astype(bool)

        source_node_features = np.zeros([len(source_nodes), self.node_features.shape[1]])
        # source_node_features[mask] = self.node_features.loc[[(s, t) for s, t in st_index[mask]]].values

        ## get feature
        node_features = self.node_features.reset_index()
        unique_t = np.unique(st_index[:, 1])
        ut_res = []
        for ut in unique_t:
            ut_feat = node_features[node_features['timestamp'] == ut].set_index('role_id')
            ut_s = st_index[st_index[:, 1] == ut][:, 0]
            assert (np.in1d(ut_s,
                            ut_feat.index.values)).all(), f'missing features at: {ut}, {ut_s[~np.in1d(ut_s, ut_feat.index.values)]}'
            ut_res.append(ut_feat.loc[ut_s].drop(columns=['timestamp']).values)
        source_node_features = np.concatenate(ut_res)

        source_node_features = torch.from_numpy(source_node_features).long().to(self.device)

        # timestamps = torch.unsqueeze(timestamps, dim=1).float()

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps.unsqueeze(dim=1).float()))

        # source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            source_node_conv_embeddings = self.compute_tensor_embedding(memory,
                                                                        source_nodes,
                                                                        timestamps,
                                                                        n_layers=n_layers - 1,
                                                                        n_neighbors=n_neighbors)

            t1 = time.time()
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes.cpu().numpy(),
                timestamps.cpu().numpy(),
                n_neighbors=n_neighbors)
            # print(f"get_temporal_neighbor takes {time.time()-t1}")

            # assert (edge_times < timestamps[:, np.newaxis]).all()

            neighbors = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps.unsqueeze(dim=1) - torch.from_numpy(edge_times).float().to(self.device)

            # neighbors = neighbors.flatten()
            # edge_times = edge_times.flatten()
            neighbor_embeddings = self.compute_tensor_embedding(memory,
                                                                neighbors.flatten(),
                                                                torch.from_numpy(edge_times.flatten()).to(self.device),
                                                                # timestamps.repeat(n_neighbors),
                                                                n_layers=n_layers - 1,
                                                                n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors == 0

            source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                edge_features=edge_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_edge_features=n_edge_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                n_heads=n_heads, dropout=dropout,
                                                use_memory=use_memory)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                             n_edge_features, embedding_dimension)
                                             for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                             embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       edge_features=edge_features,
                                       memory=memory,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 n_heads=n_heads, dropout=dropout, use_memory=use_memory)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout,
                             n_neighbors=n_neighbors)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))


