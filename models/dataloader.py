import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pickle, pandas as pd
import os, re, math
import numpy as np

import dgl
from scipy.sparse import csr_matrix
from data_processing.read_tweet import dict_to_df


class StaticTwitter(Dataset):
    def __init__(self, nodes, labels, graphs):
        '''
        Twitter dataset
        :param nodes: np array
        :param labels: DataFrame with columns [role_id, label]
        :param graphs: csr_matrix
        '''
        super(StaticTwitter, self).__init__()
        self.nodes = nodes if type(nodes) == np.ndarray else np.array(nodes)
        self.labels = labels
        self.graphs = graphs
        self.length = labels.shape[0]
        assert self.length == len(self.nodes)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if type(index) != slice:
            index = [index]
        nodes = self.nodes[index]
        labels = self.labels.loc[nodes].values
        graphs = self.graphs[nodes]
        return nodes, labels, graphs

    def collate_fn(self, data):
        batch_node = np.array([item[0] for item in data])
        batch_label = np.array([item[1] for item in data])
        batch_graph = [item[2] for item in data]
        return batch_node, batch_label, batch_graph


class DGLDataset(Dataset):
    def __init__(self, graphs, mode, batch_size, sampling=False):
        '''
        Twitter dataset
        :param features: DataFrame with columns [role_id, ...]
        :param labels: DataFrame with columns [role_id, label]
        :param graphs: DataFrame with columns [role_id, t_role_id]
        '''
        super(DGLDataset, self).__init__()
        self.graphs = graphs
        self.mode = mode
        self.batch_size = batch_size
        self.sampling = sampling

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    def collate_fn(self, graph):
        return graph


class DGLData(Dataset):
    def __init__(self, features, train_data, batch_size=128):
        super(DGLData, self).__init__()
        self.sources = train_data.sources
        self.destinations = train_data.destinations
        self.timestamps = train_data.timestamps
        self.edge_idxs = train_data.edge_idxs
        self.labels = train_data.labels
        self.dst_labels = train_data.dst_labels
        self.src_ids = train_data.src_ids
        self.n_interactions = len(self.sources)
        self.unique_nodes = set(self.sources) | set(self.destinations)
        self.n_unique_nodes = len(self.unique_nodes)

        self.features = features.reset_index()
        self.batch_size = batch_size
        # self.length = int(self.n_interactions / self.BATCH_SIZE) + 1
        self.graphs = []

        self.initialize()

    def initialize(self):
        unique_time = np.unique(self.timestamps)
        for t in unique_time:
            t_mask = self.timestamps == t
            sources, destinations = self.sources[t_mask], self.destinations[t_mask]
            edge_idxs = self.edge_idxs[t_mask]
            timestamps = self.timestamps[t_mask]
            labels = self.labels[t_mask]
            dst_labels = self.dst_labels[t_mask]
            src_ids = self.src_ids[t_mask]
            features = self.features[self.features['timestamp'] == t].drop('timestamp', axis=1).set_index('role_id')

            self._np_to_graph(sources, destinations, labels, dst_labels,
                              edge_idxs, timestamps, src_ids, features)

    def __len__(self):
        return self.n_batch

    def __getitem__(self, item):
        return self.graphs[item]

    def collate_fn(self, data):
        return data

    def _np_to_graph(self, src, dst, label, dst_label, edge, ts, src_ids, features):
        # node re-mapping
        nodes = np.unique(np.concatenate([src, dst]))
        node_to_index = {node: index for index, node in enumerate(nodes)}
        n2i_fun = np.vectorize(node_to_index.get)
        src = n2i_fun(src)
        dst = n2i_fun(dst)
        assert np.concatenate([src, dst]).max() == len(node_to_index) - 1

        tmp_np = np.concatenate(
            [np.concatenate([src, dst], axis=0)[:, None], np.concatenate([label, dst_label], axis=0)], axis=1)
        labels = pd.DataFrame(tmp_np, columns=['role_id', 0, 1]).drop_duplicates()
        labels = labels.set_index('role_id')
        labels = labels.loc[n2i_fun(nodes)]

        features = features.loc[nodes]
        features.index = features.index.map(node_to_index)

        # # label to one hot
        # n_classes = len(np.unique(labels))
        # one_hot = np.eye(n_classes)
        # labels = pd.DataFrame(one_hot[labels.values].squeeze(),
        #                       index=labels.index, columns=[0, 1])

        # transform to CSR matrix
        n_nodes = len(nodes)
        row_indices = src
        col_indices = dst
        data = np.ones(len(row_indices), dtype=int)
        csr_graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
        # csr_graph = csr_graph + csr_graph.transpose()
        csr_graph.data[:] = 1

        # transform to DGL graph
        self.graph = dgl.from_scipy(csr_graph)
        node_features = torch.from_numpy(features.to_numpy().astype(np.float32))
        node_labels = torch.from_numpy(labels.to_numpy().astype(np.float32))  # int
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.ndata['role_id'] = torch.from_numpy(nodes.astype(np.int32))

        node_index = labels[labels[1]==1].index.to_numpy()
        # node_index = labels.index.to_numpy()
        n_batch = int(np.ceil(len(node_index) / self.batch_size))

        for g_id in range(n_batch):
            g_nodes = node_index[g_id * self.batch_size: (g_id + 1) * self.batch_size]
            g = self.graph.out_subgraph(g_nodes, relabel_nodes=True,
                                        store_ids=True, output_device='cpu')  # raw IDs are store in g.ndata['_ID']
            g = dgl.add_self_loop(g)
            # print(sum(g.ndata['label'][:,1]==1)/g.num_nodes())
            if g.num_nodes() == 0:
                continue
            self.graphs.append(g)

        self.n_batch = len(self.graphs)


class DGLDataInf(Dataset):
    def __init__(self, features, train_data, batch_size=128):
        super(DGLDataInf, self).__init__()
        self.sources = train_data.sources
        self.destinations = train_data.destinations
        self.timestamps = train_data.timestamps
        self.edge_idxs = train_data.edge_idxs
        # self.labels = train_data.labels
        # self.dst_labels = train_data.dst_labels
        self.src_ids = train_data.src_ids
        self.n_interactions = len(self.sources)
        self.unique_nodes = set(self.sources) | set(self.destinations)
        self.n_unique_nodes = len(self.unique_nodes)

        self.features = features.reset_index()
        self.batch_size = batch_size
        self.graphs = []

        self.initialize()

    def initialize(self):
        unique_time = np.unique(self.timestamps)
        for t in unique_time:
            t_mask = self.timestamps == t
            sources, destinations = self.sources[t_mask], self.destinations[t_mask]
            edge_idxs = self.edge_idxs[t_mask]
            timestamps = self.timestamps[t_mask]
            # labels = self.labels[t_mask]
            # dst_labels = self.dst_labels[t_mask]
            src_ids = self.src_ids[t_mask]
            features = self.features[self.features['timestamp'] == t].drop('timestamp', axis=1).set_index('role_id')

            self._np_to_graph(sources, destinations, edge_idxs, timestamps, src_ids, features)

    def __len__(self):
        return self.n_batch

    def __getitem__(self, item):
        return self.graphs[item]

    def collate_fn(self, data):
        return data

    def _np_to_graph(self, src, dst, edge, ts, src_ids, features):
        # node re-mapping
        nodes = np.unique(np.concatenate([src, dst]))
        node_to_index = {node: index for index, node in enumerate(nodes)}
        n2i_fun = np.vectorize(node_to_index.get)
        src = n2i_fun(src)
        dst = n2i_fun(dst)
        assert np.concatenate([src, dst]).max() == len(node_to_index) - 1

        features = features.loc[nodes]
        features.index = features.index.map(node_to_index)

        # transform to CSR matrix
        n_nodes = len(nodes)
        row_indices = src
        col_indices = dst
        data = np.ones(len(row_indices), dtype=int)
        csr_graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
        csr_graph = csr_graph + csr_graph.transpose()
        csr_graph.data[:] = 1

        # transform to DGL graph
        self.graph = dgl.from_scipy(csr_graph)
        node_features = torch.from_numpy(features.to_numpy().astype(np.float32))
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['role_id'] = torch.from_numpy(nodes.astype(np.int32))

        node_index = n2i_fun(nodes)
        n_batch = int(np.ceil(len(node_index) / self.batch_size))

        for g_id in range(n_batch):
            g_nodes = node_index[g_id * self.batch_size: (g_id + 1) * self.batch_size]
            g = self.graph.out_subgraph(g_nodes, relabel_nodes=True,
                                        store_ids=True, output_device='cpu')  # raw IDs are store in g.ndata['_ID']
            g = dgl.add_self_loop(g)
            if g.num_nodes() == 0:
                continue
            self.graphs.append(g)

        self.n_batch = len(self.graphs)


class StaticTwitterDGL():
    def __init__(self, features, labels, edges, batch_size=64, split_test=True,
                 graph_file='./data_processing/DGLgraph/twitter_graph.bin', sampling=True):
        '''
        Twitter dataset
        :param features: DataFrame with columns [role_id, ...]
        :param labels: DataFrame with columns [role_id, label]
        :param graphs: DataFrame with columns [role_id, t_role_id]
        '''
        super(StaticTwitterDGL, self).__init__()
        self.batch_size = batch_size
        # self.features = features
        # self.labels = labels
        # self.edges = edges
        self.split_test = split_test
        self.sampling = sampling
        self.graphs = []

        if type(edges) is dict:
            for week in sorted(edges.keys()):
                if not week in labels: continue
                wedges, wlabels, wfeatures = edges[week], labels[week], features[week]
                self.df_to_graph_v2(wedges, wlabels, wfeatures, graph_file)
        else:
            self.df_to_graph_v2(edges, labels, features, graph_file)

        dgl.save_graphs(graph_file, self.graphs)
        print('num of batch is %d' % self.n_batch)

    def df_to_graph_v2(self, df, labels, features, path):
        '''
        :param df: x, y in each row
        '''
        # node re-mapping
        nodes = np.unique(df)
        self.node_to_index = {node: index for index, node in enumerate(nodes)}
        x_col, y_col = df.columns[0], df.columns[1]
        df.loc[:, x_col] = df[x_col].map(self.node_to_index)
        df.loc[:, y_col] = df[y_col].map(self.node_to_index)
        assert df.values.max() == len(self.node_to_index) - 1

        # check feature missing
        totalset = set(nodes.tolist()) | set(features.index.tolist())
        if len(totalset) > features.shape[0]:
            diff = totalset - set(features.index.tolist())
            print('------feature missing for %d nodes------' % len(diff))
            diff_df = pd.DataFrame(np.zeros([len(diff), features.shape[1]]), index=list(diff), columns=features.columns)
            features = pd.concat([features, diff_df])

        features = features.loc[nodes]
        labels = labels.loc[nodes]
        features.index = features.index.map(self.node_to_index)
        labels.index = labels.index.map(self.node_to_index)

        # label to one hot
        n_classes = len(np.unique(labels))
        one_hot = np.eye(n_classes)
        labels = pd.DataFrame(one_hot[labels.values].squeeze(),
                              index=labels.index, columns=[0, 1])

        # transform to CSR matrix
        n_nodes = len(nodes)
        row_indices = df[x_col].to_numpy()
        col_indices = df[y_col].to_numpy()
        data = np.ones(len(row_indices), dtype=int)
        csr_graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
        csr_graph = csr_graph + csr_graph.transpose()
        csr_graph.data[:] = 1

        # transform to DGL graph
        self.graph = dgl.from_scipy(csr_graph)
        node_features = torch.from_numpy(features.to_numpy().astype(np.float32))
        node_labels = torch.from_numpy(labels.to_numpy().astype(np.float32))  # int
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.ndata['role_id'] = torch.from_numpy(nodes.astype(np.int32))

        # split nodes, train:test:val=8:1:1
        node_index = labels[labels[1] == 1].index.to_numpy()
        n_batch = int(np.ceil(len(node_index) / self.batch_size))

        # self.graphs = []
        # node_index = self.graph.nodes().numpy()
        np.random.shuffle(node_index)
        for g_id in range(n_batch):
            g_nodes = node_index[g_id * self.batch_size: (g_id + 1) * self.batch_size]
            g = self.graph.out_subgraph(g_nodes, relabel_nodes=True,
                                        store_ids=True)  # raw IDs are store in g.ndata['_ID']
            # g_nodes = self.graph.nodes()

            # g_mask = np.zeros(n_nodes, dtype=bool)
            # g_mask[g_nodes] = 1

            # g = self.graph.subgraph(g_nodes)
            # g = dgl.add_self_loop(g)
            # g.ndata['feat'] = node_features[g_mask]
            # g.ndata['label'] = node_labels[g_mask]

            pos_num = sum(g.ndata['label'].argmax(axis=1) == 1)
            pos_ratio = pos_num / g.num_nodes()
            expected_ratio = 0.3
            if pos_ratio < expected_ratio and self.sampling:
                neg_idx = g.ndata['label'].argmax(axis=1) == 0
                neg_nodes = g.ndata['_ID'][neg_idx]
                sampled_num = int(pos_num * (1 / expected_ratio - 1))
                sampled_nodes = neg_nodes[torch.randperm(len(neg_nodes))[:sampled_num]]

                pos_nodes = g.ndata['_ID'][~neg_idx]
                all_nodes = torch.cat([pos_nodes, sampled_nodes])
                assert np.isin(g_nodes, all_nodes.numpy()).any()
                g = self.graph.subgraph(all_nodes, relabel_nodes=True,
                                        store_ids=True)
                sampled_pos_ratio = sum(g.ndata['label'].argmax(axis=1) == 1) / g.num_nodes()
                print('p/n: %f < %f => p/n: %f' % (pos_ratio, expected_ratio, sampled_pos_ratio))
            else:
                print('p/n: %f' % (pos_ratio))

            g = dgl.add_self_loop(g)
            self.graphs.append(g)

        self.n_batch = len(self.graphs)

    def get_split(self, mode='train'):
        if mode == 'inference':
            return DGLDataset(self.graphs, mode, self.batch_size)
        self.n_batch = len(self.graphs)
        assert self.n_batch >= 10, 'n_batch=%d<=10' % self.n_batch
        n_train = int(self.n_batch * 0.8)
        if self.split_test:
            n_val = int(self.n_batch * 0.1)
        else:
            n_val = int(self.n_batch * 0.2)

        if mode == "val":
            lo, hi = n_train, n_train + n_val
        elif mode == "test":
            if self.split_test:
                lo, hi = n_train + n_val, self.n_batch
            else:
                lo, hi = n_train, n_train + n_val
        elif mode == 'train':
            lo, hi = 0, n_train

        graph = self.graphs[lo:hi]
        return DGLDataset(graph, mode, self.batch_size)


class DynamicMultiTwitterDGL():
    def __init__(self, features, labels, edge_dict, batch_size=64,
                 one_hot=True, split_test=True, graph_path='data_processing/DGLMultiGraph',
                 graph_save=True, split_method='time', batch_method='edge'):
        '''
        Twitter dataset
        :param features: DataFrame with columns [role_id, ...]
        :param labels: DataFrame with columns [role_id, label]
        :param graphs: DataFrame with columns [role_id, t_role_id]
        '''
        super(DynamicMultiTwitterDGL, self).__init__()
        self.x_col, self.y_col = 'role_id', 't_role_id'
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.features = features
        self.labels = labels
        # self.edge_dict = edge_dict
        self.split_test = split_test
        self.split_method = split_method
        self.batch_method = batch_method
        self.graph_path = graph_path
        self.graph_save = graph_save

        if self.graph_save:
            self.graph_init(edge_dict)

    def graph_init(self, edge_dict):
        total_df = self.concat_df(edge_dict.copy())
        print('df concat done')
        # split df
        if self.split_method == 'time':
            _split_f = self.split_by_time
        elif self.split_method == 'tgn':
            _split_f = self.split_by_tgn
        else:
            raise NotImplementedError
        self.split_dataset, self.timestamps, self.etypes, self.nodes = \
            self.split_by_time(total_df)
        for k, v in self.split_dataset.items():
            save_path = os.path.join(self.graph_path, f'{k}.csv')
            v.to_csv(save_path)
            print(f'save {save_path}')

        # df to batch graphs
        self.split_graphs = {}
        if self.batch_method == 'node':
            _batch_f = self.node_batch_df
        elif self.batch_method == 'edge':
            _batch_f = self.edge_batch_df
        elif self.batch_method == 'time':
            _batch_f = self.time_batch_df
        elif self.batch_method == 'label':
            _batch_f = self.label_batch_df
        else:
            raise NotImplementedError

        for k, v in self.split_dataset.items():
            self.split_graphs[k] = _batch_f(v)
            save_path = os.path.join(self.graph_path, f'{k}_graph.bin')
            dgl.save_graphs(save_path, self.split_graphs[k])
            print(f'save {save_path}')

    def concat_df(self, edge_dict, columns=['role_id', 't_role_id', 'etype', 'timestamp']):
        total_df = pd.DataFrame([], columns=columns)
        for etype, edges in edge_dict.items():
            for timestamp in sorted(edges.keys()):
                t_edge = edges[timestamp]
                t_edge.loc[:, 'etype'] = etype
                t_edge.loc[:, 'timestamp'] = timestamp
                total_df = pd.concat([total_df, t_edge], axis=0)
        print('------concat edges------')

        return total_df

    def df_to_heterograph(self, df):
        '''
        :param df: x, y in each row
        '''
        df = df.copy()
        # transform to DGL hetorograph
        timestamps = np.unique(df['timestamp'])
        etypes = np.unique(df['etype'])
        dgl_dict = {}
        dgl_node = {}
        dgl_feat = {}
        dgl_lab = {}
        for ts in timestamps:
            t_df = df[df['timestamp'] == ts]
            t_df, t_node = self.reindex(t_df)
            dgl_node[ts] = t_node
            dgl_feat[ts] = self.features[ts].loc[t_node].values
            dgl_lab[ts] = self.labels[ts].loc[t_node].values
            for et in etypes:
                _df = t_df[t_df['etype'] == et]
                dgl_dict[('node_t%d' % ts, '%s_t%d' % (et, ts), 'node_t%d' % ts)] = \
                    (torch.from_numpy(_df[self.x_col].values), torch.from_numpy(_df[self.y_col].values))

        graph = dgl.heterograph(dgl_dict)

        for ts in timestamps:
            graph.nodes[f'node_t{ts}'].data['node'] = torch.from_numpy(dgl_node[ts].astype(np.int64))
            graph.nodes[f'node_t{ts}'].data['feat'] = torch.from_numpy(dgl_feat[ts].astype(np.float32))
            graph.nodes[f'node_t{ts}'].data['label'] = torch.from_numpy(dgl_lab[ts].astype(np.int64))

        return graph

    def split_by_time(self, df):
        '''
        follow TGN to split batch graphs
        '''
        random.seed(2020)

        timestamps = np.unique(df['timestamp'])
        etypes = np.unique(df['etype'])
        nodes = np.unique(df[[self.x_col, self.y_col]])
        if self.split_test:
            val_time, test_time = int(len(timestamps) * 0.8), int(len(timestamps) * 0.9)
        else:
            val_time, test_time = int(len(timestamps) * 0.8), int(len(timestamps) * 0.8)
        train_df = df[df['timestamp'] < val_time]
        val_df = df[(df['timestamp'] >= val_time) & (df['timestamp'] < test_time)]
        test_df = df[df['timestamp'] >= test_time]

        return {'full_df': df, 'train_df': train_df, 'val_df': val_df, 'test_df': test_df}, \
            timestamps, etypes, nodes

    def split_by_tgn(self, df, different_new_nodes_between_val_and_test=False):
        timestamps = np.unique(df['timestamp'])
        etypes = np.unique(df['etype'])
        nodes = np.unique(df[[self.x_col, self.y_col]])
        if self.split_test:
            val_time, test_time = int(len(timestamps) * 0.8), int(len(timestamps) * 0.9)
        else:
            val_time, test_time = int(len(timestamps) * 0.8), int(len(timestamps) * 0.8)

        sources = df['role_id'].values
        destinations = df['t_role_id'].values
        edge_idxs = df.index.values
        timestamps = df['timestamp'].values

        full_df = df

        random.seed(2020)

        node_set = set(sources) | set(destinations)
        n_total_unique_nodes = len(node_set)

        # Compute nodes which appear at test time
        test_node_set = set(sources[timestamps > val_time]).union(
            set(destinations[timestamps > val_time]))
        # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
        # their edges from training
        new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

        # Mask saying for each source and destination whether they are new test nodes
        new_test_source_mask = df[self.x_col].index.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = df[self.y_col].index.map(lambda x: x in new_test_node_set).values

        # Mask which is true for edges with both destination and source not being new test nodes (because
        # we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # For train we keep edges happening before the validation time which do not involve any new node
        # used for inductiveness
        train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

        train_df = df[train_mask]

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_df.sources).union(train_df.destinations)
        assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        if different_new_nodes_between_val_and_test:
            n_new_nodes = len(new_test_node_set) // 2
            val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
            test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

            edge_contains_new_val_node_mask = np.array(
                [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
            edge_contains_new_test_node_mask = np.array(
                [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


        else:
            edge_contains_new_node_mask = np.array(
                [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # validation and test with all edges
        val_df = df[val_mask]

        test_df = df[test_mask]

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = df[new_node_val_mask]

        new_node_test_data = df[new_node_test_mask]

        print("The dataset has {} interactions, involving {} different nodes".format(len(full_df),
                                                                                     len(np.unique(full_df))))
        print("The training dataset has {} interactions, involving {} different nodes".format(
            len(train_df), len(np.unique(train_df))))
        print("The validation dataset has {} interactions, involving {} different nodes".format(
            len(val_df), len(np.unique(val_df))))
        print("The test dataset has {} interactions, involving {} different nodes".format(
            len(test_df), len(np.unique(test_df))))
        print("The new node validation dataset has {} interactions, involving {} different nodes".format(
            len(new_node_val_data), len(np.unique(new_node_val_data))))
        print("The new node test dataset has {} interactions, involving {} different nodes".format(
            len(new_node_test_data), len(np.unique(new_node_test_data))))
        print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
            len(new_test_node_set)))

        # return full_df, train_df, val_df, test_df, \
        #     new_node_val_data, new_node_test_data, timestamps, etypes, nodes
        return {'full_df': full_df, 'train_df': train_df, 'val_df': new_node_val_data, 'test_df': new_node_test_data}, \
            timestamps, etypes, nodes

    def edge_batch_df(self, df):
        arr = df.index.to_numpy()
        np.random.shuffle(arr)
        n_batch = len(arr) // self.batch_size
        graphs = []
        for i in range(n_batch):
            iidx = arr[i * self.batch_size: (i + 1) * self.batch_size]
            idf = df.loc[iidx]
            igraph = self.df_to_heterograph(idf)
            graphs.append(igraph)
        return graphs

    def node_batch_df(self, df):
        arr = np.arange(len(self.nodes))
        np.random.shuffle(arr)
        n_batch = len(arr) // self.batch_size
        graphs = []
        for i in range(n_batch):
            inodes = arr[i * self.batch_size: (i + 1) * self.batch_size]
            umask = df[self.x_col].map(lambda x: x in inodes)
            vmask = df[self.y_col].map(lambda x: x in inodes)
            edge_mask = np.logical_and(umask, vmask)
            idf = df[edge_mask]
            igraph = self.df_to_heterograph(idf)
            graphs.append(igraph)
        return graphs

    def time_batch_df(self, df):
        '''
        TODO: left implementation for time-continuous temporal GNN
        '''
        raise NotImplementedError

    def label_batch_df(self, df):
        nodes = df[df['label'] == 1]
        arr = nodes.index.to_numpy()
        np.random.shuffle(arr)
        n_batch = len(arr) // self.batch_size
        graphs = []
        for i in range(n_batch):
            inodes = arr[i * self.batch_size: (i + 1) * self.batch_size]
            umask = df[self.x_col].map(lambda x: x in inodes)
            vmask = df[self.y_col].map(lambda x: x in inodes)
            edge_mask = np.logical_and(umask, vmask)
            idf = df[edge_mask]
            igraph = self.df_to_heterograph(idf)
            graphs.append(igraph)
        return graphs

    def get_split(self, mode):
        '''
        :param mode: full, train, val, test, etc
        :return: DGL heterograph list
        '''
        load_path = os.path.join(self.graph_path, f'{mode}_df_graph.bin')
        self.graphs = dgl.load_graphs(load_path)

        return DGLDataset(self.graphs, mode, self.batch_size)

    def get_split_df(self, mode):
        '''
        :param mode: full, train, val, test, etc
        :return: DGL heterograph list
        '''
        load_path = os.path.join(self.graph_path, f'{mode}_df.csv')
        df = pd.read_csv(load_path, index_col=0)

        return df

    def get_split_data(self, mode):
        '''
        :param mode: full, train, val, test, etc
        :return: DGL heterograph list
        '''
        # read edges
        load_path = os.path.join(self.graph_path, f'{mode}_df.csv')
        df = pd.read_csv(load_path, index_col=0)
        # filter
        max_train_time = df['timestamp'].max()
        features = dict(filter(lambda x: x[0] <= max_train_time, self.features.items()))
        labels = dict(filter(lambda x: x[0] <= max_train_time, self.labels.items()))
        t_feat, ts, t_lab = list(features.values()), list(features.keys()), list(labels.values())
        # timestamp
        ts_df = np.concatenate([np.ones([len(t_feat[i]), 1]) * item for i, item in enumerate(ts)])
        # index
        t_index = np.concatenate([item.index.to_numpy() for item in t_feat])[:, None]
        # feature concat
        t_feat = np.concatenate(t_feat, axis=0)
        feat_df = np.concatenate([t_feat, ts_df, t_index], axis=1)
        feat_df = pd.DataFrame(feat_df, columns=self.features[0].columns.tolist() + ['timestamp', 'role_id'])
        feat_df = feat_df.set_index(['role_id', 'timestamp'])
        # label concat
        t_lab = np.concatenate(t_lab, axis=0)
        lab_df = np.concatenate([t_lab, ts_df, t_index], axis=1)
        lab_df = pd.DataFrame(lab_df, columns=['label', 'timestamp', 'role_id'])
        lab_df = lab_df.set_index(['role_id', 'timestamp'])
        # etype
        etype_mapping = {'strong': 0, 'weak': 1}
        etype = df['etype'].map(etype_mapping).values

        data = TGNData(df['role_id'].values, df['t_role_id'].values, df['timestamp'].values,
                       df.index.values, etype, lab_df, feat_df)

        return data

    def get_num_nodes(self):
        df = self.get_split_df('full')
        return len(np.unique(df[[self.x_col, self.y_col]]))

    def reindex(self, df, ):
        '''
        build a static graph
        :param df: edges
        :param features: features
        :param labels: labels
        :return: u, v, features, labels, node_index
        '''
        # node re-mapping
        df = df.copy()
        nodes = np.unique(df[[self.x_col, self.y_col]])
        node_to_index = {node: index for index, node in enumerate(nodes)}
        df.loc[:, self.x_col] = df[self.x_col].map(node_to_index)
        df.loc[:, self.y_col] = df[self.y_col].map(node_to_index)
        assert df[[self.x_col, self.y_col]].values.max() == len(node_to_index) - 1

        return df, nodes