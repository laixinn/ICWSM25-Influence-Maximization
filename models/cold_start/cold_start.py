import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import dgl
import torch
import concurrent.futures
import itertools

from data_processing.read_tweet import get_dataset, get_dataset_inference, get_netease_dataset, dict_to_df, read_nodes, \
    build_tgraph
from Trie import StringTrie

timestamps = None

def load_data(platform='netease'):
    if platform == 'local':
        data = get_dataset_inference(path='tweet_week', filter_feature=True)
    else:
        # data = get_netease_inference(path='netease_week', filter_feature=True)
        pass
    weak_edges, strong_edges, features = [dict_to_df(item) for item in data]

    # reindex
    unique_nodes = np.unique(weak_edges[['role_id', 't_role_id']])
    node_to_index = {node: i for i, node in enumerate(unique_nodes)}
    weak_edges.loc[:, 'role_id'] = weak_edges['role_id'].map(node_to_index)
    weak_edges.loc[:, 't_role_id'] = weak_edges['t_role_id'].map(node_to_index)

    timestamps = np.unique(weak_edges['timestamp'])
    graphs = {}
    for t in timestamps:
        tedges = weak_edges[['role_id', 't_role_id']][weak_edges['timestamp'] == t]
        dgraph = dgl.graph((tedges['role_id'].to_numpy(), tedges['t_role_id'].to_numpy()))
        # dgraph = dgl.add_self_loop(dgraph)
        dgraph = dgl.add_reverse_edges(dgraph)
        graphs[t] = dgraph

    return graphs, features, node_to_index


def lookup_neighbor(graph, node, out_degrees):
    if node >= len(out_degrees):
        return '-1', None
    d1 = out_degrees[node]
    if d1 == 0:
        return '-1', None
    all_n1 = np.unique(graph.out_edges(node)[1].numpy())
    n1d_list = out_degrees[all_n1]

    unique_n1, unique_count = np.unique(n1d_list, return_counts=True)
    _ind = np.argsort(unique_count)[-3:][::-1]
    unique_n1 = unique_n1[_ind]
    unique_n1 = np.minimum(unique_n1, 99)
    unique_count = unique_count[_ind]
    unique_count = np.minimum(unique_count, 99)
    # if unique_n1[0] == 0:
    #     return '-1', None
    assert unique_n1[0] != 0, unique_n1

    node_string = [np.minimum(d1, 99)] + np.stack([unique_n1, unique_count], axis=1).flatten().tolist()

    padding = 7 - len(node_string)
    node_string.extend([0] * padding)
    # print(node_string)
    node_string = '_'.join(map(lambda x: '%02d' % x, node_string))
    return node_string, all_n1


def graph_encoding(graphs, node_to_index):
    str_list = []
    node_list = []
    time_list = []

    index_to_node = {val: key for key, val in node_to_index.items()}

    timestamps = list(graphs.keys())
    timestamps.sort()
    timestamps.reverse()
    for cur_t in timestamps:
        if cur_t < 2:
            continue
        cur_node = graphs[cur_t].nodes()
        cur_deg = graphs[cur_t].out_degrees().numpy()
        print(f'cold node {sum(cur_deg==1)}, {sum(cur_deg>0)} at {cur_t}')
        cur_node = cur_node[(cur_deg > 0) & (cur_deg <= 1)]
        print(f'cur_node {len(cur_node)} at {cur_t}')
        cnt_nnbr = 0
        # before_cnt = 0
        final_cnt = 0
        for n in cur_node:
            str1, n_nbr = lookup_neighbor(graphs[cur_t], n, graphs[cur_t].out_degrees().numpy())
            if str1 == '-1':
                continue
            assert len(str1) == 20, str1
            # print('t1:', str1)
            # before_cnt += len(n_nbr)

            # different max id across time
            t1_deg = graphs[cur_t - 1].out_degrees().numpy()
            n_nbr = n_nbr[n_nbr < len(t1_deg)]
            n_nbr = n_nbr[t1_deg[n_nbr] > 0]

            cnt_nnbr += len(n_nbr)
            for n1 in n_nbr:
                str2, n1_nbr = lookup_neighbor(graphs[cur_t - 1], n1, t1_deg)
                if str2 == '-1':
                    continue
                assert len(str2) == 20, str1
                # print('t2:', str2)

                # different max id across time
                t2_deg = graphs[cur_t - 2].out_degrees().numpy()
                n1_nbr = n1_nbr[n1_nbr < len(t2_deg)]
                n1_nbr = n1_nbr[t2_deg[n1_nbr] > 0]
                final_cnt += len(n1_nbr)

                for n2 in n1_nbr:
                    str3, n2_nbr = lookup_neighbor(graphs[cur_t - 2], n2, t2_deg)
                    if str3 == '-1':
                        continue
                    assert len(str3) == 20, str1

                    final_str = str1 + '_' + str2 + '_' + str3
                    assert len(final_str) == 62, final_str
                    # print(final_str)
                    str_list.append(final_str)

                    # index to node id
                    node_list.append((index_to_node[n.item()], \
                                      index_to_node[n1], \
                                      index_to_node[n2]))
                    # node_list.append((n.item(), n1, n2))

                    time_list.append((cur_t, cur_t - 1, cur_t - 2))
        # print('before n_nbr', before_cnt)
    #     print('cnt n_nbr', cnt_nnbr)
    #     print('final cnt', final_cnt)
    # print(len(str_list))
    return np.array(str_list), np.array(node_list), np.array(time_list)


def path_encoding(cold_paths, graphs, node_to_index):
    str_list = []
    node_list = []
    time_list = []

    index_to_node = {val: key for key, val in node_to_index.items()}
    global timestamps
    timestamps = sorted(np.unique(cold_paths[['week_dt_AB', 'week_dt_BC']]))
    t2i = {val: key for key, val in enumerate(timestamps)}
    assert len(t2i) == len(graphs), 'number of cold_path timestamps is different from graphs'

    cold_paths = cold_paths[cold_paths[['A', 'B', 'C']].isin(list(node_to_index.keys())).all(1)]

    for t, cur_t in list(t2i.items())[1:len(graphs) - 1]:
        t_cold_paths = cold_paths[cold_paths['week_dt_AB'] == t]
        prev_t = list(t2i.values())[cur_t - 1]
        next_t = list(t2i.values())[cur_t + 1]
        t_cold_paths['Ai'] = t_cold_paths['A'].map(node_to_index)
        t_cold_paths['Bi'] = t_cold_paths['B'].map(node_to_index)
        t_cold_paths['Ci'] = t_cold_paths['C'].map(node_to_index)

        amap = {}
        for ai in np.unique(t_cold_paths['Ai']):
            stra, a_nbr = lookup_neighbor(graphs[prev_t], ai, graphs[prev_t].out_degrees().numpy())
            '''only preserve edges for cold-start nodes'''
            if (not a_nbr is None) and (len(a_nbr) != 1):
                stra = '-1'
            amap[ai] = stra

        bmap = {}
        for bi in np.unique(t_cold_paths['Bi']):
            strb, b_nbr = lookup_neighbor(graphs[cur_t], bi, graphs[cur_t].out_degrees().numpy())
            bmap[bi] = strb

        cmap = {}
        for ci in np.unique(t_cold_paths['Ci']):
            strc, c_nbr = lookup_neighbor(graphs[next_t], ci, graphs[next_t].out_degrees().numpy())
            cmap[ci] = strc

        t_cold_paths['Ai'] = t_cold_paths['Ai'].map(amap)
        t_cold_paths['Bi'] = t_cold_paths['Bi'].map(bmap)
        t_cold_paths['Ci'] = t_cold_paths['Ci'].map(cmap)
        t_cold_paths = t_cold_paths[(t_cold_paths[['Ai', 'Bi', 'Ci']] != '-1').all(1)]
        t_cold_paths['final_str'] = t_cold_paths['Ai'] + '_' + t_cold_paths['Bi'] + '_' + t_cold_paths['Ci']

        str_list += t_cold_paths['final_str'].to_list()
        node_list += list(t_cold_paths[['C', 'B', 'A']].itertuples(index=False, name=None))
        time_list += [(next_t, cur_t, prev_t) for i in range(len(t_cold_paths))]

    return np.array(str_list), np.array(node_list), np.array(time_list)


def generate_cold_edges(cold_path=None, platform='local'):
    graphs, features, node_to_index = load_data(platform)

    '''graph structure to string'''
    print('start graph encoding')
    if cold_path is None:
        str_list, node_list, time_list = graph_encoding(graphs, node_to_index)
    else:
        str_list, node_list, time_list = path_encoding(cold_path, graphs, node_to_index)
    # np.save('labeling/str_list.npy', str_list)
    # np.save('labeling/node_list.npy', node_list)
    # np.save('labeling/time_list.npy', time_list)
    # str_list, node_list, time_list = np.load('labeling/str_list.npy'), np.load('labeling/node_list.npy'), np.load('labeling/time_list.npy')
    print(f"Encode {len(str_list)} cold paths")

    '''only preserve edges for cold-start nodes'''
    # index_list = []
    # for i in range(len(str_list)):
    #     n0, t0 = node_list[i][0], time_list[i][0]
    #     if(graphs[t0].out_degrees(n0) == 1):
    #         index_list.append(i)
    # str_list, time_list, node_list = str_list[index_list], time_list[index_list], node_list[index_list]
    # print(f'{len(str_list)} cold-start nodes attach new edges.')

    '''insert nodes to Trie'''
    nodes = np.unique(node_list)
    trie = StringTrie(len(node_list), time_list, node_list)
    for i, s in enumerate(str_list):
        trie.insertV2(s, i)
    print(f'Trie has {len(nodes)} nodes, encoding {len(trie.data_index)} entity.')

    '''Pre-order traverse'''
    pre_arr = trie.preOrderTraverse()

    '''attach edge by their near neighbor'''
    cold_edges = None
    for i in range(len(str_list)):
        s, n, t = str_list[i], node_list[i], time_list[i]
        output, output_time = trie.searchNear(s, t[0], n[0], num=2, threshold=0.75)
        iedges = np.stack([np.ones_like(output) * n[0], output, np.ones_like(output) * t[0], output_time], axis=1)
        if cold_edges is None:
            cold_edges = iedges
        else:
            cold_edges = np.concatenate([cold_edges, iedges], axis=0)

    cold_edges_df = pd.DataFrame(cold_edges, columns=['role_id', 't_role_id', 'dt', 't_dt']).drop_duplicates(
        ignore_index=True)
    print('before pruning', cold_edges_df.shape)

    '''preserve only few nbr for each node'''
    left_node = cold_edges_df[['role_id', 'dt']].drop_duplicates().values
    keep_edge = 2
    keep_list = []
    for n, t in left_node:
        n_edge_index = cold_edges_df.index[(cold_edges_df['role_id'] == n) & (cold_edges_df['dt'] == t)].values
        shuf_index = np.arange(len(n_edge_index))
        np.random.shuffle(shuf_index)
        n_edge_index = n_edge_index[shuf_index[:keep_edge]]
        keep_list.append(n_edge_index)
    keep_list = np.concatenate(keep_list)
    cold_edges_df = cold_edges_df.loc[keep_list]
    print('after pruning', cold_edges_df.shape)
    '''group by and count'''
    print('average edge for each (dt, role_id)', cold_edges_df.groupby(['dt', 'role_id']).size().mean())

    '''check id existance'''
    unique_timestamps = np.unique(cold_edges_df['dt'].values)
    for t in unique_timestamps:
        '''for debug'''
        # debug_node = [features[features['timestamp']==t].index[0]]
        # drop_index = features[features['timestamp']==t].index[features[features['timestamp']==t].index.isin(debug_node)]
        # features = features.drop(drop_index, axis=0)

        t_colddf = cold_edges_df[cold_edges_df['dt'] == t]
        t_roleid = set(t_colddf['role_id'].values.tolist())
        t_troleid = set(t_colddf['t_role_id'].values.tolist())
        t_feat = set(features[features['timestamp'] == t].index.values.tolist())

        print(f'At time {t}')
        print(len(t_feat & t_roleid), len(t_feat), len(t_roleid))
        print(len(t_feat & t_troleid), len(t_feat), len(t_troleid))

        t_roleid = t_feat & t_roleid
        t_troleid = t_feat & t_troleid
        drop_index = t_colddf.index[(~t_colddf['role_id'].isin(t_roleid)) | (~t_colddf['t_role_id'].isin(t_troleid))]
        cold_edges_df = cold_edges_df.drop(drop_index, axis=0)
        assert len(t_feat & t_roleid) <= len(t_feat) and len(t_feat & t_roleid) == len(t_roleid) and \
               len(t_feat & t_troleid) <= len(t_feat) and len(t_feat & t_troleid) == len(t_troleid)

    '''save'''
    if platform == 'local':
        filepath = 'tweet_week'
    else:
        filepath = 'netease_week'
    # cold_edges_df.to_csv('labeling/cold_edges_df.csv')
    global timestamps
    for ut in unique_timestamps:
        t_df = cold_edges_df[cold_edges_df['dt'] == ut]
        if timestamps:
            ts = timestamps[ut]
        else:
            ts = ut
        ts = int(ts)
        # t_df = t_df[['dt','t_dt']].replace(timestamps)
        t_df.to_csv(f'data_processing/{filepath}/{ts}_cold.csv')
        print(f'save {t_df.shape[0]} edges at time {ts} to data_processing/{filepath}/{ts}_cold.csv')


def cold_node_statistics(platform='netease'):
    if platform == 'local':
        data = get_dataset(filter_feature=True, positive_subgraph=True)
    else:
        # data = get_netease_inference(path='netease_week', filter_feature=True)
        pass
    weak_edges, strong_edges, features = [dict_to_df(item) for item in data]

    # reindex
    unique_nodes = np.unique(strong_edges[['role_id', 't_role_id']])
    node_to_index = {node: i for i, node in enumerate(unique_nodes)}
    strong_edges.loc[:, 'role_id'] = strong_edges['role_id'].map(node_to_index)
    strong_edges.loc[:, 't_role_id'] = strong_edges['t_role_id'].map(node_to_index)

    timestamps = np.unique(strong_edges['timestamp'])
    graphs = {}
    for t in timestamps:
        tedges = strong_edges[['role_id', 't_role_id']][strong_edges['timestamp'] == t]
        dgraph = dgl.graph((tedges['role_id'].to_numpy(), tedges['t_role_id'].to_numpy()))
        # dgraph = dgl.add_self_loop(dgraph)
        dgraph = dgl.add_reverse_edges(dgraph)
        graphs[t] = dgraph

    timestamps = list(graphs.keys())
    timestamps.sort()
    timestamps.reverse()
    for cur_t in timestamps:
        cur_node = graphs[cur_t].nodes()
        cur_deg = graphs[cur_t].out_degrees().numpy()

        deg0 = sum(cur_deg == 0)
        print(f'cold node {sum(cur_deg <= 1) - deg0, sum(cur_deg <= 2) - deg0, sum(cur_deg <= 3) - deg0, sum(cur_deg <= 4) - deg0, sum(cur_deg <= 5) - deg0, max(cur_deg), np.mean(cur_deg), sum(cur_deg > 0)} at {cur_t}')


if __name__ == '__main__':
    path = pd.read_csv('data_processing/tweet_week/path.csv', index_col=0)
    generate_cold_edges(path)

