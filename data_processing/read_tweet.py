import os
import os.path as osp
import random
import time, datetime
import re
from itertools import count, islice
from collections import defaultdict, Counter
import pickle5 as pickle

import pandas as pd
import numpy as np
import math, argparse
from scipy.sparse import csr_matrix, triu

'''test'''
# class testClass:
#     def __init__(self, c):
#         self.c = c
#
#     def fun1(self, a):
#         return a+self.c
#
#     def fun2(self, a):
#         return a+3
#
#     def fun3(self, a):
#         return a
#
# class MyCSR:
#     def __init__(self, target_class):
#         self.target_class = target_class(10)
#         self.csr_dict = lambda x: x+1
#
#     def df_to_csr(self, df):
#         return None
#
#     def __getattr__(self, name):
#         def transformed_method(input_value):
#             map_value = self.csr_dict(input_value)
#             target_method = getattr(self.target_class, name)
#             return target_method(map_value)
#         return transformed_method
#
# obj = MyCSR(testClass)
# print(obj.fun1(1))
# print(obj.fun2(1))
# print(obj.fun3(1))
# print(1)

class CSRMatrix:
    def __init__(self, df, is_map=True):
        self.is_map = is_map
        self.df_to_csr(df, is_map)

    def df_to_csr(self, df, is_map):
        '''
        :param df: x, y in each row
        '''
        # node re-mapping
        nodes = np.unique(df)
        self.node_to_index = {node: index for index, node in enumerate(nodes)}
        self.index_to_node = {val:key for key, val in self.node_to_index.items()}
        if is_map:
            self.node_fun = np.vectorize(lambda x: self.node_to_index[x])
        else:
            self.node_fun = np.vectorize(lambda x: x)
        x_col, y_col = df.columns[0], df.columns[1]
        df.loc[:, x_col] = df[x_col].map(self.node_to_index)
        df.loc[:, y_col] = df[y_col].map(self.node_to_index)
        assert df.values.max() == len(self.node_to_index) - 1

        # transform to CSR matrix
        num_nodes = len(nodes)
        row_indices = df[x_col].to_numpy()
        col_indices = df[y_col].to_numpy()
        data = np.ones(len(row_indices), dtype=int)
        self.target_class = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

        # undirected
        self.target_class = self.target_class + self.target_class.transpose()
        self.target_class.data[:] = 1
        print('transform df to csr')

    def __getattr__(self, name):
        def transformed_method(input_value):
            if type(input_value) == np.ndarray:
                input_value = np.array(input_value)
            map_value = self.node_fun(input_value)
            target_method = getattr(self.target_class, name)
            return target_method(map_value).indices.tolist()
        return transformed_method

    def __getitem__(self, input_value):
        if type(input_value) == np.ndarray:
            input_value = np.array(input_value)
        map_value = self.node_fun(input_value)
        target_method = getattr(self.target_class, '__getitem__')
        return target_method(map_value).indices.tolist()


class read_tweet(object):
    def __init__(self, rootpath='./', min_tf=2):
        self.path_table = osp.join(rootpath, "WordTable.txt")
        self.rootpath = rootpath
        self.min_tf = min_tf
        if os.path.exists('./tweet_pair/node_id.pkl'):
            with open('./tweet_pair/node_id.pkl', 'rb') as f:
                self.name2node = pickle.load(f)
            self.index = max(list(self.name2node.values())) + 1
        else:
            self.name2node = {}
            self.index = 0
        # self.read_wordtable()

        # self.processed_file = ['tweet_result_15_.txt', 'tweet_result_3_.txt',
        #                        'tweet_result_6_.txt', 'tweet_result_5_.txt',
        #                        'tweet_result_8_.txt', 'tweet_result_16_.txt',
        #                        'tweet_result_9_.txt', 'tweet_result_17_.txt',
        #                        'tweet_result_4_.txt', 'tweet_result_19_.txt',
        #                        'tweet_result_12_.txt']
        self.processed_file = []

    def __iter_txtfiles(self, path="tweets-withoutwords/"):
        rootpath = osp.join(self.rootpath, path)
        print("Reading text files in:", rootpath)
        for root, dirs, files in os.walk(rootpath, topdown=True):
            for fname in filter(lambda fname: fname.endswith('_.txt') and fname not in self.processed_file, files):
                yield os.path.join(root, fname)

    # generator for reading text
    def __iter_text(self, path):
        with open(path, 'r') as f:
            for line in f:
                yield line.strip('\n')

    # called by read_wordtable
    # generator for reading wordtable
    def __iter_wordtable(self, min_tf=2):
        itr = self.__iter_text(self.path_table)
        next(itr)
        for text in itr:
            ID, tf, word = text.split('\t')
            ID, tf = map(int, [ID, tf])
            if tf > min_tf and tf < 10000:
                yield (ID, tf, word)

    # called by __init__
    def read_wordtable(self):
        print('begin reading WordTable')
        min_tf = self.min_tf
        # dic={'id':ID,'tf':tf,'word':word}
        self.id2tf = {}
        self.id2word = {}
        self.word2tf = {}
        self.id2newid = {}
        for i, tmp in enumerate(self.__iter_wordtable(min_tf=min_tf)):
            self.id2tf[tmp[0]] = tmp[1]
            self.id2word[tmp[0]] = tmp[2]
            self.word2tf[tmp[2]] = tmp[1]
            self.id2newid[tmp[0]] = i
        print('num_words:', len(self.id2tf))
        print('get wordtable dic:id2tf, word2tf, id2word')

    def __iter_text2(self, path):
        with open(path, 'rb') as f:
            for line in f:
                # yield str(line).lstrip("b'").rstrip("\\r\\n'").strip()
                yield line.decode('utf-8').rstrip('\r\n').strip()

    ## called by dic_contents
    def __iter_block(self, path):
        itr = self.__iter_text2(path)
        for text in itr:
            if not text == '':
                block = [text]  # block[0]=name
                for attr in itr:
                    block.append(re.split(' |\\\\t|\t', attr.strip()))
                    if len(block) > 7:
                        if len(block) == 9 + int(block[7][0]) and attr == '':
                            # timestamp = time.strptime(block[2][0]+' '+block[2][1]+' '+block[2][2]+' '+block[2][-1], '%a %b %d %Y')# reach the end of a tweet block
                            timestamp = block[2][0]+' '+block[2][1]+' '+block[2][2]+' '+block[2][-1]
                            rolename = block[0]
                            retweet_name = block[4][0]
                            reply_name = block[5][0]
                            yield (timestamp, rolename, retweet_name, reply_name)  # name & content
                            break

    def _ret_id(self, name):
        if name == '-1':
            return -1
        elif name in self.name2node:
            return self.name2node[name]
        else:
            self.name2node[name] = self.index
            self.index += 1
            return self.name2node[name]

    def dic_contents(self, num_files=None, end_date=None):
        min_date = None

        filefolder = islice(self.__iter_txtfiles(), 0, num_files)
        print('-----Read tweets .txt-----')
        start = time.perf_counter()
        for path in filefolder:
            node_pair = []
            print('Reading:', path)
            sample = islice(self.__iter_block(path), 0, None)
            for timestamp, rolename, retweet_name, reply_name in sample:
                if (retweet_name == '-1' and reply_name == '-1') or (datetime.datetime.strptime(timestamp, '%a %b %d %Y')>end_date):
                    continue

                roleid = self._ret_id(rolename)
                retweet_id = self._ret_id(retweet_name)
                reply_id = self._ret_id(reply_name)

                node_pair.append([timestamp, roleid, retweet_id, reply_id])

            print('processing time:', time.perf_counter() - start)
            # save
            with open(path.replace('tweets-withoutwords', 'tweet_pair').replace('.txt', '.pkl'), 'wb') as f:
                pickle.dump(node_pair, f)
            # with open(path.replace('tweets-withoutwords', 'tweet_pair').replace('.txt', '.pkl'), 'rb') as f:
            #     node_pair = pickle.load(f)
            with open('./tweet_pair/node_id.pkl', 'wb') as f:
                pickle.dump(self.name2node, f)
            # with open('./tweet_pair/node_id.pkl', 'rb') as f:
            #     name2node = pickle.load(f)

            # min date: 2010-01-01
            # _date = min(list(map(lambda x: datetime.datetime.strptime(x[0], '%a %b %d %Y'), node_pair)))
            # if min_date is None:
            #     min_date = _date
            # else:
            #     min_date = min(min_date, _date)
        print('-----End of reading tweets-----')
        # print('min date:', min_date)

    def __count_word(self, word_list, n=10):
        cont = Counter(word_list).most_common(n)
        # Replace name with node id
        cont = [(self.id2newid[cnt[0]], cnt[1]) for cnt in cont if cnt[0] in self.id2tf]
        return cont

class build_tgraph:
    def __init__(self, datapath='tweet_week', is_accumu=True):
        # self.rootpath = rootpath
        self.rootpath = os.path.dirname(os.path.abspath(__file__))
        self.datapath = datapath
        # with open(os.path.join(rootpath, 'node_id.pkl'), 'rb') as f:
        #     self.name2node = pickle.load(f)
        self.graph = None
        self.is_accumu=is_accumu

    def __iter_pickle(self, path='tweet_pair', condition=lambda fname: fname.startswith('tweet_result_') and fname.endswith('_.pkl')):
        rootpath = os.path.join(self.rootpath, path)
        print("Reading text files in:", rootpath)
        for root, dirs, files in os.walk(rootpath, topdown=True):
            for fname in filter(condition, files):
                yield os.path.join(root, fname)

    def _update(self, graph):
        if not self.is_accumu or self.graph is None:
            self.graph = graph
        else:
            self.graph = pd.concat([self.graph, graph], axis=0)
            if 'real_dt' in self.graph.columns:
                self.graph = self.graph.groupby(['dt', 'role_id', 't_role_id', 'real_dt']).sum().reset_index()
            else:
                self.graph = self.graph.groupby(['dt', 'role_id', 't_role_id']).sum().reset_index()

    def _update_v2(self, graph):
        if not self.is_accumu or self.graph is None:
            self.graph = graph
        else:
            self.graph = pd.concat([self.graph, graph], axis=0)
            self.graph = self.graph.groupby(['dt', 'role_id', 't_role_id', 'real_dt']).sum().reset_index()

    def _save(self, path='tweet_graph', right_str='graph'):
        timestamps = self.graph['dt'].unique()
        for ts in timestamps:
            tgraph = self.graph[self.graph['dt'] == ts]
            tpath = os.path.join(self.rootpath, '%s/%s_%s.csv' % (path, ts, right_str))
            tgraph.to_csv(tpath)

    def build_tgraph(self):
        filefolder = islice(self.__iter_pickle(), 0, None)

        print('-----Read tweets .pkl-----')
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            with open(path, 'rb') as f:
                node_pair = pickle.load(f)

            node_dict = []
            for np in node_pair:
                timestamp = datetime.datetime.strptime(np[0], '%a %b %d %Y').strftime('%Y%m%d')
                if np[2] != -1:
                    node_dict.append({'dt': timestamp, 'role_id': np[1], 't_role_id': np[2], 'weight': 1})
                if np[3] != -1:
                    node_dict.append({'dt': timestamp, 'role_id': np[1], 't_role_id': np[3], 'weight': 1})
            pgraph = pd.DataFrame(node_dict)
            graph = pgraph.groupby(['dt', 'role_id', 't_role_id']).sum().reset_index()

            self._update(graph)
            self._save()

            print('processing time:', time.perf_counter() - start)

            # min date: 2010-01-01
            # _date = min(list(map(lambda x: datetime.datetime.strptime(x[0], '%a %b %d %Y'), node_pair)))
            # if min_date is None:
            #     min_date = _date
            # else:
            #     min_date = min(min_date, _date)
        print('-----End of reading tweets-----')

    def graph_freqent(self, timestamp='20100101', path='tweet_graph'):
        tpath = os.path.join(self.rootpath, '%s/%s_graph.csv' % (path, timestamp))
        graph = pd.read_csv(tpath, index_col=0)
        median = graph[['weight']].value_counts().quantile()
        print('median:', median)
        print('greater than median:', graph[graph['weight'] >= median].shape[0]/graph.shape[0])

    def strong_tie(self, path='tweet_graph', start_date=datetime.datetime(2010, 1, 1), end_date=datetime.datetime(2010, 3, 1)):
        timespan = end_date - start_date
        for t in range(timespan.days+1):
            ts = start_date + datetime.timedelta(days=t)
            ts = ts.strftime('%Y%m%d')
            # find median weight
            tpath = os.path.join(self.rootpath, '%s/%s_graph.csv' % (path, ts))
            graph = pd.read_csv(tpath, index_col=0)
            median = graph[['weight']].value_counts().quantile()
            print('median:', median)
            # weight>median is the strong tie
            strong_tie = graph[graph['weight'] >= median]
            print('greater than median:', strong_tie.shape[0] / graph.shape[0])
            strong_tie.to_csv(os.path.join(self.rootpath, 'tweet_strong/%s_strong.csv' % (ts)))

        # find strong graph from weak graph in tweet_week
    def strong_tie_v2(self, path='tweet_week'):
        filefolder = islice(self.__iter_pickle(path=path,
                                               condition=lambda fname: fname.endswith('_graph.csv')), 0, None)
        for tpath in filefolder:
            print('Reading:', tpath)

            # ACCU
            graph_df = pd.read_csv(tpath, index_col=0)
            graph_df['weight'] = graph_df['weight'].astype(int)
            # 按照 real_dt 将 role_id, t_role_id 相同的行去重，保留 real_dt 最早的那一行
            df_unique = graph_df.sort_values('real_dt').drop_duplicates(['role_id', 't_role_id'])

            # 统计 role_id, t_role_id 相同的行的 weight 总和
            df_sum = graph_df.groupby(['role_id', 't_role_id']).sum().reset_index()

            # 将统计的 weight 总和赋值给去重后的行
            df_merged = pd.merge(df_unique, df_sum[['role_id', 't_role_id', 'weight']], on=['role_id', 't_role_id'],
                                 how='left')
            df_merged['weight_x'] = df_merged['weight_y']
            df_merged = df_merged.drop('weight_y', axis=1).rename(columns={'weight_x': 'weight'})

            # SET MEDIAN
            median = df_merged[['weight']].value_counts().quantile(0.3)
            if median == 1:
                median = 2
            print('median:', median)
            # weight>median is the strong tie
            strong_tie = df_merged[df_merged['weight'] >= median]
            print('greater than median:', strong_tie.shape[0] / df_merged.shape[0])

            # SAVE STRONG
            ts = graph_df['dt'].unique()[0]
            strong_tie.to_csv(os.path.join(self.rootpath, '%s/%s_strong.csv' % (path, ts)))

            # UPDATE and SAVE WEAK
            df_unique.to_csv(os.path.join(self.rootpath, '%s/%s_graph.csv' % (path, ts)))

    def week_graph(self, path='tweet_graph', right_str='graph'):
        filefolder = islice(self.__iter_pickle(path=path,
                    condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        print('-----Read tweet _%s.csv-----' % right_str)
        self.graph = None
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            graph_df = pd.read_csv(path, index_col=0)
            timestamp = path.split('/')[-1].split('_%s.csv' % right_str)[0]
            graph_df.loc[:, 'real_dt'] = timestamp
            timestamp = datetime.datetime.strptime(timestamp, '%Y%m%d')
            ts_weak = timestamp.strftime('%W')
            graph_df.loc[:, 'dt'] = int(ts_weak)

            self._update_v2(graph_df)
            self._save(path=self.datapath, right_str=right_str)

            print('processing time:', time.perf_counter() - start)

        print('-----End of reading tweets-----')

    def week_graph_v2(self, path='tweet_graph', right_str='graph'):
        filefolder = islice(self.__iter_pickle(path=path,
                                               condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        # print('-----Read tweet _%s.csv-----' % right_str)

        # gai 给定起始和最终的时间戳
        start_timestamp = pd.Timestamp('2010-01-01')
        end_timestamp = pd.Timestamp('2010-03-01')

        # gai 生成每隔n天的时间戳
        timestamps = pd.date_range(start=start_timestamp, end=end_timestamp, freq='3D')

        self.graph = None
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            graph_df = pd.read_csv(path, index_col=0)
            ori_timestamp = path.split('/')[-1].split('_%s.csv' % right_str)[0]
            timestamp = datetime.datetime.strptime(ori_timestamp, '%Y%m%d')

            if [ts for ts in timestamps if ts >= timestamp]:  # This checks if the sequence is not empty
                closest_timestamp = min([ts for ts in timestamps if ts >= timestamp]).strftime("%Y%m%d")
            else:
                closest_timestamp = end_timestamp.strftime("%Y%m%d")

            graph_df.loc[:, 'real_dt'] = timestamp.strftime("%Y%m%d")
            graph_df.loc[:, 'dt'] = closest_timestamp

            if self.graph is None:
                print('none!')
            self._update(graph_df)
            self._save(path='tweet_week', right_str=right_str)

            # print('processing time:', time.perf_counter() - start)

        # print('-----End of reading tweets-----')

    def load_sparse_graph(self, path='tweet_week', right_str='graph', is_map=True):
        filefolder = islice(self.__iter_pickle(path=path,
                    condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        graphs, nodes = {}, {}
        print('-----Read tweet _%s.csv-----' % right_str)
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            graph_df = pd.read_csv(path, index_col=0)
            weekstamp = int(path.split('/')[-1].split('_%s.csv' % right_str)[0])

            # sgraph = csr_matrix(graph_df[['role_id', 't_role_id']].values)
            sgraph = CSRMatrix(graph_df[['role_id', 't_role_id']], is_map)
            graphs[weekstamp] = sgraph
            nodes[weekstamp] = np.array(list(sgraph.node_to_index.keys()))

            print('processing time:', time.perf_counter() - start)

        print('-----End of reading tweets-----')

        return graphs, nodes

    def load_edges(self, path='tweet_week', right_str='graph', is_map=True, all_column=False):
        filefolder = islice(self.__iter_pickle(path=path,
                    condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        edges = {}
        print('-----Read tweet _%s.csv-----' % right_str)
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            graph_df = pd.read_csv(path, index_col=0)
            weekstamp = int(path.split('/')[-1].split('_%s.csv' % right_str)[0])
            if not all_column:
                edges[weekstamp] = graph_df[['role_id', 't_role_id']]
            else:
                edges[weekstamp] = graph_df
            print('processing time:', time.perf_counter() - start)

        print('-----End of reading tweets-----')

        return edges
    
    def load_labels(self, path='netease_week', right_str='labels', random_feature=None):
        filefolder = islice(self.__iter_pickle(path=path,
                    condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        labels = {}
        print('-----Read tweet _%s.csv-----' % right_str)
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            label_df = pd.read_csv(path, index_col=0)
            weekstamp = int(path.split('/')[-1].split('_%s.csv' % right_str)[0])
            labels[weekstamp] = label_df[['role_id','label']].set_index('role_id')
            print('processing time:', time.perf_counter() - start)

        print('-----End of reading tweets-----')
        
        if not random_feature is None:
            features = {}
            for key, val in labels.items():
                node_index = labels[key].index.to_numpy()
                num_node = len(node_index)
                features[key] = pd.DataFrame(np.random.rand(num_node, random_feature), index=node_index, columns=list(range(random_feature)))
        else:
            features = None

        print('-----End of reading tweets-----')
                
        return labels, features

    def load_labeled_nodes(self, path='tweet_week', right_str='labels'):
        filefolder = islice(self.__iter_pickle(path=path,
                                               condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        labels = {}
        print('-----Read tweet _%s.csv-----' % right_str)
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            label_df = pd.read_csv(path, index_col=0)
            label_df = label_df[label_df['label'] == 1]
            print(len(label_df))

            weekstamp = path.split('/')[-1].split('_%s.csv' % right_str)[0]
            labels[weekstamp] = label_df['role_id'].reset_index(drop=True).tolist()
            print('processing time:', time.perf_counter() - start)

        print('-----End of reading tweets-----')
        return labels

    def update_sampled_labels(self, dt, sampled_nodes, path='tweet_week', dst_path='tweet_sampled_graph'):
        filename = os.path.join(self.rootpath, path + '/%s_labels.csv' % dt)
        label_df = pd.read_csv(filename, index_col=0)
        label_df = label_df[label_df['role_id'].isin(sampled_nodes)]

        label_path = os.path.join(self.rootpath, '%s/%s_labels.csv' % (dst_path, dt))
        label_df.to_csv(label_path)
        return label_df

    def load_features(self, path='netease_week', right_str='features'):
        filefolder = islice(self.__iter_pickle(path=path,
                    condition=lambda fname: fname.endswith('_%s.csv' % right_str)), 0, None)

        features = {}
        print('-----Read tweet _%s.csv-----' % right_str)
        start = time.perf_counter()
        for path in filefolder:
            print('Reading:', path)

            feature_df = pd.read_csv(path, index_col=0)
            weekstamp = int(path.split('/')[-1].split('_%s.csv' % right_str)[0])
            drop_col = ['role_sex', 'week_dt']
            drop_col = list(filter(lambda x: x in feature_df.columns, drop_col))
            if len(drop_col) > 0:
                feature_df = feature_df.drop(columns=drop_col)
            features[weekstamp] = feature_df.set_index('role_id')
            print('processing time:', time.perf_counter() - start)

        print('-----End of reading tweets-----')
        return features

    def sample_weak_graph(self, path='tweet_week', weak_rstr='graph', strong_rstr='strong'):
        # 加载标签节点
        week_labels = self.load_labeled_nodes(path)
        upd_dir = 'sampled_week'

        # 遍历每周的标签节点
        for ts, nodes in week_labels.items():
            # 加载每周的strong graph
            tpath = os.path.join(self.rootpath, '%s/%s_%s.csv' % (path, ts, weak_rstr))
            graph = pd.read_csv(tpath, index_col=0)

            # 找到邻居节点（即在strong graph中有连接的节点）
            neighbors = graph[(graph['role_id'].isin(nodes)) | (graph['t_role_id'].isin(nodes))]
            # 找到不是邻居节点的节点
            without_neighbors = graph[~graph.isin(neighbors)].dropna()
            # 获取邻居节点中的唯一节点
            unique_elements = graph[['role_id', 't_role_id']].values.flatten().tolist()
            unique_elements = list(set(unique_elements))

            # 设置保存更新后的图的路径
            upd_path = os.path.join(self.rootpath, '%s/%s_%s.csv' % (upd_dir, ts, weak_rstr))
            # upd_path = tpath

            # 如果有标签节点
            if len(nodes) != 0:
                # 计算标签节点在所有节点中的比例
                percentage = len(nodes) / len(unique_elements)
                print('before update', percentage)

                # 如果比例大于0.22
#                 if percentage > 0.22:
#                     # 计算需要添加的边的数量以将比例降至约0.20
#                     num_add = len(nodes) / 0.2 - len(unique_elements)
#                     # 从非邻居节点中随机抽取节点添加，现在是放回采样
#                     # 不放回需要在后面添加 replace=False
#                     edge2add = without_neighbors.sample(n=math.ceil(num_add / 2))
#                     # 连接邻居节点和采样后的节点
#                     sample_results = pd.concat([neighbors, edge2add], axis=0)
#                     # 更新唯一节点列表
#                     uni_ele = sample_results[['role_id', 't_role_id']].values.flatten().tolist()
#                     uni_ele = list(set(uni_ele))

#                     # 打印相关信息
#                     print(ts, len(nodes), '/', len(uni_ele))
#                     print('updated precentage', len(nodes) / len(uni_ele))

#                     # 保存采样结果
#                     sample_results.to_csv(upd_path)

#                     # 更新标签节点
#                     self.update_sampled_labels(ts, uni_ele, path, upd_dir)

                # 如果比例小于0.18
                if percentage < 0.18:
                    # 从包含标签节点的边中找到label点（分别包括role_id, t_role_id)，并添加到采样结果中
                    role = neighbors[neighbors['role_id'].isin(nodes)].drop_duplicates('role_id')
                    trole = neighbors[neighbors['t_role_id'].isin(nodes)].drop_duplicates('t_role_id')
                    sample_results = pd.concat([role, trole], axis=0).drop_duplicates()
                    # 在neighbors中找到label点的邻居
                    without = neighbors[~neighbors.isin(sample_results)].dropna()

                    # 计算需要添加的节点数量以将比例提高到约0.18
                    num_add = len(nodes) / 0.18 - len(sample_results)
                    # 从非邻居节点中随机抽取节点添加，现在是放回采样
                    # 不放回需要在最后添加 replace=False
                    edge2add = without.sample(n=min(len(without), math.ceil(num_add / 2)))
                    sample_results = pd.concat([sample_results, edge2add], axis=0)

                    # 更新唯一节点列表
                    uni_ele = sample_results[['role_id', 't_role_id']].values.flatten().tolist()
                    uni_ele = list(set(uni_ele))

                    # 打印相关信息
                    print(ts, len(nodes), '/', len(uni_ele))
                    print('updated precentage', len(nodes) / len(uni_ele))

                    # 保存采样结果
                    sample_results.to_csv(upd_path)

                # 如果比例在0.18到0.22之间
                else:
                    uni_ele = neighbors[['role_id', 't_role_id']].values.flatten().tolist()
                    uni_ele = list(set(uni_ele))

                    # 打印相关信息
                    print(ts, len(nodes), '/', len(uni_ele))
                    print('updated precentage', len(nodes) / len(uni_ele))

                    # 保存采样结果
                    neighbors.to_csv(upd_path)

                    
                '''update strong'''
                spath = os.path.join(self.rootpath, '%s/%s_%s.csv' % (upd_dir, ts, strong_rstr))
                sgraph = pd.read_csv(tpath, index_col=0)
                upd_sgraph = sgraph[sgraph['role_id'].isin(uni_ele) | sgraph['t_role_id'].isin(uni_ele)]
                upd_sgraph.to_csv(spath)
                print('updated strong', len(sgraph)/len(upd_sgraph))

                # 更新标签节点
                uni_ele = list(set(uni_ele) | set(sgraph[['role_id', 't_role_id']].values.flatten().tolist()))
                upd_labels = self.update_sampled_labels(ts, uni_ele, path, upd_dir)

                '''copy features'''
                fpath = os.path.join(self.rootpath, '%s/%s_features.csv' % (path, ts))
                tfeatures = pd.read_csv(fpath, index_col=0)
                upd_tfeatures = tfeatures[tfeatures['role_id'].isin(uni_ele)]
                upd_tfeatures.to_csv(os.path.join(self.rootpath, '%s/%s_features.csv' % (upd_dir, ts)))
                print('updated features', len(tfeatures)/len(upd_tfeatures))

    def sample_weak_graph_v2(self, path='tweet_week', weak_rstr='graph', strong_rstr='strong'):
        week_labels = self.load_labeled_nodes(path)
        upd_dir = 'sampled_week'

        for ts, nodes in week_labels.items():
            tpath = os.path.join(self.rootpath, '%s/%s_%s.csv' % (path, ts, weak_rstr))
            graph = pd.read_csv(tpath, index_col=0)

            # 找到邻居节点（即在weak graph中有连接的节点）
            neighbors = graph[(graph['role_id'].isin(nodes)) | (graph['t_role_id'].isin(nodes))]
            assert len(set(neighbors[['role_id', 't_role_id']].values.flatten().tolist()) & set(nodes)) == len(nodes)
            # 找到不是邻居节点的节点
            without_neighbors = graph[~graph.isin(neighbors)].dropna()
            # 获取邻居节点中的唯一节点
            unique_elements = graph[['role_id', 't_role_id']].values.flatten().tolist()
            unique_elements = list(set(unique_elements))

            upd_path = os.path.join(self.rootpath, '%s/%s_%s.csv' % (upd_dir, ts, weak_rstr))
            # upd_path = tpath

            if len(nodes) != 0:
                percentage = len(nodes) / len(unique_elements)
                print('before update', percentage)

                if percentage < 0.18:
                    num_add = int(len(nodes) * (1 / 0.18 - 1))
                    uni_ele = list(set(unique_elements) - set(nodes))
                    random.shuffle(uni_ele)
                    uni_ele = uni_ele[:num_add] + nodes
                    sample_results = neighbors[(neighbors['role_id'].isin(uni_ele)) & (neighbors['t_role_id'].isin(uni_ele))]
                    uni_ele = list(set(sample_results[['role_id', 't_role_id']].values.flatten().tolist()))

                    print(ts, len(nodes), '/', len(uni_ele))
                    print('updated precentage', len(nodes) / len(uni_ele))

                    sample_results.to_csv(upd_path)

                else:
                    uni_ele = neighbors[['role_id', 't_role_id']].values.flatten().tolist()
                    uni_ele = list(set(uni_ele))

                    print(ts, len(nodes), '/', len(uni_ele))
                    print('updated precentage', len(nodes) / len(uni_ele))

                    neighbors.to_csv(upd_path)

                '''update strong'''
                spath = os.path.join(self.rootpath, '%s/%s_%s.csv' % (upd_dir, ts, strong_rstr))
                sgraph = pd.read_csv(spath, index_col=0)
                upd_sgraph = sgraph[sgraph['role_id'].isin(uni_ele) & sgraph['t_role_id'].isin(uni_ele)]
                upd_sgraph.to_csv(spath)
                print('updated strong', len(sgraph) / len(upd_sgraph))

                '''update labels'''
                # uni_ele = list(set(uni_ele) | set(sgraph[['role_id', 't_role_id']].values.flatten().tolist()))
                upd_labels = self.update_sampled_labels(ts, uni_ele, path, upd_dir)

                '''copy features'''
                fpath = os.path.join(self.rootpath, '%s/%s_features.csv' % (path, ts))
                tfeatures = pd.read_csv(fpath, index_col=0)
                upd_tfeatures = tfeatures[tfeatures['role_id'].isin(uni_ele)]
                upd_tfeatures.to_csv(os.path.join(self.rootpath, '%s/%s_features.csv' % (upd_dir, ts)))
                print('updated features', len(tfeatures) / len(upd_tfeatures))

                assert len(upd_labels) == len(upd_tfeatures) and len(upd_tfeatures) == len(uni_ele)

def read_nodes_v1(path='./', weeks=10):
    '''random version'''
    with open(os.path.join(path, 'tweet_pair/node_id.pkl'), 'rb') as f:
        name2node = pickle.load(f)
    num_node = len(name2node)
    node_index = list(name2node.keys())
    features, labels = None, None
    for w in range(weeks):
        # read w-th week feature then concat
        wfeat = pd.DataFrame(np.random.rand(num_node, 20))
        wfeat.loc['role_id'] = node_index
        wfeat.loc['dt'] = w
        features = pd.concat([features, wfeat], axis=0)
        # read w-th week label
        wlab = pd.DataFrame(np.random.rand(num_node, 1))
        wlab.loc['role_id'] = node_index
        wlab.loc['dt'] = w
        labels = pd.concat([labels, wlab], axis=0)
    return features, labels

def read_nodes_v2(path='./', weeks=10):
    path = os.path.dirname(os.path.abspath(__file__))
    '''random version'''
    with open(os.path.join(path, 'tweet_pair/node_id.pkl'), 'rb') as f:
        name2node = pickle.load(f)
    node_index = np.array(list(set(name2node.values())))
    num_node = len(node_index)
    features, labels = {}, {}
    for w in range(weeks):
        # read w-th week feature then concat
        wfeat = pd.DataFrame(np.random.rand(num_node, 10), index=node_index)
        features[w] = wfeat
        # read w-th week label
        wlab = pd.DataFrame((np.random.rand(num_node, 1)>0.5).astype(int), index=node_index)
        labels[w] = wlab
    return features, labels, node_index

def read_nodes_v3(path='tweet_week', weeks=10):
    bg = build_tgraph()
    labels, _ = bg.load_labels(path, random_feature=None)
    features = bg.load_features(path, 'features')

    path = os.path.dirname(os.path.abspath(__file__))
    '''random version'''
    with open(os.path.join(path, 'tweet_pair/node_id.pkl'), 'rb') as f:
        name2node = pickle.load(f)
    node_index = np.array(list(set(name2node.values())))
    num_node = len(node_index)
    # features, labels = {}, {}
    # for w in range(weeks):
    #     # read w-th week feature then concat
    #     wfeat = pd.DataFrame(np.random.rand(num_node, 10), index=node_index)
    #     features[w] = wfeat
    #     # read w-th week label
    #     wlab = pd.DataFrame((np.random.rand(num_node, 1)>0.5).astype(int), index=node_index)
    #     labels[w] = wlab
    return features, labels, node_index

read_nodes = read_nodes_v3


def get_dataset(path='tweet_week', filter_feature=False, positive_subgraph=False):
    '''
    read all data to a dict: timestamp -> data
    '''
    bg = build_tgraph('./data_processing')
    weak_edges = bg.load_edges(path=path, right_str='graph', is_map=False)  # load strong tie
    strong_edges = bg.load_edges(path=path, right_str='strong', is_map=False)
    features, labels, _ = read_nodes(path='tweet_week', weeks=len(weak_edges))

    global timestamps
    timestamps = sorted(labels.keys())
    weak_edges = {key: weak_edges[key] for key in timestamps}
    strong_edges = {key: strong_edges[key] for key in timestamps}
    features = {key: features[key] for key in timestamps}

    if positive_subgraph:
        for t in labels.keys():
            t_lab = labels[t]
            t_strong = strong_edges[t]
            t_weak = weak_edges[t]

            pnodes = np.unique(t_lab[t_lab['label'] == 1].index)
            all_nodes = set(features[t].index.values) & set(t_lab.index.values) & \
                        set(np.unique(t_strong[['role_id', 't_role_id']].values)) & \
                        set(np.unique(t_weak[['role_id', 't_role_id']].values))
            all_nodes = np.array(list(all_nodes))

            strong_edges[t] = t_strong[((t_strong['role_id'].isin(pnodes)) | (t_strong['t_role_id'].isin(pnodes))) &
                                       (t_strong['role_id'].isin(all_nodes)) & (t_strong['t_role_id'].isin(all_nodes))]

            weak_edges[t] = t_weak[((t_weak['role_id'].isin(pnodes)) | (t_weak['t_role_id'].isin(pnodes))) &
                                   (t_weak['role_id'].isin(all_nodes)) & (t_weak['t_role_id'].isin(all_nodes))]

            strong_nodes = np.unique(strong_edges[t][['role_id', 't_role_id']])
            weak_nodes = np.unique(weak_edges[t][['role_id', 't_role_id']])
            assert len(np.intersect1d(strong_nodes, weak_nodes)) == len(strong_nodes)

            strong_labels = labels[t].loc[strong_nodes]
            weak_labels = labels[t].loc[weak_nodes]
            print('strong ratio: %f, weak ratio: %f' % (len(strong_labels[strong_labels['label']==1])/len(strong_labels),
                                                        len(weak_labels[weak_labels['label']==1])/len(weak_labels)))

    if filter_feature:
        for t in labels.keys():
            all_nodes = np.unique(pd.concat([weak_edges[t], strong_edges[t]])[['role_id', 't_role_id']])
            features[t] = features[t].loc[all_nodes]
            labels[t] = labels[t].loc[all_nodes]

    sedge, snode, slab, wedge, wnode, wlab = [], [], [], [], [], []
    for t in labels.keys():
        sedge.append(len(strong_edges[t]))
        sn = np.unique(strong_edges[t][['role_id', 't_role_id']])
        snode.append(len(sn))
        slab.append((labels[t].loc[sn]==1).sum().item())
        wn = np.unique(weak_edges[t][['role_id', 't_role_id']])
        wnode.append(len(wn))
        wedge.append(len(weak_edges[t]))
    # print('strong edge: %d, strong node: %d, strong label: %d, weak edge: %d' % (np.sum(sedge), np.sum(snode), np.sum(slab), np.sum(wedge)))
    print(f'strong edge: {np.sum(sedge)}, strong node: {np.sum(snode)}, strong label: {np.sum(slab)}, weak node: {np.sum(wnode)}, weak edge: {np.sum(wedge)}')

    return weak_edges, strong_edges, features, labels

def get_dataset_inference(path='netease_inference', filter_feature=False):
    '''
    read all data to a dict: timestamp -> data
    '''
    bg = build_tgraph()
    weak_edges = bg.load_edges(path=path, right_str='graph', is_map=False)  # load strong tie
    strong_edges = bg.load_edges(path=path, right_str='strong', is_map=False)
    features = bg.load_features(path, 'features')

    global timestamps
    timestamps = sorted(strong_edges.keys())
    weak_edges = {i: weak_edges[key] for i, key in enumerate(timestamps)}
    features = {i: features[key] for i, key in enumerate(timestamps)}
    strong_edges = {i: strong_edges[key] for i, key in enumerate(timestamps)}

    for t in strong_edges.keys():
        t_strong = strong_edges[t]
        strong_edges[t] = t_strong[t_strong[['role_id', 't_role_id']].isin(features[t].index).all(axis=1)]

        t_weak = weak_edges[t]
        weak_edges[t] = t_weak[t_weak[['role_id', 't_role_id']].isin(features[t].index).all(axis=1)]

        strong_nodes = set(np.unique(strong_edges[t][['role_id', 't_role_id']]).tolist())
        weak_nodes = set(np.unique(weak_edges[t][['role_id', 't_role_id']]).tolist())
        assert len(strong_nodes & weak_nodes) <= len(strong_nodes), 'In %s, intersection=%d and strong nodes=%d' % (
        t, len(strong_nodes & weak_nodes), len(strong_nodes))

        strong_nodes = strong_nodes & weak_nodes
        strong_edges[t] = t_strong[
            (t_strong['role_id'].isin(strong_nodes)) & (t_strong['t_role_id'].isin(strong_nodes))]

    return weak_edges, strong_edges, features

def get_dataset_cold(path='tweet_week', filter_feature=False, positive_subgraph=False, is_cold=False):
    '''
    read all data to a dict: timestamp -> data
    '''
    weak_edges, strong_edges, features, labels = get_dataset(path, False, positive_subgraph)

    cold_edges = None
    if is_cold:
        # load cold edges and filtering nodes
        bg = build_tgraph()
        cold_edges = bg.load_edges(path=path, right_str='cold', is_map=False, all_column=True)
        cold_edges = {i: cold_edges[key] for i, key in enumerate(timestamps) if key in cold_edges}
        for t in cold_edges.keys():
            t_cold = cold_edges[t]
            t_node = np.intersect1d(features[t].index.values, labels[t].index.values)
            cold_edges[t] = t_cold[t_cold['role_id'].isin(t_node)]

        feature_df = dict_to_df(features)
        feature_df = feature_df.reset_index().set_index(['role_id'])
        label_df = dict_to_df(labels)
        label_df = label_df.reset_index().set_index(['role_id'])

        # cold edge reindexing
        max_roleid = max([item.index.max() for item in features.values()])
        min_roleid = min([item.index.min() for item in features.values()])
        max_time = (int(max(labels.keys()) / 10) + 1) * 10
        print(f'cold index bias: (x + {max_roleid}) * {max_time}')
        cold_class = max(np.unique(dict_to_df(labels)['label'])) + 1
        print(f'cold class in label: {cold_class}')

        cold_bias = np.array([max_roleid, max_time])
        np.save('inference/cold_bias.npy', cold_bias)

        for t, t_edges in cold_edges.items():
            t_edges = cold_edges[t]
            ## retrieve
            unique_t = np.unique(t_edges['t_dt'])
            t_cold_feat, t_cold_lab, t_cold_idx = [], [], []
            for ut in unique_t:
                ut_edges = t_edges[t_edges['t_dt'] == ut]['t_role_id']
                ut_features = feature_df[feature_df['timestamp'] == ut]
                ut_labels = label_df[label_df['timestamp'] == ut]

                ## filtering
                ut_idx = ut_edges.isin(ut_features.index.values)
                ut_edges = ut_edges[ut_idx].drop_duplicates()
                t_cold_idx.append(t_edges[t_edges['t_dt'] == ut].index.values[ut_idx])

                t_cold_feat.append(ut_features.loc[ut_edges])
                t_cold_lab.append(ut_labels.loc[ut_edges])
            t_cold_feat = pd.concat(t_cold_feat).reset_index()
            t_cold_lab = pd.concat(t_cold_lab).reset_index()
            t_cold_idx = np.concatenate(t_cold_idx)

            ## filtering cold edges
            # print(f'At {t}, filtering reduce cold edges from {len(t_edges)} to {len(t_cold_idx)}, cold nodes from {len(np.unique(t_edges))} to {len(np.unique(t_edges.loc[t_cold_idx]))}')
            assert (len(t_edges) == len(t_cold_idx)) and (
                        len(np.unique(t_edges)) == len(np.unique(t_edges.loc[t_cold_idx]))), \
                f'At {t}, filtering reduce cold edges from {len(t_edges)} to {len(t_cold_idx)}, cold nodes from {len(np.unique(t_edges))} to {len(np.unique(t_edges.loc[t_cold_idx]))}'
            t_edges = t_edges.loc[t_cold_idx]

            ## reindexing
            t_cold_feat['role_id'] = (
                        t_cold_feat['role_id'].map(lambda x: (abs(x) + max_roleid) * max_time) + t_cold_feat[
                    'timestamp']).astype(np.int64)
            t_cold_lab['role_id'] = (
                    t_cold_lab['role_id'].map(lambda x: (abs(x) + max_roleid) * max_time) + t_cold_lab[
                'timestamp']).astype(
                np.int64)
            t_edges['t_role_id'] = (
                    t_edges['t_role_id'].map(lambda x: (abs(x) + max_roleid) * max_time) + t_edges['t_dt']).astype(
                np.int64)

            cold_edges[t] = t_edges[strong_edges[t].columns]
            ## concat feature
            t_cold_feat = t_cold_feat.drop('timestamp', axis=1).set_index('role_id')
            features[t] = pd.concat([features[t], t_cold_feat], axis=0)
            ## concat label
            t_cold_lab = t_cold_lab.drop('timestamp', axis=1).set_index('role_id')
            labels[t] = pd.concat([labels[t], t_cold_lab], axis=0)

        aftermin = min([item.index.min() for item in features.values()])
        aftermax = max([item.index.max() for item in features.values()])
        print(f'role id min: {min_roleid}, max: {max_roleid}; after min: {aftermin}, after max: {aftermax}')

    if filter_feature:
        for t in labels.keys():
            if (is_cold and (t in cold_edges)):
                all_nodes = np.unique(
                    pd.concat([weak_edges[t], strong_edges[t], cold_edges[t]])[['role_id', 't_role_id']])
            else:
                all_nodes = np.unique(pd.concat([weak_edges[t], strong_edges[t]])[['role_id', 't_role_id']])
            features[t] = features[t].loc[all_nodes]
            labels[t] = labels[t].loc[all_nodes]

        ## check index alignment
        for t in labels.keys():
            if (is_cold and (t in cold_edges)):
                t_edges = pd.concat([weak_edges[t], strong_edges[t], cold_edges[t]])
            else:
                t_edges = pd.concat([weak_edges[t], strong_edges[t]])
            t_feat_index = set(features[t].index.values.tolist())
            t_label_index = set(labels[t].index.values.tolist())
            t_index = set(np.unique(t_edges[['role_id', 't_role_id']]).tolist())
            assert len(t_feat_index & t_label_index & t_index) == len(t_index) and len(t_feat_index) == len(
                t_label_index)
            # final ratio
            weak_nodes = set(np.unique(weak_edges[t][['role_id', 't_role_id']]).tolist())
            weak_nodes = t_index | weak_nodes

            weak_labels = labels[t].loc[list(weak_nodes)]
            print('weak ratio becomes: %f' % (len(weak_labels[weak_labels['label'] == 1]) / len(weak_labels)))

    sedge, snode, slab, wedge, wnode, wlab = [], [], [], [], [], []
    for t in labels.keys():
        sedge.append(len(strong_edges[t]))
        sn = np.unique(strong_edges[t][['role_id', 't_role_id']])
        snode.append(len(sn))
        slab.append((labels[t].loc[sn] == 1).sum().item())
        wn = np.unique(weak_edges[t][['role_id', 't_role_id']])
        wnode.append(len(wn))
        wedge.append(len(weak_edges[t]))
    print(
        f'strong edge: {np.sum(sedge)}, strong node: {np.sum(snode)}, strong label: {np.sum(slab)}, weak node: {np.sum(wnode)}, weak edge: {np.sum(wedge)}')

    return weak_edges, strong_edges, cold_edges, features, labels

def get_netease_dataset(path='netease_week', filter_feature=False, positive_subgraph=False):
    '''
    read all data to a dict: timestamp -> data
    '''
    bg = build_tgraph()
    weak_edges = bg.load_edges(path=path, right_str='graphs', is_map=False)  # load strong tie
    strong_edges = bg.load_edges(path=path, right_str='strong', is_map=False)
    labels, _ = bg.load_labels(path, random_feature=None)
    features = bg.load_features(path, 'features')

    weak_edges = {key: weak_edges[key] for key in labels.keys()}
    strong_edges = {key: strong_edges[key] for key in labels.keys()}
    features = {key: features[key] for key in labels.keys()}

    if positive_subgraph:
        for t in labels.keys():
            t_lab = labels[t]
            pnodes = np.unique(t_lab[t_lab['label'] == 1].index)

            t_strong = strong_edges[t]
            strong_edges[t] = t_strong[(t_strong['role_id'].isin(pnodes)) | (t_strong['t_role_id'].isin(pnodes))]

            t_weak = weak_edges[t]
            weak_edges[t] = t_weak[(t_weak['role_id'].isin(pnodes)) | (t_weak['t_role_id'].isin(pnodes))]

            strong_nodes = set(np.unique(strong_edges[t][['role_id', 't_role_id']]).tolist())
            weak_nodes = set(np.unique(weak_edges[t][['role_id', 't_role_id']]).tolist())
            assert len(strong_nodes & weak_nodes) == len(strong_nodes), 'In %s, intersection=%d and strong nodes=%d'% (t, len(strong_nodes & weak_nodes), len(strong_nodes))

            strong_labels = labels[t].loc[strong_nodes]
            weak_labels = labels[t].loc[weak_nodes]
            print('strong ratio: %f, weak ratio: %f' % (len(strong_labels[strong_labels['label']==1])/len(strong_labels),
                                                        len(weak_labels[weak_labels['label']==1])/len(weak_labels)))

    if filter_feature:
        for t in labels.keys():
            all_nodes = np.unique(pd.concat([weak_edges[t], strong_edges[t]])[['role_id', 't_role_id']])
            features[t] = features[t].loc[all_nodes]
            labels[t] = labels[t].loc[all_nodes]
    return weak_edges, strong_edges, features, labels

def get_netease_dataset_cold(path='netease_week', filter_feature=False, positive_subgraph=False):
    '''
    read all data to a dict: timestamp -> data
    '''
    bg = build_tgraph()
    weak_edges = bg.load_edges(path=path, right_str='graphs', is_map=False)  # load strong tie
    strong_edges = bg.load_edges(path=path, right_str='strong', is_map=False)
    labels, _ = bg.load_labels(path, random_feature=None)
    features = bg.load_features(path, 'features')

    weak_edges = {key: weak_edges[key] for key in labels.keys()}
    strong_edges = {key: strong_edges[key] for key in labels.keys()}
    features = {key: features[key] for key in labels.keys()}

    cold_edges = bg.load_edges(path=path, right_str='cold', is_map=False)
    cold_edges = {key: cold_edges[key] for key in labels.keys() if key in cold_edges}

    if positive_subgraph:
        for t in labels.keys():
            t_lab = labels[t]
            pnodes = np.unique(t_lab[t_lab['label'] == 1].index)

            t_strong = strong_edges[t]
            strong_edges[t] = t_strong[(t_strong['role_id'].isin(pnodes)) | (t_strong['t_role_id'].isin(pnodes))]

            t_weak = weak_edges[t]
            weak_edges[t] = t_weak[(t_weak['role_id'].isin(pnodes)) | (t_weak['t_role_id'].isin(pnodes))]

            strong_nodes = set(np.unique(strong_edges[t][['role_id', 't_role_id']]).tolist())
            weak_nodes = set(np.unique(weak_edges[t][['role_id', 't_role_id']]).tolist())
            assert len(strong_nodes & weak_nodes) == len(strong_nodes), 'In %s, intersection=%d and strong nodes=%d'% (t, len(strong_nodes & weak_nodes), len(strong_nodes))

            strong_labels = labels[t].loc[strong_nodes]
            weak_labels = labels[t].loc[weak_nodes]
            print('strong ratio: %f, weak ratio: %f' % (len(strong_labels[strong_labels['label']==1])/len(strong_labels),
                                                        len(weak_labels[weak_labels['label']==1])/len(weak_labels)))

            if t in cold_edges:
                t_cold = cold_edges[t]
                cond1 = (t_cold['role_id'].isin(pnodes)) | (t_cold['t_role_id'].isin(pnodes))
                all_unique_nodes = np.unique(t_lab.index.values)
                cond2 = (t_cold['rold_id'].isin(all_unique_nodes)) & (t_cold['t_rold_id'].isin(all_unique_nodes))
                cold_edges[t] = t_cold[cond1 & cond2]

    if filter_feature:
        for t in labels.keys():
            if t in cold_edges:
                all_nodes = np.unique(pd.concat([weak_edges[t], strong_edges[t], cold_edges[t]])[['role_id', 't_role_id']])
            else:
                all_nodes = np.unique(pd.concat([weak_edges[t], strong_edges[t]])[['role_id', 't_role_id']])
            features[t] = features[t].loc[all_nodes]
            labels[t] = labels[t].loc[all_nodes]
    return weak_edges, strong_edges, cold_edges, features, labels

def concat_df(edge_dict, columns=['role_id', 't_role_id', 'etype', 'timestamp']):
    total_df = pd.DataFrame([], columns=columns)
    for etype, edges in edge_dict.items():
        for timestamp in sorted(edges.keys()):
            t_edge = edges[timestamp]
            t_edge.loc[:, 'etype'] = etype
            t_edge.loc[:, 'timestamp'] = timestamp
            total_df = pd.concat([total_df, t_edge], axis=0)
    print('------concat edges------')

    return total_df

def dict_to_df(dict_data, columns='timestamp'):
    data = []
    new_col = []
    for key, val in dict_data.items():
        data.append(val)
        new_col.append(np.ones(len(val)) * int(key))
    data = pd.concat(data, axis=0)
    new_col = np.concatenate(new_col)
    data[columns] = np.array(new_col)
    return data

def pkl2csv():
    rootpath = os.path.dirname(os.path.abspath(__file__))
    '''node index'''
    with open(os.path.join(rootpath, 'tweet_pair/node_id.pkl'), 'rb') as f:
        name2node = pickle.load(f)

    path = os.path.join(rootpath, 'encoded_feature')
    for filename in os.listdir(path):
        if filename.endswith('.pkl'):
            fpath = os.path.join(path, filename)
            with open(fpath, 'rb') as fh:
                data = pickle.load(fh)
            df = pd.DataFrame(np.array(list(data.values())), index=map(lambda x: name2node[x], data.keys()))
            df.index.name = 'role_id'
            df = df.reset_index()

            weekid = int(filename.split('_')[1])
            df.to_csv(os.path.join('tweet_week/%d_features.csv' % weekid))

def check_len():
    for filename in os.listdir('tweet_graph'):
        df = pd.read_csv('tweet_graph/' + filename)
        print('file: %s len: %d' % (filename, len(df)))
    print('##### weak graph #####')
    for t in range(10):
        df = pd.read_csv('tweet_week/%d_graph.csv' % t)
        print('data: %d len: %d' % (t, len(df)))
    print('##### strong graph #####')
    for t in range(10):
        df = pd.read_csv('tweet_week/%d_strong.csv' % t)
        print('data: %d len: %d' % (t, len(df)))

def check_align(path='tweet_week'):
    print('##### weak graph #####')
    for t in range(9):
        df = pd.read_csv('%s/%d_graph.csv' % (path, t))
        print('weak: %d len: %d' % (t, len(set(df[['role_id', 't_role_id']].values.flatten().tolist()))))
        df = pd.read_csv('%s/%d_strong.csv' % (path, t))
        print('strong: %d len: %d' % (t, len(set(df[['role_id', 't_role_id']].values.flatten().tolist()))))
        df = pd.read_csv('%s/%d_features.csv' % (path, t))
        print('feature: %d len: %d' % (t, len(df[['role_id']])))
        df = pd.read_csv('%s/%d_labels.csv' % (path, t))
        print('label: %d len: %d' % (t, len(df[['role_id']])))

if __name__ == '__main__':
    '''process data from raw text'''
    # tw = read_tweet()
    # end_date = datetime.datetime(2010, 3, 1)
    # tw.dic_contents(end_date=end_date)

    # bg = build_tgraph(is_accumu=False)
    bg = build_tgraph()
    '''build graph edge by dt, role_id, t_role_id, weight'''
    # bg.build_tgraph()

    # '''week accmulative graph: weak and strong tie'''
    bg.week_graph()

    '''find strong tie according to graph frequent'''
    bg.strong_tie_v2()

    '''Please run label.ipynb first'''
    '''sampling based on labeled nodes'''
    # bg.sample_weak_graph_v2()

    # pkl2csv()
    # check_len()
    # check_align('sampled_week')
    # read_nodes_v3('tweet_week')
