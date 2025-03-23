import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np


class StringTrie:
    '''
  String Trie
  format: [Degree(2), 3x [Degree(2), Count(2)]] x3 + '_' x10 -> 62 length string
  '''

    def __init__(self, LEAF_NUM, time_list, node_list, types=100):
        self.LEAF_NUM = LEAF_NUM
        self.NODE_NUM = 10 * LEAF_NUM
        self.TYPES = types
        self.NUM_SIZE = 62

        self.time_list = time_list
        self.node_list = node_list

        self.count = np.zeros(self.NODE_NUM)
        self.children = -np.ones([self.NODE_NUM, self.TYPES], dtype=np.int32)
        self.data = np.empty(self.NODE_NUM, dtype=object)
        self.id = np.zeros(self.NODE_NUM)
        self.time = np.zeros(self.NODE_NUM)
        self.data_index = {}
        self.LAST_INDEX = 0

        self.data[0] = ''
        self.LAST_INDEX = 1
        self.ROOT = 0

        self.pre_order_arr = np.zeros(self.NODE_NUM, dtype=np.int32)
        self.unique_str = None
        self._near_cache = {}
        self._search_cache = {}

        self.nx_graph = None

    def _bits(self, np_num):
        np_num = np_num.split('_')
        np_list = []
        for i in range(len(np_num)):
            stri = np_num[i]
            np_list.append(stri)
            yield '_'.join(np_list), stri

    def sequential(self, x):
        return '_'.join(map(str, x))

    def desequential(self, x):
        return list(map(int, x.split('_')))

    def insert(self, word, id, timestamp):
        '''
    insert word and update Trie
    '''
        assert len(word) == self.NUM_SIZE
        # root index
        cur_idx = self.ROOT
        for num, bit in self._bits(word):
            ibit = int(bit)
            if self.children[cur_idx][ibit] == -1:
                self.children[cur_idx][ibit] = self.LAST_INDEX
                self.LAST_INDEX += 1

                cur_idx = self.children[cur_idx][ibit]
                self.data[cur_idx] = num
            else:
                cur_idx = self.children[cur_idx][ibit]

            self.count[cur_idx] += 1
        # self.id[cur_idx] = self.sequential(id)
        # self.time[cur_idx] = self.sequential(timestamp)
        self.time[cur_idx] = timestamp[0]
        self.id[cur_idx] = id[0]

    def insertV2(self, word, idx):
        '''
        insert word and update Trie
        '''
        assert len(word) == self.NUM_SIZE
        # root index
        cur_idx = self.ROOT
        for num, bit in self._bits(word):
            ibit = int(bit)
            if self.children[cur_idx][ibit] == -1:
                self.children[cur_idx][ibit] = self.LAST_INDEX
                self.LAST_INDEX += 1

                cur_idx = self.children[cur_idx][ibit]
                self.data[cur_idx] = num
            else:
                cur_idx = self.children[cur_idx][ibit]

            self.count[cur_idx] += 1
        if cur_idx not in self.data_index:
            self.data_index[cur_idx] = [idx]
        else:
            self.data_index[cur_idx].append(idx)

    def preOrderTraverse(self):
        cur_idx = -1
        cnt = 0
        last_idx = []
        visited = np.zeros(self.NODE_NUM)

        last_idx.append(self.ROOT)
        while len(last_idx) != 0:
            cur_idx = last_idx.pop()
            if not visited[cur_idx]:
                visited[cur_idx] = 1
                children = self.children[cur_idx][self.children[cur_idx] != -1]
                if len(children) > 0:
                    last_idx += children.tolist()
                else:
                    self.pre_order_arr[cnt] = cur_idx
                    cnt += 1

        self.unique_str = np.unique(self.data[self.pre_order_arr])
        return self.pre_order_arr

    #
    # def retNear(self, window=5):
    #   pad_size = int(window / 2)
    #   pad_arr = np.concatenate([np.zeros(pad_size), self.pre_order_arr, np.zeros(pad_size)])
    #   index_arr = np.concatenate([
    #     np.arange(i, len(self.pre_order_arr) + i)[:, np.newaxis] for i in range(window)
    #   ], axis=1)
    #   near_arr = pad_arr[index_arr].reshape(window, -1).transpose().astype(np.int32)
    #   num_arr = self.data[near_arr]
    #   id_arr = self.id[near_arr]
    #   time_arr = self.time[near_arr]
    #   return num_arr, id_arr, time_arr

    def findSimilar(self, word, current_time, current_node):
        same_word = np.where(self.data == word)[0]

        similar_id, id_time = [], []
        for _idx in same_word:
            _didx = self.data_index[_idx]
            itime, iid = self.time_list[_didx], self.node_list[_didx]
            before_time = itime <= current_time
            not_node = iid != current_node
            similar_id.append(iid[before_time & not_node])
            id_time.append(itime[before_time & not_node])
        similar_id = np.concatenate(similar_id, axis=0)
        id_time = np.concatenate(id_time, axis=0)
        # before_time = self.time <= current_time
        # similar_id = self.id[same_word & before_time]
        # id_time = self.time[same_word & before_time]
        return similar_id, id_time

    def searchNear(self, word, current_time, current_node, num=3, threshold=0.5):
        '''
    Return before or after words in pre-order traverse
    '''
        if word in self._search_cache:
            similar_id, id_time = self._search_cache[word]
        else:
            similar_id, id_time = self.findSimilar(word, current_time, current_node)
            # print('similar id:', similar_id)

            if len(similar_id) < num:
                similar_words = self.retNear(word, threshold=threshold)
                # print(similar_words)
                for item in similar_words:
                    _id, _time = self.findSimilar(item, current_time, current_node)
                    similar_id = np.concatenate([similar_id, _id])
                    id_time = np.concatenate([id_time, _time])
                    # print(item, ':', similar_id)
            self._search_cache[word] = [similar_id, id_time]

        shuf_idx = np.arange(len(similar_id))
        np.random.shuffle(shuf_idx)
        similar_id = similar_id[shuf_idx][:num].astype(np.int32)
        id_time = id_time[shuf_idx][:num].astype(np.int32)

        return similar_id, id_time

    def retNear(self, word, windows=2, threshold=0.5):
        if word in self._near_cache:
            return self._near_cache[word]
        pre_order_index = np.where(self.unique_str == word)[0][0]
        similar_words = []
        for i in range(pre_order_index - windows, pre_order_index + windows):
            if i >= 0 and i < len(self.unique_str) and self.similarity(word, self.unique_str[i]) >= threshold \
                    and i != pre_order_index:
                similar_words.append(self.unique_str[i])
        if word not in self._near_cache:
            self._near_cache[word] = similar_words
        return similar_words

    def similarity(self, word1, word2):
        if word1 == '' or word2 == '':
            return -1
        seq1, seq2 = np.array(self.desequential(word1)), np.array(self.desequential(word2))
        index = (seq1 > 0) | (seq2 > 0)
        seq1, seq2 = seq1[index], seq2[index]
        return (seq1 == seq2).sum() / len(seq1)

    def to_networkx(self):
        max_index = self.LAST_INDEX
        # add edges
        self.nx_graph = nx.DiGraph()
        for cur_idx in range(max_index):
            nbr = self.children[cur_idx]
            nbr = nbr[nbr != -1]
            nbr_pair = np.stack([cur_idx * np.ones_like(nbr), nbr], axis=1)
            self.nx_graph.add_edges_from(nbr_pair)

        # set node attributes
        nodes = self.nx_graph.nodes()
        nodes_data = self.data[nodes]
        nodes_count = self.count[nodes]
        # nodes_id = self.id[nodes]
        # nodes_time = self.time[nodes]
        # filter leaf
        nodes_leaf = (self.children[nodes] != -1).sum(axis=1) == 0

        nodes_attribute = {
            n: {'data': nd, 'count': nc, 'is_leaf': nf}
            for n, nd, nc, nf in zip(nodes, nodes_data, nodes_count, nodes_leaf)
        }
        nx.set_node_attributes(self.nx_graph, nodes_attribute)

    def visualize(self, is_leaf=False, attr='data', layout='dot'):
        if self.nx_graph is None:
            assert 'run to_networkx first'
        pos = graphviz_layout(self.nx_graph, prog=layout)
        # pos = nx.spring_layout(nx_graph)
        nx.draw(self.nx_graph, pos=pos, node_size=200)
        # node_labels = {node: f"data: {data['data']}\ncount: {data['count']}\nid: {data['id']}\ntime: {data['time']}\n" for node, data in nx_graph.nodes(data=True)}
        node_labels = {node: f"{data[attr]}" if (not is_leaf) or data['is_leaf'] else '' for node, data in
                       self.nx_graph.nodes(data=True)}
        # print(node_labels)
        nx.draw_networkx_labels(self.nx_graph, pos, labels=node_labels, font_size=5, verticalalignment='center')
        plt.savefig('graph_debug.png')