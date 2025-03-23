import numpy as np

class BinaryTrie:
  '''
  binary Trie
  '''
  def __init__(self, LEAF_NUM, stride=4):
    self.LEAF_NUM = LEAF_NUM
    self.NODE_NUM = 10 * LEAF_NUM
    self.stride = stride
    self.count = np.zeros(self.NODE_NUM)
    self.children = -np.ones([self.NODE_NUM, 2**stride])
    self.data = np.zeros(self.NODE_NUM)
    self.LAST_INDEX = 0
    self.NUM_SIZE = 256# int32

    # 0000 sign
    self.data[0] = -1
    self.data[1] = 0b0000
    self.LAST_INDEX = 2
    self.ROOT = 0
    self.SIGN = 1

    self.pre_order_arr = np.zeros(LEAF_NUM)

  def _bits(self, np_num):
    cur_num = 0b0
    if self.stride == 1:
        mask = 0b1
    elif self.stride == 4:
        mask = 0b1111
    for bit_pos in range(self.NUM_SIZE, -1, -self.stride):
      bit = (np_num >> bit_pos) & mask
      cur_num = (cur_num << self.stride) | bit
      yield cur_num, bit

  def insert(self, word):
    '''
    insert word and update Trie
    '''
    # root index
    cur_idx = self.ROOT
    for num, bit in self._bits(word):
      ibit = int(bit)
      if ibit == self.SIGN:
          cur_idx = self.SIGN
      elif self.children[cur_idx][ibit] == -1:
        self.children[cur_idx][ibit] = self.LAST_INDEX
        self.LAST_INDEX += 1

        cur_idx = self.children[cur_idx][ibit]
        self.data[cur_idx] = num
      else:
        cur_idx = self.children[cur_idx][ibit]

      self.count[cur_idx] += 1

  def preOrderTraverse(self):
    cur_idx = self.ROOT
    cnt = 0
    last_idx = []

    last_idx.append(-1)
    while cur_idx != -1:
      if self.data[cur_idx] == self.SIGN:
          cur_idx = self.SIGN
      elif self.children[cur_idx][0] != -1:
        last_idx.append(cur_idx)
        cur_idx = self.children[cur_idx][0]
      elif self.children[cur_idx][1] != -1:
        cur_idx = self.children[cur_idx][1]
      else:
        self.pre_order_arr[cnt] = cur_idx
        cnt += 1
        cur_idx = last_idx.pop()

    return self.pre_order_arr

  def retNear(self, window=5):
    pad_size = int(window/2)
    pad_arr = np.concatenate([np.zeros(pad_size), self.pre_order_arr, np.zeros(pad_size)])
    index_arr = np.concatenate([
      np.arange(i, len(self.pre_order_arr)+i)[:, np.new_axis] for i in range(window)
    ])
    near_arr = pad_arr[index_arr].reshape(window, -1).transpose()
    num_arr = self.data[near_arr]
    return num_arr

  def searchNear(self, word):
    '''
    Return before or after words in pre-order traverse
    '''
    pass


  def startsWith(self, prefix):
    '''
    Return if a word starts with prefix exists
    '''
    pass


class StringTrie:
  '''
  String Trie
  format: [Degree(2), 3x [Degree(2), Count(2)]] x3 + '_' x10 -> 62 length string
  '''
  def __init__(self, LEAF_NUM, types=99):
    self.LEAF_NUM = LEAF_NUM
    self.NODE_NUM = 10 * LEAF_NUM
    self.TYPES = types
    self.NUM_SIZE = 62

    self.count = np.zeros(self.NODE_NUM)
    self.children = -np.ones([self.NODE_NUM, self.TYPES])
    self.data = np.empty(self.NODE_NUM, dtype=object)
    self.id = np.zeros(self.NODE_NUM)
    self.time = np.zeros(self.NODE_NUM)
    self.LAST_INDEX = 0

    # self.data[0] = ''
    # self.LAST_INDEX = 1
    # self.ROOT = 0

    self.pre_order_arr = np.zeros(LEAF_NUM)

  def _bits(self, np_num):
    np_num = np_num.split('_')
    np_list = []
    for i in range(len(np_num)):
      stri = np_num[i]
      np_list.append(stri)
      yield '_'.join(np_list), stri

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
    self.id[cur_idx] = id
    self.time[cur_idx] = timestamp

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
        children = self.children[self.children[cur_idx] != -1]
        if len(children) > 0:
          last_idx.extend([children.tolist()])
        else:
          self.pre_order_arr[cnt] = cur_idx
          cnt += 1

    return self.pre_order_arr

  def retNear(self, window=5):
    pad_size = int(window / 2)
    pad_arr = np.concatenate([np.zeros(pad_size), self.pre_order_arr, np.zeros(pad_size)])
    index_arr = np.concatenate([
      np.arange(i, len(self.pre_order_arr) + i)[:, np.new_axis] for i in range(window)
    ])
    near_arr = pad_arr[index_arr].reshape(window, -1).transpose()
    num_arr = self.data[near_arr]
    id_arr = self.id[near_arr]
    time_arr = self.time[near_arr]
    return num_arr, id_arr, time_arr

  def searchNear(self, word):
    '''
    Return before or after words in pre-order traverse
    '''
    pass

  def startsWith(self, prefix):
    '''
    Return if a word starts with prefix exists
    '''
    pass
