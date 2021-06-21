# DataLoader

import math
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from variables import Variable


class DataLoader:

    def __init__(self, dataset, n_batch=1, do_shuffle=True):

        self.dataset = dataset
        self.n_dataset = len(dataset)
        self.n_batch = n_batch
        self.do_shuffle = do_shuffle
        assert(self.n_dataset % n_batch == 0)

        self.max_iter = self.n_dataset / n_batch
        self.iteration = None
        self.mapping_table = None
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.iteration < self.max_iter):
            x, y = zip(*[self.dataset[_n_batch] for _n_batch in self.mapping_table[self.iteration * self.n_batch : (self.iteration+1) * self.n_batch]])
            x, y = np.array(x), np.array(y)
            self.iteration += 1
            return x, y
        else:
            self.reset()
            raise StopIteration
    
    def reset(self):
        self.iteration = 0
        if (self.do_shuffle):
            self.mapping_table = np.random.permutation(np.arange(self.n_dataset))
        else:
            self.mapping_table = np.arange(self.n_dataset)


class SequentialDataLoader(DataLoader):

    def __init__(self, sequential_dataset, n_batch=1, l_bptt=-1,  do_shuffle=False):
        self.seq_dataset = sequential_dataset
        self.l_seq = len(sequential_dataset)
        self.l_bptt = l_bptt
        self.n_batch = n_batch
        self.do_shuffle = do_shuffle
        assert(self.l_seq % (n_batch * l_bptt) == 0)

        self.max_iter = int(self.l_seq / (n_batch * l_bptt))
        self.iteration = None
        self.mapping_table = None
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if (self.iteration < self.max_iter):
            x, y = zip(*[self.seq_dataset[_n_batch * int(self.l_seq / self.n_batch) + self.iteration * self.l_bptt + _l_bptt] for _l_bptt in range(self.l_bptt) for _n_batch in range(self.n_batch)])
            x, y = np.array(x), np.array(y)
            x, y = x.reshape(self.l_bptt, self.n_batch, -1), y.reshape(self.l_bptt, self.n_batch, -1)
            self.iteration += 1
            return x, y
        else:
            self.reset()
            raise StopIteration

    def reset(self):
        self.iteration = 0
        if (self.do_shuffle):
            self.mapping_table = np.random.permutation(np.arange(int(self.l_seq / self.l_bptt)))
        else:
            self.mapping_table = np.arange(int(self.l_seq / self.l_bptt))

    