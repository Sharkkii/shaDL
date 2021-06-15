# DataLoader

import math
import numpy as np


class DataLoader:

    def __init__(self, dataset, n_batch=1, do_shuffle=True):

        self.dataset = dataset
        self.n_dataset = len(dataset)
        self.n_batch = n_batch
        self.do_shuffle = do_shuffle
        self.max_iter = math.floor(self.n_dataset / n_batch)
        # self.max_iter = math.ceil(self.n_dataset / n_batch)
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

    def __init__(self, sequential_dataset, n_batch=1, do_shuffle=False):
        super(SequentialDataLoader, self).__init__(dataset=sequential_dataset, n_batch=n_batch, do_shuffle=do_shuffle)

    def __next__(self):
        if (self.iteration < self.max_iter):
            x, y = zip(*[self.dataset[self.max_iter * _n_batch + self.iteration] for _n_batch in range(self.n_batch)])
            x, y = np.array(x), np.array(y)
            self.iteration += 1
            return x, y
        else:
            self.reset()
            raise StopIteration

    