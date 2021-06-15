# DataLoader

import numpy as np


class DataLoader:

    def __init__(self, dataset, n_batch=1, do_shuffle=True):

        self.dataset = dataset
        self.n_dataset = len(dataset)
        self.n_batch = n_batch
        self.do_shuffle = do_shuffle
        self.max_iter = np.ceil(self.n_dataset / n_batch)
        self.iteration = None
        self.index = None
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.iteration < self.max_iter):
            x, y = zip(*[self.dataset[index] for index in self.index[self.iteration * self.n_batch : (self.iteration+1) * self.n_batch]])
            self.iteration += 1
            x, y = np.array(x), np.array(y)
            return x, y
        else:
            self.reset()
            raise StopIteration
    
    def reset(self):
        self.iteration = 0
        if (self.do_shuffle):
            self.index = np.random.permutation(np.arange(self.n_dataset))
        else:
            self.index = np.arange(self.n_dataset)
