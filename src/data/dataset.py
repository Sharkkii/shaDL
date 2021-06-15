# DataLoader

import numpy as np


class Dataset:

    def __init__(self, X, y=None):
        assert((y is None) or (len(X) == len(y)))
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, key):
        X = self.X[key]
        y = None if (self.y is None) else self.y[key]
        return X, y
    
    def __setitem__(self, key, value):
        raise NotImplementedError
