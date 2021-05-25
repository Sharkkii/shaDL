# Loss functions

import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from . import Function


class MeanSquaredError(Function):

    def __init__(self):
        super(Function, self).__init__()
        self.N = None
    
    def forward(self, y_pred, y):
        self.N = y.shape[0]
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def backward(self, dout=None):
        """MeanSquaredError

        Note:
            find out why the gradient of dy_pred & dy are divided by 2
            (seems natural to multiply by 2)
        """
        y_pred = self.inputs[0].data
        y = self.inputs[1].data
        dy_pred = (y_pred - y) / (2 * self.N)
        dy = (y - y_pred) / (2 * self.N)
        return dy_pred, dy

def mean_squared_error(y_pred, y):
    f = MeanSquaredError()
    return f(y_pred, y)
