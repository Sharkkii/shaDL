# Layer functions

import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from . import Function


class Linear(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, x, w, b):
        """Linear.forward

        Args:
            x(numpy.ndarray<N,d_in>)
            w(numpy.ndarray<d_out,d_in>)
            b(numpy.ndarray<1,d_out>)
        
        Note:
            take care of the shape of b; b should be 2-dimensional array so that it can broadcast
        """
        y = np.dot(x, w.T) + b
        return y
    
    def backward(self, dy):
        x = self.inputs[0].data
        w = self.inputs[1].data
        dx = np.dot(dy, w)
        dw = np.dot(dy.T, x)
        db = np.sum(dy, axis=0, keepdims=True)
        return dx, dw, db
    
def linear(x, w, b):
    f = Linear()
    return f(x, w, b)