# Activation functions

import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import utility
from . import Function


class Sigmoid(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a):
        b = np.exp(np.minimum(0, a)) / (1 + np.exp(- np.abs(a)))
        return b
    
    def backward(self, db):
        b = self.outputs[0].data
        da = db * (b * (1 - b))
        return da

def sigmoid(a):
    f = Sigmoid()
    return f(a)


class ReLU(Function):

    def __init__(self):
        super(Function, self).__init__()
        self.mask = None
    
    def forward(self, a):
        self.mask = np.where(a > 0, 1, 0)
        b = a * self.mask
        return b
    
    def backward(self, db):
        da = db * self.mask
        return da

def relu(a):
    f = ReLU()
    return f(a)


class Softmax(Function):
    """Softmax (for 2-dimensional Variable)
    """

    def __init__(self, axis=None, keepdims=False):
        super(Function, self).__init__()
        self.axis = 1
        self.keepdims = True
    
    def forward(self, a):
        """Softmax.forward

        Args:
            a(numpy.ndarray<N,M>)
    
        Returns:
            b(numpy.ndarray<N> | numpy.ndarray<N,1>)
        """
        b = a - np.max(a, axis=self.axis, keepdims=self.keepdims)
        b = np.exp(b)
        b = b / np.sum(b, axis=self.axis, keepdims=self.keepdims)
        return b
    
    def backward(self, db):
        """Softmax.backward

        Args:
            db(numpy.ndarray<N> | numpy.ndarray<N,1>)
    
        Returns:
            da(numpy.ndarray<N,M>)
        """
        b = self.outputs[0].data
        da = b * db - np.sum(b * db, axis=self.axis, keepdims=self.keepdims) * b
        return da

def softmax(a, axis=None, keepdims=False):
    f = Softmax(axis=axis, keepdims=keepdims)
    return f(a)
