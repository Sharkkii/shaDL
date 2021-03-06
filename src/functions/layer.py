# Layer functions

import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from . import Function


class Linear(Function):

    def __init__(self):
        super(Linear, self).__init__()
    
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


class Sigmoid(Function):

    def __init__(self):
        super(Sigmoid, self).__init__()
    
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


class Tanh(Function):

    def __init__(self):
        super(Tanh, self).__init__()
    
    def forward(self, a):
        b = (np.exp(a) - np.exp(- a)) / (np.exp(a) + np.exp(- a))
        return b
    
    def backward(self, db):
        b = self.outputs[0].data
        da = db * (1 - b**2)
        return da

def tanh(a):
    f = Tanh()
    return f(a)


class ReLU(Function):

    def __init__(self):
        super(ReLU, self).__init__()
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
        super(Softmax, self).__init__()
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