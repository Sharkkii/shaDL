# Mathematical functions

import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import utility
from . import Function


class Add(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a, b):
        c = a + b
        return c
    
    def backward(self, dc):
        da = dc
        db = dc
        return da, db

def add(a, b):
    f = Add()
    return f(a, b)


class Sub(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a, b):
        c = a - b
        return c
    
    def backward(self, dc):
        da = dc
        db = - dc
        return da, db

def sub(a, b):
    f = Sub()
    return f(a, b)


class Mul(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a, b):
        c = a * b
        return c
    
    def backward(self, dc):
        a, b = self.inputs[0].data, self.inputs[1].data
        da = dc * b
        db = dc * a
        return da, db

def mul(a, b):
    f = Mul()
    return f(a, b)


class Truediv(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a, b):
        c = a / b
        return c
    
    def backward(self, dc):
        a, b = self.inputs[0].data, self.inputs[1].data
        da = dc / b
        db = - dc * a * (b ** (-2.0))
        return da, db

def truediv(a, b):
    f = Truediv()
    return f(a, b)


class Pow(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a, b):
        c = a ** b
        return c
    
    def backward(self, dc):
        a, b = self.inputs[0].data, self.inputs[1].data
        c = self.outputs[0].data
        da = dc * (b * c / a)
        db = dc * (c * np.log(a))
        return da, db

def pow(a, b):
    f = Pow()
    return f(a, b)


class Exp(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a):
        b = np.exp(a)
        return b
    
    def backward(self, db):
        b = self.outputs[0].data
        da = db * b
        return da

def exp(a):
    f = Exp()
    return f(a)


class Log(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a):
        b = np.log(a)
        return b
    
    def backward(self, db):
        a = self.inputs[0].data
        da = db / a
        return da

def log(a):
    f = Log()
    return f(a)


class Reshape(Function):

    def __init__(self, shape):
        super(Function, self).__init__()
        self.shape = shape
    
    def forward(self, a):
        b = np.reshape(a, self.shape)
        return b
    
    def backward(self, db):
        a = self.inputs[0].data
        da = np.reshape(db, a.shape)
        return da

def reshape(a, shape):
    f = Reshape(shape=shape)
    return f(a)


class Sum(Function):

    def __init__(self, axis=None, keepdims=False):
        super(Function, self).__init__()
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, a):
        b = np.sum(a, axis=self.axis, keepdims=self.keepdims)
        return b
    
    def backward(self, db):
        a = self.inputs[0].data
        da = utility.recover_shape(db, shape=a.shape, axis=self.axis, keepdims=self.keepdims)
        return da

def sum(a, axis=None, keepdims=False):
    f = Sum(axis=axis, keepdims=keepdims)
    return f(a)


class Mean(Function):

    def __init__(self, axis=None, keepdims=False):
        super(Function, self).__init__()
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, a):
        b = np.mean(a, axis=self.axis, keepdims=self.keepdims)
        return b
    
    def backward(self, db):
        a = self.inputs[0].data
        b = self.outputs[0].data
        db = db / (a.size / b.size)
        da = utility.recover_shape(db, shape=a.shape, axis=self.axis, keepdims=self.keepdims)
        return da

def mean(a, axis=None, keepdims=False):
    f = Mean(axis=axis, keepdims=keepdims)
    return f(a)


class Dot(Function):

    def __init__(self):
        super(Function, self).__init__()
    
    def forward(self, a, b):
        """Dot.forward

        Args:
            a(numpy.ndarray<M>)
            b(numpy.ndarray<M>)
        
        Note:
            take care of the shape of a, b; they must be 1-dimensional array
        """
        c = np.dot(a, b)
        return c
    
    def backward(self, dc):
        a = self.inputs[0].data
        b = self.inputs[1].data
        da = dc * b
        db = dc * a
        return da, db
    
def dot(a, b):
    f = Dot()
    return f(a, b)
