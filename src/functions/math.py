# Mathematical functions

import numpy as np
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

