# Layers

from abc import ABCMeta, abstractclassmethod, abstractmethod

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import functions
from variables import Parameter, ParameterList


class Layer(metaclass=ABCMeta):

    def __init__(self, name=""):
        self.parameters = ParameterList()
        self.name = name
    
    def __setattr__(self, name, value):
        if (isinstance(value, Parameter)):
            self.parameters.append(value)
        super(Layer, self).__setattr__(name, value)
    
    # def parameters(self):
    #     return self.parameters

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    # @abstractmethod
    # def forward(self):
    #     raise NotImplementedError
    
    # @abstractmethod
    # def backward(self):
    #     raise NotImplementedError


class LayerList:

    def __init__(self, *layers):

        self.layers = []
        for layer in layers:
            assert(isinstance(layer, Layer))
        for layer in layers:
            self.layers.append(layer)

    def __iter__(self):
        return iter(self.layers)
    
    def append(self, layer):
        assert(isinstance(layer, Layer))
        self.layers.append(layer)

    def parameters(self):
        parameter_list = ParameterList()
        for layer in self.layers:
            parameter_list.extend(layer.parameters)
        return parameter_list



class Linear(Layer):

    def __init__(self, d_in, d_out, has_bias=True, name=""):
        super(Linear, self).__init__(name=name)

        self.d_in = d_in
        self.d_out = d_out
        self.weight = Parameter(np.random.randn(d_out, d_in), name=name+".weight")
        if (has_bias):
            self.bias = Parameter(np.random.randn(1, d_out), name=name+".bias")
        else:
            self.bias = None
    
    def __call__(self, x):
        w = self.weight
        b = self.bias
        return functions.linear(x, w, b)


class Sigmoid(Layer):

    def __init__(self, name=""):
        super(Sigmoid, self).__init__(name=name)

    def __call__(self, x):
        return functions.sigmoid(x)


class ReLU(Layer):

    def __init__(self, name=""):
        super(ReLU, self).__init__(name=name)
    
    def __call__(self, x):
        return functions.relu(x)


