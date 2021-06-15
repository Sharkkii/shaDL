# Neural Networks

from abc import ABCMeta, abstractclassmethod, abstractmethod

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from . import Layer, LayerList


class Network(metaclass=ABCMeta):

    def __init__(self, layers, name=""):
        assert(isinstance(layers, LayerList))
        self.layers = layers
        self.name = name
    
    def __setattr__(self, name, value):
        if (isinstance(value, Layer)):
            self.layers.append(value)
        super().__setattr__(name, value)

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    # @abstractmethod
    # def forward(self):
    #     raise NotImplementedError
    
    # @abstractmethod
    # def backward(self):
    #     raise NotImplementedError

    def parameters(self):
        return self.layers.parameters()


class FeedForwardNeuralNetwork(Network):

    def __init__(self, layers, name=""):
        super(FeedForwardNeuralNetwork, self).__init__(layers, name)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
