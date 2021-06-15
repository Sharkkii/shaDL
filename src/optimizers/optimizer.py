# Optimizers

from abc import ABCMeta, abstractmethod, abstractclassmethod

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from variables import ParameterList


class Optimizer(metaclass=ABCMeta):

    def __init__(self, parameters):
        assert(isinstance(parameters, ParameterList))
        self.parameters = parameters

    def reset(self):
        for parameter in self.parameters:
            parameter.reset_gradient()

    # @abstractmethod
    # def set(self):
    #     raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD (Stochastic Gradient Descent)
    """
    def __init__(self, parameters, lr):
        super(SGD, self).__init__(parameters)
        self.lr = lr
    
    def step(self):
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.grad.data
