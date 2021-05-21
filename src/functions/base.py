# Function

from abc import ABCMeta, abstractmethod, abstractclassmethod

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from variables import Variable


class Function(metaclass=ABCMeta):

    def __init__(self):
        self.inputs = None
        self.outputs = None

    def __call__(self, *inputs):
        """
        __call__: Variables -> a Variable | tuple(Variables)
        """
        self.inputs = inputs
        for var in self.inputs:
            var.children = self

        tmp_inputs = tuple([x.data for x in self.inputs])
        tmp_outputs = self.forward(*tmp_inputs)
        tmp_outputs = tmp_outputs if isinstance(tmp_outputs, tuple) else (tmp_outputs,)
        
        self.outputs = tuple([Variable(x) for x in tmp_outputs])
        for var in self.outputs:
            var.parent = self
            var.generation = max([x.generation for x in inputs]) + 1
        outputs = self.outputs if (len(self.outputs) > 1) else self.outputs[0]

        return outputs


    def grad(self, *douts):
        """
        grad: Variables -> a Variable | tuple(Variables)
        """
        tmp_douts = tuple([dout.data for dout in douts])
        tmp_dins = self.backward(*tmp_douts)
        dins = tuple([Variable(din) for din in tmp_dins]) if isinstance(tmp_dins, tuple) else Variable(tmp_dins)
        return dins


    @abstractmethod
    def forward(self):
        """
        forward: numpy.ndarrays -> a numpy.ndarray | tuple(numpy.ndarrays)
        """
        raise NotImplementedError


    @abstractmethod
    def backward(self, dys):
        """
        backward: numpy.ndarrays -> a numpy.ndarray | tuple(numpy.ndarrays)
        """
        raise NotImplementedError


