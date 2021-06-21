# Layers

from abc import ABCMeta, abstractclassmethod, abstractmethod

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import functions
from variables import Variable, Parameter, ParameterList


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


class RNNCell(Layer):

    def __init__(self, d_in, d_hidden, has_bias=True, name=""):
        super(RNNCell, self).__init__(name=name)

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.hidden_state = None
        self.linear_x2h = Linear(d_in=d_in, d_out=d_hidden, has_bias=has_bias)
        self.linear_h2h = Linear(d_in=d_hidden, d_out=d_hidden, has_bias=has_bias)

    def __call__(self, x, h=None):
        """RNNCell.__call__

        Args:
            x(numpy.ndarray<n_batch,d_in>)
            h(numpy.ndarray<n_batch,d_hidden>)
        """
        if (self.hidden_state is None):
            n_batch = x.shape[0]
            self.hidden_state = Parameter(h) if (h is not None) else Parameter(np.zeros((n_batch, self.d_hidden)))
        y = self.hidden_state = functions.tanh(self.linear_x2h(x) + self.linear_h2h(self.hidden_state))
        return y, self.hidden_state
    
    def reset(self):
        self.hidden_state = None


class RNN(Layer):

    def __init__(self, d_in, d_hidden, has_bias=True, is_stateful=False, name=""):
        super(RNN, self).__init__(name=name)
        self.rnn_cell = RNNCell(d_in=d_in, d_hidden=d_hidden, has_bias=has_bias, name=name)
        self.is_stateful = is_stateful

    def __call__(self, x, h=None):
        """RNN.__call__

        Args:
            x(numpy.ndarray<l_seq,n_batch,d_in>)
            h(numpy.ndarray<n_batch,d_hidden>)
        """
        if (not self.is_stateful):
            self.rnn_cell.reset()
        l_seq, n_batch = x.shape[0], x.shape[1]
        y = Variable(np.zeros((l_seq, n_batch, self.rnn_cell.d_hidden)))
        for idx in range(l_seq):
            x_idx = x.data[idx]
            y_idx, h = self.rnn_cell(x=Variable(x_idx), h=h)
            y.data[idx] = y_idx.data
        return y, h


class LSTMCell(Layer):

    def __init__(self, d_in, d_hidden, has_bias=True, name=""):
        super(LSTMCell, self).__init__(name=name)

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.hidden_state = None
        self.memory_cell = None
        # (forget-gate, gain, input-gate, output-gate)
        self.linear_x2h = Linear(d_in=d_in, d_out=d_hidden * 4, has_bias=has_bias)
        self.linear_h2h = Linear(d_in=d_hidden, d_out=d_hidden * 4, has_bias=has_bias)

    def __call__(self, x, h=None, c=None):
        """LSTMCell.__call__

        Args:
            x(numpy.ndarray<n_batch,d_in>)
            h(numpy.ndarray<n_batch,d_hidden>)
            c(numpy.ndarray<n_batch,d_hidden>)
        """
        if (self.hidden_state is None):
            n_batch = x.shape[0]
            self.hidden_state = Parameter(h) if (h is not None) else Parameter(np.zeros((n_batch, self.d_hidden)))
        if (self.memory_cell is None):
            n_batch = x.shape[0]
            self.memory_cell = Parameter(c) if (c is not None) else Parameter(np.zeros((n_batch, self.d_hidden)))
        
        fgio = self.linear_x2h(x) + self.linear_h2h(self.hidden_state)
        forget_gate = functions.sigmoid(fgio[:, :self.d_hidden])
        gain = functions.tanh(fgio[:, self.d_hidden:2*self.d_hidden])
        input_gate = functions.sigmoid(fgio[:, 2*self.d_hidden:3*self.d_hidden])
        output_gate = functions.sigmoid(fgio[:, 3*self.d_hidden:])

        self.memory_cell = forget_gate * self.memory_cell + input_gate * gain
        y = self.hidden_state = output_gate * functions.tanh(self.memory_cell)
        return y, self.hidden_state
    
    def reset(self):
        self.hidden_state = None
        self.memory_cell = None


class LSTM(Layer):

    def __init__(self, d_in, d_hidden, has_bias=True, is_stateful=False, name=""):
        super(LSTM, self).__init__(name=name)
        self.lstm_cell = LSTMCell(d_in=d_in, d_hidden=d_hidden, has_bias=has_bias, name=name)
        self.is_stateful = is_stateful

    def __call__(self, x, h=None, c=None):
        """LSTM.__call__

        Args:
            x(numpy.ndarray<l_seq,n_batch,d_in>)
            h(numpy.ndarray<n_batch,d_hidden>)
            c(numpy.ndarray<n_batch,d_hidden>)
        """
        if (not self.is_stateful):
            self.lstm_cell.reset()
        l_seq, n_batch = x.shape[0], x.shape[1]
        y = Variable(np.zeros((l_seq, n_batch, self.lstm_cell.d_hidden)))
        for idx in range(l_seq):
            x_idx = x.data[idx]
            y_idx, h = self.lstm_cell(x=Variable(x_idx), h=h)
            y.data[idx] = y_idx.data
        return y, h


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


