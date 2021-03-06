# Variable

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

class Variable:

    def __init__(self, data):
        assert(isinstance(data, (int, float, np.ndarray, Variable)))
        if (isinstance(data, (int, float))):
            self.data = np.array(data)
        elif (isinstance(data, np.ndarray)):
            self.data = data.copy()
        elif (isinstance(data, Variable)):
            self.data = data.data.copy()
        self.grad = None

        self.shape = data.shape
        self.parent = None
        self.children = []
        self.generation = 0

    def __getitem__(self, key):
        """Variable.__getitem__

        implemented in src/variables/__init__.py
        """
        raise NotImplementedError
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
    
    def backward(self):
        if (self.grad is None):
            self.grad = Variable(np.ones_like(self.data))

        vars = [ self ]
        while (len(vars) > 0):
            idx = np.argmax([x.generation for x in vars])
            var = vars.pop(idx)
            fun = var.parent
            if (fun is not None):
                vars_fun = fun.inputs
                grads_fun = fun.grad(var.grad)
                grads_fun = grads_fun if isinstance(grads_fun, tuple) else (grads_fun,)
                for var_fun, grad_fun in zip(vars_fun, grads_fun):
                    var_fun.grad = grad_fun if (var_fun.grad is None) else (var_fun.grad + grad_fun)
                vars.extend(vars_fun)

    # def ancestors(self):
    #     return self.parent

    # def descendants(self):
    #     return self.children


        

        


    
