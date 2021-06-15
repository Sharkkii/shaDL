# Parameter

from . import Variable
import numpy as np

class Parameter(Variable):

    def __init__(self, data, require_grad=True, name=""):
        assert(isinstance(data, (int, float, np.ndarray, Variable, Parameter)))
        if (isinstance(data, (int, float, np.ndarray))):
            super(Parameter, self).__init__(data)
        elif (isinstance(data, (Variable, Parameter))):
            super(Parameter, self).__init__(data.data)
        self.require_grad = require_grad
        self.name = name
    
    def reset_gradient(self):
        self.grad = None


class ParameterList:

    def __init__(self, *parameters):
        
        self.parameters = []
        for parameter in parameters:
            assert(isinstance(parameter, Parameter))
        for parameter in parameters:
            self.parameters.append(parameter)

    def __iter__(self):
        return iter(self.parameters)
    
    def append(self, parameter):

        assert(isinstance(parameter, Parameter))
        self.parameters.append(parameter)

    def extend(self, parameter_list):

        assert(isinstance(parameter_list, ParameterList))
        self.parameters.extend(parameter_list.parameters)

    
    
