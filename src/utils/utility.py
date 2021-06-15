# utility

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from variables import Variable, Parameter

def as_array(x):
    assert(isinstance(x, (int, float, np.ndarray, Variable, Parameter)))
    if (isinstance(x, (int, float))):
        x = np.array(x)
    elif (isinstance(x, np.ndarray)):
        pass
    elif (isinstance(x, (Variable, Parameter))):
        x = x.data.copy()
    return x

def as_variable(x):
    assert(isinstance(x, (int, float, np.ndarray, Variable, Parameter)))
    if (isinstance(x, (int, float, np.ndarray))):
        x = as_array(x)
        x = Variable(x)
    return x

def as_parameter(x, name=""):
    assert(isinstance(x, (int, float, np.ndarray, Variable, Parameter)))
    if (isinstance(x, (int, float, np.ndarray, Variable))):
        x = as_variable(x)
        x = Parameter(x, name=name)
    return x
    

def recover_shape(x, shape, axis, keepdims):
    """recover_shape

    recover the shape of x (one of whose axis was reduced)

    Args:
        x (Variable): a Variable, one of whose axis was reduced
        shape (tuple<int>): shape
        axis (int | None): axis, which was used to get x
        keepdims (bool): keepdims, which was used to get x
    """
    if (axis is None):
        whole = (1,) * len(shape)
    else:
        first_half = x.shape[:axis]
        second_half = x.shape[axis+1:]
        if (keepdims):
            whole = first_half + second_half
        else:
            whole = first_half + (1,) + second_half
    x = x.reshape(*whole)
    x = np.broadcast_to(x, shape)
    return x