# utility

import numpy as np

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