from .function import Function

from .math import add
from .math import sub
from .math import rsub
from .math import mul
from .math import truediv
from .math import rtruediv
from .math import pow
from .math import rpow
from .math import exp
from .math import log
from .math import reshape
from .math import sum
from .math import mean
from .math import dot

from .layer import linear
from .layer import sigmoid
from .layer import relu
from .layer import softmax

from .loss import mean_squared_error

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from variables import Variable

# overload mathematical operators
Variable.__add__ = add
Variable.__radd__ = add
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__truediv__ = truediv
Variable.__rtruediv__ = rtruediv
Variable.__pow__ = pow
Variable.__rpow__ = rpow
