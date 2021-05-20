from .base import Function

from .math import add
from .math import sub
from .math import mul
from .math import truediv
from .math import pow

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from variables import Variable

# overload mathematical operators
Variable.__add__ = add
Variable.__sub__ = sub
Variable.__mul__ = mul
Variable.__truediv__ = truediv
Variable.__pow__ = pow
