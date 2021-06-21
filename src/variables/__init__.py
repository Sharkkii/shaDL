from .variable import Variable
from .parameter import Parameter
from .parameter import ParameterList

import functions
Variable.__getitem__ = lambda self, key: functions.getitem(self, key)