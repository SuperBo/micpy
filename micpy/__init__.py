from pymic import OffloadArray as ndarray
from pymic import OffloadStream as stream

from . import device
from .util import get_array_module
from .core import *

from .math.numeric import vdot, dot, matmul_transA, matmul_transB
from .math.numeric import sum, argmax, argmin

from .statistics import random

from .dnn import grad_decrease
from .dnn import relu, relu_grad
from .dnn import sigmoid, sigmoid_grad
from .dnn import tanh, tanh_grad
