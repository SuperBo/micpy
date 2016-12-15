import numpy as np
from chainer.micpy import stream
from chainer.micpy import library as lib
from chainer.micpy import NotSupportedTypeException

def relu_forward(x):
    y = stream.empty_like(x)
    funcs = (None, lib.relu_forward_float32, lib.relu_forward_float64)
    util.invoke(funcs, x, y, x.size)
    return y

def relu_backward(x, gy):
    gx = stream.empty_like(x)
    funcs = (None, lib.relu_backward_float32, lib.relu_backward_float64)
    util.invoke(funcs, x, gy, gx, x.size)
   return gx
