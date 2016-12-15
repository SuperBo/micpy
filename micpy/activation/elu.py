import numpy as np
from chainer.micpy import stream
from chainer.micpy import library as lib
from chainer.micpy import NotSupportedTypeException

def elu_forward(x, alpha):
    y = stream.empty_like(x)
    funcs = (None, lib.elu_forward_float32, lib.elu_forward_float64)
    util.invoke(funcs, x, y, x.size, alpha)
    return y

def elu_backward(x, gy, alpha):
    gx = stream.empty_like(x)
    funcs = (None, lib.elu_backward_float32, lib.elu_backward, float64)
    util.invoke(funcs, x, gy, gx, x.size, alpha)
    return gx
