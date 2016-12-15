import numpy as np
from chainer.micpy import stream
from chainer.micpy import library as lib
from chainer.micpy import NotSupportedTypeException

def hard_sigmoid_forward(x):
    y = stream.empty_like(x)
    funcs = (None, lib.hard_sigmoid_forward_float32, lib.hard_sigmoid_forward_float64)
    util.invoke(funcs, x, y, x.size)
    return y

def hard_sigmoid_backward(x, gy):
    gx = stream.empty_like(x)
    funcs = (None, lib.hard_sigmoid_backward_float32, lib.hard_sigmoid_backward_float64)
    util.invoke(funcs, x, gy, gx, x.size)
    return gx
