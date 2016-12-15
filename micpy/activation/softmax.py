import numpy as np

from chainer.micpy import stream
from chainer.micpy import library as lib

def softmax_forward(x):
    y = stream.empty_like(x)
    funcs = (None, lib.softmax_forward_float32, lib.softmax_forward_float64)
    util.invoke(forward_funcs, x, y, x.size)
    return y

def softmax_backward(x, gy):
    gx = stream.empty_like(x)
    funcs = (None, lib.softmax_backward_float32, lib.softmax_backward_float64)
    util.invoke(backward_funcs, x, gy, gx, x.size)
    return gx
