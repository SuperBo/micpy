from chainer.micpy import util
from chainer.micpy import stream
from chainer.micpy import library as lib

def tanh_forward(x):
    y = stream.empty_like(x)
    funcs = (None, lib.tanh_forward_float32, lib.tanh_forward_float64)
    util.invoke(funcs, x, y, x.size)
    return y

def tanh_backward(x, gy):
    gx = stream.empty_like(x)
    funcs = (None, lib.tanh_backward_float32, lib.tanh_forward_float64)
    util.invoke(funcs, x, gy, gx, x.size)
    return gx
