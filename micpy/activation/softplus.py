from chainer.micpy import stream
from chainer.micpy import util
from chainer.micpy import library as lib

def softplus_forward(x, alpha, beta):
    y = stream.empty_like(x)
    funcs = (None, lib.softplus_forward_float32, lib.softmax_forward_float64)
    util.invoke(funcs, x, y, x.size, alpha, beta)
    return y

def softplus_backward(x, gy, alpha, beta):
    gx = stream.empty_like(x)
    funcs = (None, lib.softplus_backward_float32, lib.softmax_backward_float64)
    util.invoke(funcs, x, gy, gx, x.size, alpha, beta)
    return y
