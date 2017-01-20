from pymic import OffloadArray
from numpy import empty as nempty


class ndarray(OffloadArray):
    """ Wrapper of pymic OffloadArray.
        with all operation are synchronized
    """
    def __init__(self, *args, **kwargs):
        super(ndarray, self).__init__(*args, **kwargs)

    def __int__(self):
        if self.size > 1:
            raise TypeError("only length-1 arrays can be "
                    "converted to Python scalars")
        if self.array is None:
            self.array = nempty(self.shape, self.dtype)
        self.update_host()
        self.stream.sync()
        return int(self.array)

    def __float__(self):
        if self.size > 1:
            raise TypeError("only length-1 arrays can be "
                    "converted to Python scalars")
        if self.array is None:
            self.array = nempty(self.shape, self.dtype)
        self.update_host()
        self.stream.sync()
        return float(self.array)

    def __complex__(self):
        if self.size > 1:
            raise TypeError("only length-1 arrays can be "
                    "converted to Python scalars")
        if self.array is None:
            self.array = nempty(self.shape, self.dtype)
        self.update_host()
        self.stream.sync()
        return complex(self.array)

    def reverse(self):
        r = super(ndarray, self).reverse()
        self.stream.sync()
        return r

    def __add__(self, other):
        r = super(ndarray, self).__add__(other)
        self.stream.sync()
        return r

    def __iadd__(self, other):
        r = super(ndarray, self).__iadd__(other)
        self.stream.sync()
        return r

    def __radd__(self, other):
        r = super(ndarray, self).__radd__(other)
        self.stream.sync()
        return r

    def __sub__(self, other):
        r = super(ndarray, self).__sub__(other)
        self.stream.sync()
        return r

    def __isub__(self, other):
        r = super(ndarray, self).__isub__(other)
        self.stream.sync()
        return r

    def __rsub__(self, other):
        r = super(ndarray, self).__rsub__(other)
        self.stream.sync()
        return r

    def __mul__(self, other):
        r = super(ndarray, self).__mul__(other)
        self.stream.sync()
        return r

    def __imul__(self, other):
        r = super(ndarray, self).__imul__(other)
        self.stream.sync()
        return r

    def __rmul__(self, other):
        r = super(ndarray, self).__rmul__(other)
        self.stream.sync()
        return r

    def __div__(self, other):
        r = super(ndarray, self).__div__(other)
        self.stream.sync()
        return r

    def __idiv__(self, other):
        r = super(ndarray, self).__idiv__(other)
        self.stream.sync()
        return r

    def __rdiv__(self, other):
        r = super(ndarray, self).__rdiv__(other)
        self.stream.sync()
        return r

    def __pow__(self, other):
        r = super(ndarray, self).__pow__(self, other)
        self.stream.sync()
        return r

    def dot(self, other):
        return numeric.dot(self, other)

    def vdot(self, other):
        return numeric.dot(self, other)

    def sum(self, axis=None):
        return numer.sum(self, axis)

    def transpose(self, *axes):
        pass

    @property
    def T(self):
        pass
