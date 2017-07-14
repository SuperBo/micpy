import numpy as np
import micpy.multiarray as mp
import operator

try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

cimport numpy
cimport mpyrandom

ctypedef mpyrandom.ndarray micarray

cdef class RandomState:
    """
    RandomState(seed=None)

    `RandomState` exposes a number of methods for generating random numbers
    drawn from a variety of probability distributions. In addition to the
    distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    array filled with generated values is returned. If `size` is a tuple,
    then an array with that shape is filled and returned.

    *Compatibility Guarantee*
    A fixed seed and a fixed series of calls to 'RandomState' methods using
    the same parameters will always produce the same results up to roundoff
    error except when the values were incorrect. Incorrect values will be
    fixed and the NumPy version in which the fix was made will be noted in
    the relevant docstring. Extension of existing parameter ranges and the
    addition of new parameters is allowed as long the previous behavior
    remains unchanged.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.  Can
        be any integer between 0 and 2**32 - 1 inclusive, an array (or other
        sequence) of such integers, or ``None`` (the default).  If `seed` is
        ``None``, then `RandomState` will try to read data from
        ``/dev/urandom`` (or the Windows analogue) if available or seed from
        the clock otherwise.

    Notes
    -----
    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator with a number of methods that are similar
    to the ones available in `RandomState`. `RandomState`, besides being
    NumPy-aware, has the advantage that it provides a much larger number
    of probability distributions to choose from.

    """
    cdef rk_state *internal_state
    cdef object lock

    def __init__(self, seed=None):
        cdef rk_state *state = <rk_state*>PyMem_Malloc(sizeof(rk_state))
        rk_init(state, mp.ndevices)
        self.internal_state = state
        self.lock = Lock()
        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            rk_clean(<rk_state*>self.internal_state)
            PyMem_Free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called when `RandomState` is initialized. It can be
        called again to re-seed the generator. For details, see `RandomState`.

        Parameters
        ----------
        seed : int or array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        See Also
        --------
        RandomState

        """
        cdef rk_error errcode
        if seed is None:
            with self.lock:
                errcode = rk_randomseed(self.internal_state)
        else:
            idx = operator.index(seed)
            if idx > int(2**32 - 1) or idx < 0:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            with self.lock:
                rk_seed(idx, self.internal_state)

    def bytes(self, length):
        if length is None or length == 0:
            return None

        cdef long lsize
        cdef micarray arr

        arr = <micarray> mp.empty(length, np.ubyte)
        lsize = PyInt_AsLong(length)

        with self.lock, nogil:
            rk_fill_bytes(self.internal_state, arr.device, lsize, arr.data)
        return arr

    def random_sample(self, size=None):
        return self.uniform(size=size)

    def rand(self, *args):
        if len(args) == 0:
            return self.random_sample()
        else:
            return self.random_sample(size=args)

    def randn(self, *args):
        if len(args) == 0:
            return self.standard_normal()
        else:
            return self.standard_normal(args)

    def randint(self, low, high=None, size=None):
        cdef int ilow, ihigh
        cdef long lsize
        cdef micarray arr

        if high is None:
            high = low
            low = 0
        if high < low:
            raise ValueError("high < low")
        ilow = <int> PyInt_AsLong(low)
        ihigh = <int> PyInt_AsLong(high)

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_uniform(self.internal_state, arr.device, lsize,
                arr.data, ilow, ihigh)
        return arr

    def uniform(self, low=0.0, high=1.0, size=None):
        cdef double flow, fhigh
        cdef long lsize
        cdef micarray arr

        if (high < low):
            raise ValueError("'high' < 'low'")
        flow = PyFloat_AsDouble(low)
        fhigh = PyFloat_AsDouble(high)

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_uniform(self.internal_state, arr.device, lsize,
                arr.data, flow, fhigh)
        return arr

    # Complicated, continuous distributions:
    def standard_normal(self, size=1):
        """
        standard_normal(size=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).
        """
        return self.normal(size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        cdef double floc, fscale
        cdef long lsize
        cdef micarray arr

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_normal(self.internal_state, arr.device, lsize,
                arr.data, floc, fscale)
        return arr

    def beta(self, a, b, size=None):
        cdef double fa, fb
        cdef long lsize
        cdef micarray arr

        fa = PyFloat_AsDouble(a)
        fb = PyFloat_AsDouble(b)

        if fa <= 0:
            raise ValueError("a <= 0")
        if fb <= 0:
            raise ValueError("b <= 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_beta(self.internal_state, arr.device, lsize,
                arr.data, fa, fb)
        return arr

    def standard_exponential(self, size=None):
        return self.exponential(size=size)

    def exponential(self, scale=1.0, size=None):
        cdef double fscale
        cdef long lsize
        cdef micarray arr

        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_exponential(self.internal_state, arr.device, lsize,
                arr.data, fscale)
        return arr

    def standard_gamma(self, shape, size=None):
        return self.gamma(shape=shape, size=size)

    def gamma(self, shape, scale=1.0, size=None):
        cdef double fshape, fscale
        cdef long lsize

        fshape = PyFloat_AsDouble(shape)
        if np.signbit(fshape):
            raise ValueError("shape < 0")
        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_gamma(self.internal_state, arr.device, lsize,
                arr.data, fshape, fscale)
        return arr

    def standard_cauchy(self, size=None):
        return self.cauchy(size=size)

    def cauchy(self, scale=1.0, size=None):
        cdef double fscale
        cdef long lsize

        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_cauchy(self.internal_state, arr.device, lsize,
                arr.data, fscale)
        return arr

    def weibull(self, a, size=None):
        cdef double fa
        cdef long lsize
        cdef micarray arr

        fa = PyFloat_AsDouble(a)
        if np.signbit(fa):
            raise ValueError("a < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_weibull(self.internal_state, arr.device, lsize,
                arr.data, fa, 1.0)
        return arr

    def laplace(self, loc=0.0, scale=1.0, size=None):
        cdef double floc, fscale
        cdef long lsize
        cdef micarray arr

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)
        with self.lock, nogil:
            rk_dfill_laplace(self.internal_state, arr.device, lsize,
                arr.data, floc, fscale)
        return arr

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        cdef double floc, fscale
        cdef long lsize
        cdef micarray arr

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)
        with self.lock, nogil:
            rk_dfill_gumbel(self.internal_state, arr.device, lsize,
                arr.data, floc, fscale)
        return arr

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        cdef double fmean, fsigma
        cdef long lsize
        cdef micarray arr

        fmean = PyFloat_AsDouble(mean)
        fsigma = PyFloat_AsDouble(sigma)
        if np.signbit(fsigma):
            raise ValueError("sigma < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_lognormal(self.internal_state, arr.device, lsize,
                arr.data, fmean, fsigma)
        return arr

    def rayleigh(self, scale=1.0, size=None):
        cdef double fscale
        cdef long lsize
        cdef micarray arr

        fscale = PyFloat_AsDouble(scale)
        if np.signbit(fscale):
            raise ValueError("scale < 0")

        arr = <micarray> mp.empty(size, dtype=np.float)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_dfill_rayleigh(self.internal_state, arr.device, lsize,
                arr.data, fscale)
        return arr

    # Complicated, discrete distributions:
    def binomial(self, n, p, size=None):
        cdef long ln, lsize
        cdef double fp
        cdef micarray arr

        fp = PyFloat_AsDouble(p)
        ln = PyInt_AsLong(n)

        if ln < 0:
            raise ValueError("n < 0")
        if fp < 0:
            raise ValueError("p < 0")
        elif fp > 1:
            raise ValueError("p > 1")
        elif np.isnan(fp):
            raise ValueError("p is nan")

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_binomial(self.internal_state, arr.device, lsize,
                arr.data, <int> ln, fp)
        return arr

    def negative_binomial(self, n, p, size=None):
        cdef double fn, fp
        cdef long lsize
        cdef micarray arr

        fp = PyFloat_AsDouble(p)
        fn = PyFloat_AsDouble(n)

        if fn <= 0:
            raise ValueError("n <= 0")
        if fp < 0:
            raise ValueError("p < 0")
        elif fp > 1:
            raise ValueError("p > 1")

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_negative_binomial(self.internal_state, arr.device, lsize,
                arr.data, fn, fp)
        return arr

    def poisson(self, lam=1.0, size=None):
        cdef double flam
        cdef long lsize
        cdef micarray arr

        flam = PyFloat_AsDouble(lam)
        if flam < 0:
            raise ValueError("lam < 0")

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_poisson(self.internal_state, arr.device, lsize,
                arr.data, flam)
        return arr

    def bernoulli(self, p, size=None):
        cdef double fp
        cdef long lsize
        cdef micarray arr

        fp = PyFloat_AsDouble(p)
        if fp < 0.0:
            raise ValueError("p < 0.0")
        if fp > 1.0:
            raise ValueError("p > 1.0")

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_bernoulli(self.internal_state, arr.device, lsize,
                arr.data, fp)
        return arr

    def geometric(self, p, size=None):
        cdef double fp
        cdef long lsize
        cdef micarray arr

        fp = PyFloat_AsDouble(p)
        if fp < 0.0:
            raise ValueError("p < 0.0")
        if fp > 1.0:
            raise ValueError("p > 1.0")

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_geometric(self.internal_state, arr.device, lsize,
                arr.data, fp)
        return arr

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        cdef long lngood, lnbad, lnsample, lsize
        cdef micarray arr

        lngood = PyInt_AsLong(ngood)
        lnbad = PyInt_AsLong(nbad)
        lnsample = PyInt_AsLong(nsample)

        if lngood < 0:
            raise ValueError("ngood < 0")
        if lnbad < 0:
            raise ValueError("nbad < 0")
        if lnsample < 1:
            raise ValueError("nsample < 1")
        if lngood + lnbad < lnsample:
            raise ValueError("ngood + nbad < nsample")

        arr = <micarray> mp.empty(size, dtype=np.int32)
        lsize = PyInt_AS_LONG(arr.size)

        with self.lock, nogil:
            rk_ifill_hypergeometric(self.internal_state, arr.device, lsize,
                arr.data, <int> lngood, <int> lnbad, <int> lnsample)
        return arr
