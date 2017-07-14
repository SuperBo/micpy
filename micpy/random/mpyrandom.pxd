from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.int cimport PyInt_AsLong, PyInt_AS_LONG
from cpython.float cimport PyFloat_AsDouble, PyFloat_AS_DOUBLE
from numpy cimport dtype, npy_intp

cdef extern from "multiarray/arrayobject.h":
    ctypedef extern class micpy.multiarray.ndarray [object PyMicArrayObject]:
        cdef char *data
        cdef int device
    npy_intp PyMicArray_SIZE(ndarray) nogil

cdef extern from "randomkit.h":
    ctypedef struct rk_state:
        pass

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]

    void rk_init(rk_state *state, int ndevice) nogil
    void rk_clean(rk_state *state) nogil
    void rk_seed(unsigned long seed, rk_state *state) nogil
    rk_error rk_randomseed(rk_state *state) nogil
    rk_error rk_devfill(void *buffer, size_t size, int strong) nogil

cdef extern from "distributions.h":
    int rk_fill_bytes(rk_state *state, int device, long size, void *data) nogil
    int rk_dfill_normal(rk_state *state, int device, long length,
                        void *data, double mean, double std_dev) nogil
    int rk_standard_exponential(rk_state *state, int device, long length,
                        void *data) nogil
    int rk_dfill_exponential(rk_state *state, int device, long length,
                        void* data, double scale) nogil
    int rk_dfill_uniform(rk_state *state, int device, long length,
                        void *data, double low, double high) nogil
    int rk_ifill_uniform(rk_state *state, int device, long length,
                        void *data, int low, int high) nogil
    int rk_dfill_gamma(rk_state *state, int device, long length,
                        void *data, double shape, double scale) nogil
    int rk_dfill_beta(rk_state *state, int device, long length,
                        void *data, double a, double b) nogil
    int rk_ifill_binomial(rk_state *state, int device, long length,
                        void *data, int n, double p) nogil
    int rk_ifill_negative_binomial(rk_state *state, int device, long length,
                        void *data, double n, double p) nogil
    int rk_ifill_poisson(rk_state *state, int device, long length,
                        void *data, double lambd) nogil
    int rk_dfill_cauchy(rk_state *state, int device, long length,
                        void *data, double scale) nogil
    int rk_dfill_weibull(rk_state *state, int device, long length,
                        void *data, double shape, double scale) nogil
    int rk_dfill_laplace(rk_state *state, int device, long length,
                        void *data, double mean, double scale) nogil
    int rk_dfill_gumbel(rk_state *state, int device, long length,
                        void *data, double loc, double scale) nogil
    int rk_dfill_lognormal(rk_state *state, int device, long length,
                        void *data, double mean, double sigma) nogil
    int rk_dfill_rayleigh(rk_state *state, int device, long length,
                        void *data, double scale) nogil
    int rk_ifill_geometric(rk_state *state, int device, long length,
                        void *data, double p) nogil
    int rk_ifill_hypergeometric(rk_state *state, int device, long length,
                        void *data, int ngood, int nbad, int nsample) nogil
    int rk_ifill_bernoulli(rk_state *state, int device, long length,
                        void *data, double p) nogil