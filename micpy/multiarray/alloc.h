#ifndef _NPY_ARRAY_ALLOC_H_
#define _NPY_ARRAY_ALLOC_H_
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

NPY_NO_EXPORT void *
mpy_alloc_cache(npy_uintp sz, int device);

NPY_NO_EXPORT void *
mpy_alloc_cache_zero(npy_uintp sz, int device);

NPY_NO_EXPORT void
mpy_free_cache(void * p, npy_uintp sd, int device);

NPY_NO_EXPORT void *
mpy_alloc_cache_dim(npy_uintp sz);

NPY_NO_EXPORT void
mpy_free_cache_dim(void * p, npy_uintp sd);

NPY_NO_EXPORT void *
PyDataMemMic_NEW(size_t sz, int device);

NPY_NO_EXPORT void *
PyDataMemMic_NEW_ZEROED(size_t sz, size_t elsize, int device);

NPY_NO_EXPORT void *
PyDataMemMic_RENEW(void * p, size_t sz, int device);

NPY_NO_EXPORT void
PyDataMemMic_FREE(void * p, int device);

NPY_NO_EXPORT PyObject *
PyMicArray_NewFromDescr_int(int dev, PyTypeObject *subtype,
                            PyArray_Descr *descr, int nd, npy_intp *dims,
                            npy_intp *strides, void *data, int flags,
                            PyObject *obj, int zeroed, int allow_emptystring);

NPY_NO_EXPORT PyObject *
PyMicArray_NewFromDescr(int dev, PyTypeObject *subtype,
                        PyArray_Descr *descr, int nd, npy_intp *dims,
                        npy_intp *strides, void *data, int flags,
                        PyObject *obj);
#endif
