#ifndef _MPY_ARRAY_CTORS_H_
#define _MPY_ARRAY_CTORS_H_

#include "mpy_common.h"

NPY_NO_EXPORT MPY_TARGET_MIC void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize);

NPY_NO_EXPORT MPY_TARGET_MIC void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);

NPY_NO_EXPORT MPY_TARGET_MIC void
byte_swap_vector(void *p, npy_intp n, int size);

NPY_NO_EXPORT PyObject *
PyMicArray_NewFromDescr(int device, PyTypeObject *subtype,
                     PyArray_Descr *descr, int nd,
                     npy_intp *dims, npy_intp *strides, void *data,
                     int flags, PyObject *obj);

NPY_NO_EXPORT PyObject *
PyMicArray_NewFromDescr_int(int device, PyTypeObject *subtype,
                         PyArray_Descr *descr, int nd,
                         npy_intp *dims, npy_intp *strides, void *data,
                         int flags, PyObject *obj, int zeroed,
                         int allow_emptystring);

NPY_NO_EXPORT PyObject *
PyMicArray_New(int device, PyTypeObject *,
                    int nd, npy_intp *,
                    int, npy_intp *, void *, int, int, PyObject *);

NPY_NO_EXPORT PyObject *
PyMicArray_NewLikeArray(int device, PyArrayObject *prototype, NPY_ORDER order,
                     PyArray_Descr *dtype, int subok);

NPY_NO_EXPORT PyObject *
PyMicArray_Empty(int device, int nd, npy_intp *dims,
                    PyArray_Descr *type, int is_f_order);

NPY_NO_EXPORT PyObject *
PyMicArray_Zeros(int device, int nd, npy_intp *dims,
                    PyArray_Descr *type, int is_f_order);


NPY_NO_EXPORT PyObject *
PyMicArray_FromAny(int device, PyObject *op, PyArray_Descr *newtype, int min_depth,
                    int max_depth, int flags, PyObject *context);

NPY_NO_EXPORT PyObject *
PyMicArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int device, int flags);

NPY_NO_EXPORT int
PyMicArray_CopyAnyInto(PyMicArrayObject *dst, PyMicArrayObject *src);

NPY_NO_EXPORT int
PyMicArray_CopyInto(PyMicArrayObject *dst, PyMicArrayObject *src);

NPY_NO_EXPORT int
PyMicArray_CopyIntoHost(PyArrayObject *dst, PyMicArrayObject *src);

NPY_NO_EXPORT int
PyMicArray_CopyIntoFromHost(PyMicArrayObject *dst, PyArrayObject *src);

NPY_NO_EXPORT int
PyMicArray_MoveInto(PyMicArrayObject *dst, PyMicArrayObject *src);

NPY_NO_EXPORT int
PyMicArray_CopyAsFlat(PyMicArrayObject *dst, PyMicArrayObject *src, NPY_ORDER order);

NPY_NO_EXPORT PyObject *
PyMicArray_EnsureArray(PyObject *op, int device);

/* Some utilities function */
NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags);

NPY_NO_EXPORT PyMicArrayObject *
PyMicArray_SubclassWrap(PyMicArrayObject *arr_of_subclass, PyMicArrayObject *towrap);

#endif
