#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "common.h"
#include "arrayobject.h"

#include "npy_config.h"


NPY_NO_EXPORT int
_zerofill(PyMicArrayObject *ret)
{
    if (PyDataType_REFCHK(PyMicArray_DESCR(ret))) {
        PyErr_SetString(PyExc_TypeError, "micpy do not support Object type");
        return -1;
    }
    else {
        npy_intp n = PyMicArray_NBYTES(ret);
        #pragma omp target device(ret->device)
        memset(PyMicArray_DATA(ret), 0, n);
    }
    return 0;
}

NPY_NO_EXPORT int
_IsAligned(PyMicArrayObject *ap)
{
    unsigned int i;
    npy_uintp aligned;
    npy_uintp alignment = PyMicArray_DESCR(ap)->alignment;

    /* alignment 1 types should have a efficient alignment for copy loops */
    if (PyMicArray_ISFLEXIBLE(ap) || PyMicArray_ISSTRING(ap)) {
        npy_intp itemsize = PyMicArray_ITEMSIZE(ap);
        /* power of two sizes may be loaded in larger moves */
        if (((itemsize & (itemsize - 1)) == 0)) {
            alignment = itemsize > NPY_MAX_COPY_ALIGNMENT ?
                NPY_MAX_COPY_ALIGNMENT : itemsize;
        }
        else {
            /* if not power of two it will be accessed bytewise */
            alignment = 1;
        }
    }

    if (alignment == 1) {
        return 1;
    }
    aligned = (npy_uintp)PyMicArray_DATA(ap);

    for (i = 0; i < PyMicArray_NDIM(ap); i++) {
#if NPY_RELAXED_STRIDES_CHECKING
        /* skip dim == 1 as it is not required to have stride 0 */
        if (PyMicArray_DIM(ap, i) > 1) {
            /* if shape[i] == 1, the stride is never used */
            aligned |= (npy_uintp)PyMicArray_STRIDES(ap)[i];
        }
        else if (PyMicArray_DIM(ap, i) == 0) {
            /* an array with zero elements is always aligned */
            return 1;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        aligned |= (npy_uintp)PyMicArray_STRIDES(ap)[i];
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    return mpy_is_aligned((void *)aligned, alignment);
}

/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype)
{
    PyArray_Descr *base = dtype;
    if (PyDataType_HASSUBARRAY(dtype)) {
        base = dtype->subarray->base;
    }

    return (PyDataType_HASFIELDS(base) ||
            PyDataType_FLAGCHK(base, NPY_ITEM_HASOBJECT) );
}

/**
 * Convert an array shape to a string such as "(1, 2)".
 *
 * @param Dimensionality of the shape
 * @param npy_intp pointer to shape array
 * @param String to append after the shape `(1, 2)%s`.
 *
 * @return Python unicode string
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending)
{
    npy_intp i;
    PyObject *ret, *tmp;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    if (i == n) {
        return PyUString_FromFormat("()%s", ending);
    }
    else {
        ret = PyUString_FromFormat("(%" NPY_INTP_FMT, vals[i++]);
        if (ret == NULL) {
            return NULL;
        }
    }

    for (; i < n; ++i) {
        if (vals[i] < 0) {
            tmp = PyUString_FromString(",newaxis");
        }
        else {
            tmp = PyUString_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        PyUString_ConcatAndDel(&ret, tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    if (i == 1) {
        tmp = PyUString_FromFormat(",)%s", ending);
    }
    else {
        tmp = PyUString_FromFormat(")%s", ending);
    }
    PyUString_ConcatAndDel(&ret, tmp);
    return ret;
}

/* Convert NPY_CASTING to string
 * borrow from numpy */
NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}
