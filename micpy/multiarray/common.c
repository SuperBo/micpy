#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#define _MICARRAYMODULE
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

NPY_NO_EXPORT npy_bool
_IsWriteable(PyMicArrayObject *ap)
{
    PyObject *base=PyMicArray_BASE(ap);
    void *dummy;
    Py_ssize_t n;

    /* If we own our own data, then no-problem */
    if ((base == NULL) || (PyMicArray_FLAGS(ap) & NPY_ARRAY_OWNDATA)) {
        return NPY_TRUE;
    }
    /*
     * Get to the final base object
     * If it is a writeable array, then return TRUE
     * If we can find an array object
     * or a writeable buffer object as the final base object
     * or a string object (for pickling support memory savings).
     * - this last could be removed if a proper pickleable
     * buffer was added to Python.
     *
     * MW: I think it would better to disallow switching from READONLY
     *     to WRITEABLE like this...
     */

    while(PyMicArray_Check(base)) {
        if (PyMicArray_CHKFLAGS((PyMicArrayObject *)base, NPY_ARRAY_OWNDATA)) {
            return (npy_bool) (PyMicArray_ISWRITEABLE((PyMicArrayObject *)base));
        }
        base = PyMicArray_BASE((PyMicArrayObject *)base);
    }

    /*
     * here so pickle support works seamlessly
     * and unpickled array can be set and reset writeable
     * -- could be abused --
     */
    if (PyString_Check(base)) {
        return NPY_TRUE;
    }
    if (PyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        return NPY_FALSE;
    }
    return NPY_TRUE;
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

NPY_NO_EXPORT void
dot_alignment_error(PyMicArrayObject *a, int i, PyMicArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    format = PyUString_FromString("shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    shape1 = convert_shape_to_string(PyMicArray_NDIM(a), PyMicArray_DIMS(a), "");
    shape2 = convert_shape_to_string(PyMicArray_NDIM(b), PyMicArray_DIMS(b), "");

    i_obj = PyLong_FromLong(i);
    j_obj = PyLong_FromLong(j);

    shape1_i = PyLong_FromSsize_t(PyMicArray_DIM(a, i));
    shape2_j = PyLong_FromSsize_t(PyMicArray_DIM(b, j));

    if (!format || !shape1 || !shape2 || !i_obj || !j_obj ||
            !shape1_i || !shape2_j) {
        goto end;
    }

    fmt_args = PyTuple_Pack(6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (fmt_args == NULL) {
        goto end;
    }

    errmsg = PyUString_Format(format, fmt_args);
    if (errmsg != NULL) {
        PyErr_SetObject(PyExc_ValueError, errmsg);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
    }

end:
    Py_XDECREF(errmsg);
    Py_XDECREF(fmt_args);
    Py_XDECREF(format);
    Py_XDECREF(i_obj);
    Py_XDECREF(j_obj);
    Py_XDECREF(shape1);
    Py_XDECREF(shape2);
    Py_XDECREF(shape1_i);
    Py_XDECREF(shape2_j);
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

#define GET_DEVICE(ob, val) ((PyMicArray_Check(ob)) ? \
            PyMicArray_DEVICE((PyMicArrayObject *)ob) : (val))

NPY_NO_EXPORT int
get_common_device2(PyObject *op1, PyObject *op2)
{
    int cpu_device = omp_get_initial_device();
    int dev1, dev2;

    dev1 = GET_DEVICE(op1, cpu_device);
    dev2 = GET_DEVICE(op2, cpu_device);

    /* Prefer current device if devices num are different */
    if (dev1 != dev2) {
        return CURRENT_DEVICE;
    }

    return dev1;
}

NPY_NO_EXPORT int
get_common_device(PyObject **ops, int nop)
{
    int i, idevice, cdevice, cpu_device;
    PyObject *iop;

    cpu_device = omp_get_initial_device();

    cdevice = GET_DEVICE(ops[0], cpu_device);

    for (i = 1; i < nop; ++i) {
        iop = ops[i];

        idevice = GET_DEVICE(iop, cpu_device);

        /* Return current device if devices num are different */
        if (idevice != cdevice) {
            return CURRENT_DEVICE;
        }
    }

    return cdevice;
}