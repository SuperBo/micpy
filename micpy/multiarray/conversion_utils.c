#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/npy_3kcompat.h>

#include "common.h"
#include "conversion_utils.h"

/****************************************************************
* Useful function for conversion when used with PyArg_ParseTuple
****************************************************************/

/*
 * Useful to pass as converter function for O& processing in PyArgs_ParseTuple.
 *
 * This conversion function can be used with the "O&" argument for
 * PyArg_ParseTuple.  It will immediately return an object of array type
 * or will convert to a NPY_ARRAY_CARRAY any other object.
 *
 * If you use PyArray_GeneralConverter, you must DECREF the array when finished
 * as you get a new reference to it.
 */
NPY_NO_EXPORT int
PyMicArray_GeneralConverter(PyObject *object, PyObject **address)
{
    if (PyArray_Check(object) || PyMicArray_Check(object)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    }
    else {
        *address = PyArray_FromAny(object, NULL, 0, 0,
                                NPY_ARRAY_CARRAY, NULL);
        if (*address == NULL) {
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}


NPY_NO_EXPORT int
PyMicArray_Converter(PyObject *object, PyObject **address)
{
    if (PyMicArray_Check(object)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    }

    return NPY_FAIL;
}


NPY_NO_EXPORT int
PyMicArray_DeviceConverter(PyObject *object, int *device)
{
    /* Leave value untouched while object is None or NULL */
    if (object == Py_None || object == NULL) {
        return NPY_SUCCEED;
    }

    int dev = PyArray_PyIntAsInt(object);
    if (dev >= 0 && dev < NDEVICES) {
        *device = dev;
        return NPY_SUCCEED;
    }
    else {
        PyErr_Format(PyExc_ValueError, "device must be in range "
                        "[%d,%d)", 0, NDEVICES);
        return NPY_FAIL;
    }
}

/*NUMPY_API
 * Useful to pass as converter function for O& processing in
 * PyArgs_ParseTuple for output arrays
 */
NPY_NO_EXPORT int
PyMicArray_OutputConverter(PyObject *object, PyMicArrayObject **address)
{
    if (object == NULL || object == Py_None) {
        *address = NULL;
        return NPY_SUCCEED;
    }
    if (PyMicArray_Check(object)) {
        *address = (PyMicArrayObject *)object;
        return NPY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "output must be an array");
        *address = NULL;
        return NPY_FAIL;
    }
}

/* Borrow from Numpy */

static npy_intp
_PyArray_PyIntAsIntp_ErrMsg(PyObject *o, const char * msg)
{
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
    long long long_value = -1;
#else
    long long_value = -1;
#endif
    PyObject *obj, *err;

    /*
     * Be a bit stricter and not allow bools.
     * np.bool_ is also disallowed as Boolean arrays do not currently
     * support index.
     */
    if (!o || PyBool_Check(o) || PyArray_IsScalar(o, Bool)) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }

    /*
     * Since it is the usual case, first check if o is an integer. This is
     * an exact check, since otherwise __index__ is used.
     */
#if !defined(NPY_PY3K)
    if (PyInt_CheckExact(o)) {
  #if (NPY_SIZEOF_LONG <= NPY_SIZEOF_INTP)
        /* No overflow is possible, so we can just return */
        return PyInt_AS_LONG(o);
  #else
        long_value = PyInt_AS_LONG(o);
        goto overflow_check;
  #endif
    }
    else
#endif
    if (PyLong_CheckExact(o)) {
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
        long_value = PyLong_AsLongLong(o);
#else
        long_value = PyLong_AsLong(o);
#endif
        return (npy_intp)long_value;
    }

    /*
     * The most general case. PyNumber_Index(o) covers everything
     * including arrays. In principle it may be possible to replace
     * the whole function by PyIndex_AsSSize_t after deprecation.
     */
    obj = PyNumber_Index(o);
    if (obj == NULL) {
        return -1;
    }
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
    long_value = PyLong_AsLongLong(obj);
#else
    long_value = PyLong_AsLong(obj);
#endif
    Py_DECREF(obj);

    if (error_converting(long_value)) {
        err = PyErr_Occurred();
        /* Only replace TypeError's here, which are the normal errors. */
        if (PyErr_GivenExceptionMatches(err, PyExc_TypeError)) {
            PyErr_SetString(PyExc_TypeError, msg);
        }
        return -1;
    }
    goto overflow_check; /* silence unused warning */

overflow_check:
#if (NPY_SIZEOF_LONG < NPY_SIZEOF_INTP)
  #if (NPY_SIZEOF_LONGLONG > NPY_SIZEOF_INTP)
    if ((long_value < NPY_MIN_INTP) || (long_value > NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
                "Python int too large to convert to C numpy.intp");
        return -1;
    }
  #endif
#else
  #if (NPY_SIZEOF_LONG > NPY_SIZEOF_INTP)
    if ((long_value < NPY_MIN_INTP) || (long_value > NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
                "Python int too large to convert to C numpy.intp");
        return -1;
    }
  #endif
#endif
    return long_value;
}

static int
_PyArray_PyIntAsInt_ErrMsg(PyObject *o, const char * msg)
{
    npy_intp long_value;
    /* This assumes that NPY_SIZEOF_INTP >= NPY_SIZEOF_INT */
    long_value = _PyArray_PyIntAsIntp_ErrMsg(o, msg);

#if (NPY_SIZEOF_INTP > NPY_SIZEOF_INT)
    if ((long_value < INT_MIN) || (long_value > INT_MAX)) {
        PyErr_SetString(PyExc_ValueError, "integer won't fit into a C int");
        return -1;
    }
#endif
    return (int) long_value;
}

/*
 * Borrow from Numpy
 * Converts an axis parameter into an ndim-length C-array of
 * boolean flags, True for each axis specified.
 *
 * If obj is None or NULL, everything is set to True. If obj is a tuple,
 * each axis within the tuple is set to True. If obj is an integer,
 * just that axis is set to True.
 */
NPY_NO_EXPORT int
PyMicArray_ConvertMultiAxis(PyObject *axis_in, int ndim, npy_bool *out_axis_flags)
{
    /* None means all of the axes */
    if (axis_in == Py_None || axis_in == NULL) {
        memset(out_axis_flags, 1, ndim);
        return NPY_SUCCEED;
    }
    /* A tuple of which axes */
    else if (PyTuple_Check(axis_in)) {
        int i, naxes;

        memset(out_axis_flags, 0, ndim);

        naxes = PyTuple_Size(axis_in);
        if (naxes < 0) {
            return NPY_FAIL;
        }
        for (i = 0; i < naxes; ++i) {
            PyObject *tmp = PyTuple_GET_ITEM(axis_in, i);
            int axis = _PyArray_PyIntAsInt_ErrMsg(tmp,
                          "integers are required for the axis tuple elements");
            int axis_orig = axis;
            if (error_converting(axis)) {
                return NPY_FAIL;
            }
            if (axis < 0) {
                axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
                PyErr_Format(PyExc_ValueError,
                        "'axis' entry %d is out of bounds [-%d, %d)",
                        axis_orig, ndim, ndim);
                return NPY_FAIL;
            }
            if (out_axis_flags[axis]) {
                PyErr_SetString(PyExc_ValueError,
                        "duplicate value in 'axis'");
                return NPY_FAIL;
            }
            out_axis_flags[axis] = 1;
        }

        return NPY_SUCCEED;
    }
    /* Try to interpret axis as an integer */
    else {
        int axis, axis_orig;

        memset(out_axis_flags, 0, ndim);

        axis = _PyArray_PyIntAsInt_ErrMsg(axis_in,
                                   "an integer is required for the axis");
        axis_orig = axis;

        if (error_converting(axis)) {
            return NPY_FAIL;
        }
        if (axis < 0) {
            axis += ndim;
        }
        /*
         * Special case letting axis={-1,0} slip through for scalars,
         * for backwards compatibility reasons.
         */
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            return NPY_SUCCEED;
        }

        if (axis < 0 || axis >= ndim) {
            PyErr_Format(PyExc_ValueError,
                    "'axis' entry %d is out of bounds [-%d, %d)",
                    axis_orig, ndim, ndim);
            return NPY_FAIL;
        }

        out_axis_flags[axis] = 1;

        return NPY_SUCCEED;
    }
}
