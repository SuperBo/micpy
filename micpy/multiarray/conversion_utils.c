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
