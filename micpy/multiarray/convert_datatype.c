#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>

#include "arrayobject.h"

/*
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
PyMicArray_CanCastArrayTo(PyMicArrayObject *arr, PyArray_Descr *to,
                        NPY_CASTING casting)
{
    PyArray_Descr *from = PyMicArray_DESCR(arr);

    return PyArray_CanCastTypeTo(from, to, casting);
}
