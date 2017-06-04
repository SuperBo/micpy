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

/*
 * This function calls Py_DECREF on flex_dtype, and replaces it with
 * a new dtype that has been adapted based on the values in data_dtype
 * and data_obj. If the flex_dtype is not flexible, it leaves it as is.
 *
 * Usually, if data_obj is not an array, dtype should be the result
 * given by the PyArray_GetArrayParamsFromObject function.
 *
 * The data_obj may be NULL if just a dtype is is known for the source.
 *
 * If *flex_dtype is NULL, returns immediately, without setting an
 * exception. This basically assumes an error was already set previously.
 *
 * The current flexible dtypes include NPY_STRING, NPY_UNICODE, NPY_VOID,
 * and NPY_DATETIME with generic units.
 */
NPY_NO_EXPORT void
PyMicArray_AdaptFlexibleDType(PyObject *data_obj, PyArray_Descr *data_dtype,
                            PyArray_Descr **flex_dtype)
{
    /* TODO: do nothing right now
     * might enable in the future
     */
    return;
}
