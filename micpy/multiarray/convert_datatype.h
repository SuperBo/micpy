#ifndef _MPY_ARRAY_CONVERT_DATATYPE_H_
#define _MPY_ARRAY_CONVERT_DATATYPE_H_

NPY_NO_EXPORT npy_bool
PyMicArray_CanCastArrayTo(PyMicArrayObject *arr, PyArray_Descr *to, NPY_CASTING casting);

/*
 * This function calls Py_DECREF on flex_dtype, and replaces it with
 * a new dtype that has been adapted based on the values in data_dtype
 * and data_obj. If the flex_dtype is not flexible, it leaves it as is.
 *
 * The current flexible dtypes include NPY_STRING, NPY_UNICODE, NPY_VOID,
 * and NPY_DATETIME with generic units.
 */
NPY_NO_EXPORT void
PyMicArray_AdaptFlexibleDType(PyObject *data_obj, PyArray_Descr *data_dtype,
                            PyArray_Descr **flex_dtype);

#endif
