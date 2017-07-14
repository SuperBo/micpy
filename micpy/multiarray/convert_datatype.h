#ifndef _MPY_ARRAY_CONVERT_DATATYPE_H_
#define _MPY_ARRAY_CONVERT_DATATYPE_H_

NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data, int scal_device,
                    PyArray_Descr *to, NPY_CASTING casting);

NPY_NO_EXPORT npy_bool
PyMicArray_CanCastArrayTo(PyMicArrayObject *arr, PyArray_Descr *to, NPY_CASTING casting);

NPY_NO_EXPORT PyObject *
PyMicArray_CastToType(PyMicArrayObject *arr, PyArray_Descr *dtype, int is_f_order);

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

NPY_NO_EXPORT int
PyMicArray_ObjectType(PyObject *op, int minimum_type);

#endif
