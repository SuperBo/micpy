#ifndef _MPY_SCALAR_H_
#define _MPY_SCALAR_H_

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);


NPY_NO_EXPORT PyObject *
PyMicArray_Return(PyMicArrayObject *mp);


NPY_NO_EXPORT PyObject *
PyMicArray_Scalar(void *data, PyArray_Descr *descr, PyObject *base);


#define PyMicArray_ToScalar(data, arr) \
        PyMicArray_Scalar(data, PyMicArray_DESCR(arr), (PyObject *)arr)


#endif
