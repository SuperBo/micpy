#ifndef _MPY_SCALAR_H_
#define _MPY_SCALAR_H_

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

NPY_NO_EXPORT PyObject *
PyMicArray_Return(PyMicArrayObject *mp);

NPY_NO_EXPORT PyObject *
PyMicArray_ToScalar(void *data, PyMicArrayObject* obj);

#endif
