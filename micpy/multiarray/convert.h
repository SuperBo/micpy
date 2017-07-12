#ifndef _MPY_ARRAYOBJECT_CONVERT_H_
#define _MPY_ARRAYOBJECT_CONVERT_H_

NPY_NO_EXPORT int
PyMicArray_AssignZero(PyMicArrayObject *dst,
                      PyMicArrayObject *wheremask);

NPY_NO_EXPORT int
PyMicArray_AssignOne(PyMicArrayObject *dst,
                     PyMicArrayObject *wheremask);

NPY_NO_EXPORT PyObject *
PyMicArray_NewCopy(PyMicArrayObject *obj, NPY_ORDER order);

#define PyMicArray_Copy(obj) PyMicArray_NewCopy(obj, NPY_CORDER)

NPY_NO_EXPORT PyObject *
PyMicArray_CheckFromAny(int device,
                    PyObject *op, PyArray_Descr *descr, int min_depth,
                    int max_depth, int requires, PyObject *context);

NPY_NO_EXPORT PyObject *
PyMicArray_View(PyMicArrayObject *self,
                    PyArray_Descr *type, PyTypeObject *pytype);

NPY_NO_EXPORT int
PyMicArray_FillWithScalar(PyMicArrayObject *arr, PyObject *obj);

#endif
