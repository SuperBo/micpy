#ifndef _MPY_ARRAYOBJECT_CONVERT_H_
#define _MPY_ARRAYOBJECT_CONVERT_H_

NPY_NO_EXPORT int
PyMicArray_AssignZero(PyMicArrayObject *dst,
                   PyMicArrayObject *wheremask);

NPY_NO_EXPORT PyObject *
PyMicArray_NewCopy(PyMicArrayObject *obj, NPY_ORDER order);

NPY_NO_EXPORT PyObject *
PyMicArray_View(PyMicArrayObject *self,
                    PyArray_Descr *type, PyTypeObject *pytype);

#endif
