#ifndef _MPY_CALCULATION_H_
#define _MPY_CALCULATION_H_

NPY_NO_EXPORT PyObject*
PyMicArray_ArgMax(PyMicArrayObject* self, int axis, PyMicArrayObject *out);

NPY_NO_EXPORT PyObject*
PyMicArray_ArgMin(PyMicArrayObject* self, int axis, PyMicArrayObject *out);

NPY_NO_EXPORT PyObject*
PyMicArray_Max(PyMicArrayObject* self, int axis, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Min(PyMicArrayObject* self, int axis, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Ptp(PyMicArrayObject* self, int axis, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Mean(PyMicArrayObject* self, int axis, int rtype, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject *
PyMicArray_Round(PyMicArrayObject *a, int decimals, PyMicArrayObject *out);

NPY_NO_EXPORT PyObject*
PyMicArray_Trace(PyMicArrayObject* self, int offset, int axis1, int axis2,
                int rtype, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Clip(PyMicArrayObject* self, PyObject* min, PyObject* max, PyMicArrayObject *out);

NPY_NO_EXPORT PyObject*
PyMicArray_Conjugate(PyMicArrayObject* self, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Round(PyMicArrayObject* self, int decimals, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Std(PyMicArrayObject* self, int axis, int rtype, PyMicArrayObject* out,
                int variance);

NPY_NO_EXPORT PyObject *
__New_PyMicArray_Std(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out,
                  int variance, int num);

NPY_NO_EXPORT PyObject*
PyMicArray_Sum(PyMicArrayObject* self, int axis, int rtype, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_CumSum(PyMicArrayObject* self, int axis, int rtype, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Prod(PyMicArrayObject* self, int axis, int rtype, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_CumProd(PyMicArrayObject* self, int axis, int rtype, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_All(PyMicArrayObject* self, int axis, PyMicArrayObject* out);

NPY_NO_EXPORT PyObject*
PyMicArray_Any(PyMicArrayObject* self, int axis, PyMicArrayObject* out);


#endif
