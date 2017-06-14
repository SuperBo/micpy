#ifndef _MPY_MULTIARRAY_H_
#define _MPY_MULTIARRAY_H_

NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_array;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_buffer;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_ufunc;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_order;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_copy;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_dtype;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_ma_str_ndmin;

NPY_NO_EXPORT PyObject *
PyMicArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyMicArrayObject* out);

#endif
