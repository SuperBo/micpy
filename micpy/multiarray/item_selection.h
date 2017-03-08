#ifndef _MPY_PRIVATE__ITEM_SELECTION_H_
#define _MPY_PRIVATE__ITEM_SELECTION_H_

/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char *data, npy_intp *ashape, npy_intp *astrides);

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
NPY_NO_EXPORT PyObject *
PyMicArray_MultiIndexGetItem(PyMicArrayObject *self, npy_intp *multi_index);

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_MultiIndexSetItem(PyMicArrayObject *self, npy_intp *multi_index,
                                                PyObject *obj);

NPY_NO_EXPORT PyObject *
PyMicArray_Repeat(PyMicArrayObject *aop, PyObject *op, int axis);

NPY_NO_EXPORT PyObject *
PyMicArray_PutTo(PyMicArrayObject *self, PyObject* values, PyObject *indices,
              NPY_CLIPMODE clipmode);

NPY_NO_EXPORT PyObject *
PyMicArray_Choose(PyMicArrayObject *ip, PyObject *op, PyMicArrayObject *out,
               NPY_CLIPMODE clipmode);

NPY_NO_EXPORT npy_intp
PyMicArray_CountNonzero(PyMicArrayObject *self);

NPY_NO_EXPORT PyObject *
PyMicArray_Nonzero(PyMicArrayObject *self);

NPY_NO_EXPORT PyObject *
PyMicArray_Diagonal(PyMicArrayObject *self, int offset, int axis1, int axis2);

NPY_NO_EXPORT PyObject *
PyMicArray_Compress(PyMicArrayObject *self, PyObject *condition, int axis,
                 PyMicArrayObject *out);

#endif
