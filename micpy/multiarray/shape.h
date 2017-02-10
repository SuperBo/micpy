#ifndef _MPY_ARRAY_SHAPE_H_
#define _MPY_ARRAY_SHAPE_H_

/*
 * Builds a string representation of the shape given in 'vals'.
 * A negative value in 'vals' gets interpreted as newaxis.
 */
NPY_NO_EXPORT PyObject *
build_shape_string(npy_intp n, npy_intp *vals);


/*
 * Just like PyArray_Squeeze, but allows the caller to select
 * a subset of the size-one dimensions to squeeze out.
 */
NPY_NO_EXPORT PyObject *
PyArray_SqueezeSelected(PyArrayObject *self, npy_bool *axis_flags);

NPY_NO_EXPORT PyObject *
PyMicArray_Newshape(PyMicArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER order);

NPY_NO_EXPORT PyObject *
PyMicArray_Reshape(PyMicArrayObject *self, PyObject *shape);

NPY_NO_EXPORT PyObject *
PyMicArray_Transpose(PyMicArrayObject *ap, PyArray_Dims *permute);

NPY_NO_EXPORT PyObject *
PyMicArray_SwapAxes(PyMicArrayObject *ap, int a1, int a2);

NPY_NO_EXPORT PyObject *
PyMicArray_Flatten(PyMicArrayObject *a, NPY_ORDER order);

NPY_NO_EXPORT void
PyMicArray_RemoveAxesInPlace(PyMicArrayObject *arr, npy_bool *flags);

#endif
