#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _mpy_umathmodule_ARRAY_API
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

#define PyMicArray_API_UNIQUE_NAME _mpy_umathmodule_MICARRAY_API
#define PyMicArray_NO_IMPORT
#include <multiarray/arrayobject.h>
#include <multiarray/multiarray_api.h>

#include "output_creators.h"

int PyMUFunc_GetCommonDevice(int nop, PyMicArrayObject **op) {
    int i, device;

    device = PyMicArray_DEVICE(op[0]);

    for (i = 1; i < nop; ++i) {
        if (PyMicArray_DEVICE(op[i]) != device)
            return -1;
    }

    return device;
}

PyMicArrayObject *
PyMUFunc_CreateArrayBroadcast(int nop, PyMicArrayObject **arrs, PyArray_Descr *dtype)
{
    PyMicArrayObject *ret;
    npy_intp shape[NPY_MAXDIMS];
    npy_intp *arr_shape;
    int device, ndim, max_ndim, arr_ndim;
    int i, j ,k;
    npy_bool not_match = 0;

    device = PyMUFunc_GetCommonDevice(nop, arrs);

    /* Find largest ndim array */
    max_ndim = 0;
    for (i = 1; i < nop; ++i) {
        if (PyMicArray_NDIM(arrs[i]) > PyMicArray_NDIM(arrs[max_ndim])) {
            max_ndim = i;
        }
    }

    /* Copy largest ndim array to shape */
    ndim = PyMicArray_NDIM(arrs[max_ndim]);
    for (i = 0; i < ndim; ++i) {
        shape[i] = PyMicArray_DIMS(arrs[max_ndim])[i];
    }

    /* Find broadcast shape */
    for (i = 0; i < nop; ++i) {
        if (i == max_ndim)
            continue;

        arr_ndim = PyMicArray_NDIM(arrs[i]);
        arr_shape = PyMicArray_DIMS(arrs[i]);
        for (j = ndim - arr_ndim, k = 0; j < ndim; ++j, ++k) {
            if (shape[j] == 1) {
                shape[j] = arr_shape[k];
            } else if (arr_shape[k] == 1 || shape[j] == arr_shape[k]) {
                continue;
            }
            else {
                /* Shape mismatch */
                PyErr_SetString(PyExc_ValueError, "Shape mismatch");
                return NULL;
            }
        }
    }

    ret = (PyMicArrayObject *) PyMicArray_Empty(device, ndim,
                                                shape, dtype,
                                                0);
    return ret;
}
