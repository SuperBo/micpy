#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "npy_config.h"

#define _MICARRAYMODULE
#include "arrayobject.h"
#include "convert_datatype.h"
#include "creators.h"
//#include "methods.h"
#include "mpy_lowlevel_strided_loops.h"
#include "mpymem_overlap.h"
#include "dtype_transfer.h"
#include "common.h"
#include "shape.h"

#include "array_assign.h"

/* Helpers part */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, char *data, npy_intp *strides, int alignment)
{
    if (alignment > 1) {
        npy_intp align_check = (npy_intp)data;
        int idim;

        for (idim = 0; idim < ndim; ++idim) {
            align_check |= strides[idim];
        }

        return mpy_is_aligned((void *)align_check, alignment);
    }
    else {
        return 1;
    }
}

/* Returns 1 if the arrays have overlapping data, 0 otherwise */
NPY_NO_EXPORT int
arrays_overlap(PyMicArrayObject *arr1, PyMicArrayObject *arr2)
{
    mem_overlap_t result;

    result = solve_may_share_memory(arr1, arr2, NPY_MAY_SHARE_BOUNDS);
    if (result == MEM_OVERLAP_NO) {
        return 0;
    }
    else {
        return 1;
    }
}

static int
PyMicArray_CastRawArrays(int device, npy_intp count,
                      void *src, void *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references)
{
    PyMicArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned = 1, needs_api = 0;

    /* Make sure the copy is reasonable */
    if (dst_stride == 0 && count > 1) {
        PyErr_SetString(PyExc_ValueError,
                    "NumPy CastRawArrays cannot do a reduction");
        return NPY_FAIL;
    }
    else if (count == 0) {
        return NPY_SUCCEED;
    }

    /* Check data alignment */
    aligned = (((npy_intp)src | src_stride) &
                                (src_dtype->alignment - 1)) == 0 &&
              (((npy_intp)dst | dst_stride) &
                                (dst_dtype->alignment - 1)) == 0;

    /* Get the function to do the casting */
    if (PyMicArray_GetDTypeTransferFunction(device,
                        aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /* Cast */
    stransfer(dst, dst_stride, src, src_stride, count,
                src_dtype->elsize, transferdata, device);

    /* Cleanup */
    NPY_AUXDATA_FREE(transferdata);

    /* If needs_api was set to 1, it may have raised a Python exception */
    return (needs_api && PyErr_Occurred()) ? NPY_FAIL : NPY_SUCCEED;
}

/*
 * ====================
 * | Low level part   |
 * ====================
 */

/*
 * Broadcasts strides to match the given dimensions. Can be used,
 * for instance, to set up a raw iteration.
 *
 * 'strides_name' is used to produce an error message if the strides
 * cannot be broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
broadcast_strides(int ndim, npy_intp *shape,
                int strides_ndim, npy_intp *strides_shape, npy_intp *strides,
                char *strides_name,
                npy_intp *out_strides)
{
    int idim, idim_start = ndim - strides_ndim;

    /* Can't broadcast to fewer dimensions */
    if (idim_start < 0) {
        goto broadcast_error;
    }

    /*
     * Process from the end to the start, so that 'strides' and 'out_strides'
     * can point to the same memory.
     */
    for (idim = ndim - 1; idim >= idim_start; --idim) {
        npy_intp strides_shape_value = strides_shape[idim - idim_start];
        /* If it doesn't have dimension one, it must match */
        if (strides_shape_value == 1) {
            out_strides[idim] = 0;
        }
        else if (strides_shape_value != shape[idim]) {
            goto broadcast_error;
        }
        else {
            out_strides[idim] = strides[idim - idim_start];
        }
    }

    /* New dimensions get a zero stride */
    for (idim = 0; idim < idim_start; ++idim) {
        out_strides[idim] = 0;
    }

    return 0;

broadcast_error: {
        PyObject *errmsg;

        errmsg = PyUString_FromFormat("could not broadcast %s from shape ",
                                strides_name);
        PyUString_ConcatAndDel(&errmsg,
                build_shape_string(strides_ndim, strides_shape));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" into shape "));
        PyUString_ConcatAndDel(&errmsg,
                build_shape_string(ndim, shape));
        PyErr_SetObject(PyExc_ValueError, errmsg);
        Py_DECREF(errmsg);

        return -1;
   }
}


/*
 * ================
 * | Scalar Part  |
 * ================
 */

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar(int device, int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    PyMicArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    NPY_BEGIN_THREADS_DEF;


    /* Check alignment */
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
                                    dst_dtype->alignment);
    if (!mpy_is_aligned(src_data, src_dtype->alignment)) {
        aligned = 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyMicArray_PrepareOneRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    if (PyMicArray_GetDTypeTransferFunction(device, aligned,
                        0, dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        stransfer(dst_data, dst_strides_it[0], src_data, 0,
                    shape_it[0], src_itemsize, transferdata, device);
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, dst_data, dst_strides_it);

    NPY_END_THREADS;

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int device, int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    PyMicArray_MaskedStridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    NPY_BEGIN_THREADS_DEF;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
                                    dst_dtype->alignment);
    if (!mpy_is_aligned(src_data, src_dtype->alignment)) {
        aligned = 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyMicArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /* Get the function to do the casting */
    if (PyMicArray_GetMaskedDTypeTransferFunction(aligned,
                        0, dst_strides_it[0], wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        stransfer(dst_data, dst_strides_it[0], src_data, 0,
                    (npy_bool *)wheremask_data, wheremask_strides_it[0],
                    shape_it[0], src_itemsize, transferdata, device);
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            wheremask_data, wheremask_strides_it);

    NPY_END_THREADS;

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns a scalar value specified by 'src_dtype' and 'src_data'
 * to elements of 'dst'.
 *
 * dst: The destination array (on device).
 * src_dtype: The data type of the source scalar.
 * src_data: The memory element of the source scalar (on host device).
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the assignment violates this
 *          casting rule.
 *
 * This function is implemented in array_assign_scalar.c.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_AssignRawScalar(PyMicArrayObject *dst,
                        PyArray_Descr *src_dtype, char *src_data,
                        int src_device,
                        PyMicArrayObject *wheremask,
                        NPY_CASTING casting)
{
    int allocated_src_data = 0;
    int device = PyMicArray_DEVICE(dst);
    int cpu_device = omp_get_initial_device();
    npy_longlong scalarbuffer[4];

    if (PyMicArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        return -1;
    }

    /* Check the casting rule */
    if (!can_cast_scalar_to(src_dtype, src_data, src_device,
                            PyMicArray_DESCR(dst), casting)) {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast scalar from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)src_dtype));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyMicArray_DESCR(dst)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        return -1;
    }

    if (src_data == NULL) {
        return -1;
    }

    /*
     * Check whether dst_device is different from src_device
     * If different, need to transfer data
     */
    if (src_device != device) {
        void *tmp_src_data = target_alloc(src_dtype->elsize, device);
        if (tmp_src_data == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        target_memcpy(tmp_src_data, src_data, src_dtype->elsize,
                      device, src_device);
        src_data = tmp_src_data;
        src_device = device;
        allocated_src_data = 1;
    }

    /*
     * Make a copy of the src data if it's a different dtype than 'dst'
     * or isn't aligned, and the destination we're copying to has
     * more than one element. To avoid having to manage object lifetimes,
     * we also skip this if 'dst' has an object dtype.
     */
    if ((!PyArray_EquivTypes(PyMicArray_DESCR(dst), src_dtype) ||
                !mpy_is_aligned(src_data, src_dtype->alignment)) &&
                PyMicArray_SIZE(dst) > 1 &&
                !PyDataType_REFCHK(PyMicArray_DESCR(dst))) {
        void *tmp_src_data;

        tmp_src_data = target_alloc(PyMicArray_DESCR(dst)->elsize, device);
        if (tmp_src_data == NULL) {
            PyErr_NoMemory();
            goto fail;
        }

        if (PyMicArray_CastRawArrays(device, 1, src_data, tmp_src_data, 0, 0,
                            src_dtype, PyMicArray_DESCR(dst), 0) != NPY_SUCCEED) {
            target_free(tmp_src_data, device);
            goto fail;
        }

        if (allocated_src_data) {
            target_free(src_data, device);
        } else {
            allocated_src_data = 1;
        }
        /* Replace src_data/src_dtype */
        src_data = tmp_src_data;
        src_dtype = PyMicArray_DESCR(dst);
    }


    if (wheremask == NULL) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (raw_array_assign_scalar(device,
                PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                PyMicArray_DESCR(dst), PyMicArray_DATA(dst), PyMicArray_STRIDES(dst),
                src_dtype, src_data) < 0) {
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                    PyMicArray_NDIM(wheremask), PyMicArray_DIMS(wheremask),
                    PyMicArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* Do the masked assignment with raw array iteration */
        if (raw_array_wheremasked_assign_scalar(device,
                PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                PyMicArray_DESCR(dst), PyMicArray_DATA(dst), PyMicArray_STRIDES(dst),
                src_dtype, src_data,
                PyMicArray_DESCR(wheremask), PyMicArray_DATA(wheremask),
                wheremask_strides) < 0) {
            goto fail;
        }
    }

    if (allocated_src_data) {
        target_free(src_data, device);
    }

    return 0;

fail:
    if (allocated_src_data) {
        target_free(src_data, device);
    }

    return -1;
}

/*
 * ================
 * | Arrays Part  |
 * ================
 */

/*
 * Assigns the array from 'src' to 'dst'. The strides must already have
 * been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_array(int device, int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp *src_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    PyMicArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    NPY_BEGIN_THREADS_DEF;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim,
                        dst_data, dst_strides, dst_dtype->alignment) &&
              raw_array_is_aligned(ndim,
                        src_data, src_strides, src_dtype->alignment);

    /* Use raw iteration with no heap allocation */
    if (PyMicArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays and
     * opposite strides cause a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
    }

    /* Get the function to do the casting */
    if (PyMicArray_GetDTypeTransferFunction(device, aligned,
                        src_strides_it[0], dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        stransfer(dst_data, dst_strides_it[0], src_data, src_strides_it[0],
                    shape_it[0], src_itemsize, transferdata, device);
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it);

    NPY_END_THREADS;

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns the array from 'src' to 'dst, wherever the 'wheremask'
 * value is True. The strides must already have been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_array(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp *src_strides,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides)
{
    //TODO: implement
    return -1;
}

/*
 * Assigns the array from 'src'(host) to 'dst'(device). The strides must already have
 * been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_device_array(int ndim, npy_intp *shape,
        PyArray_Descr *dtype,
        int dst_device, char *dst_data, npy_intp *dst_strides,
        int src_device, char *src_data, npy_intp *src_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    npy_intp itemsize = dtype->elsize;

    NPY_BEGIN_THREADS_DEF;

    /* Use raw iteration with no heap allocation */
    if (PyMicArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it) < 0) {
        return -1;
    }

    /* Get Host device number */
    int host_device = omp_get_initial_device();

    NPY_BEGIN_THREADS;

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        if (omp_target_memcpy(dst_data, src_data,
                            itemsize * shape_it[0],
                            0, 0,
                            dst_device, src_device) < 0) {
            return -1;
        }
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it);

    NPY_END_THREADS;

    return (PyErr_Occurred()) ? -1 : 0;
}

/*
 * Internal function for checking cast rule
 * Return 1 when success and 0 when fail
 */
static int check_casting(PyArray_Descr *dst, PyArray_Descr *src, NPY_CASTING casting) {
    if (!PyArray_CanCastTypeTo(src, dst, casting)) {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast scalar from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)src));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)dst));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        return 0;
    }

    return 1;
}

/*
 * Internal function for broadcasting strides
 * Return 1 when sucess or 0 when fail
 */
static int broadcast_array_strides(PyArrayObject *dst, PyArrayObject *src, npy_intp *src_strides) {
    if (PyArray_NDIM(src) > PyArray_NDIM(dst)) {
        int ndim_tmp = PyArray_NDIM(src);
        npy_intp *src_shape_tmp = PyArray_DIMS(src);
        npy_intp *src_strides_tmp = PyArray_STRIDES(src);
        /*
         * As a special case for backwards compatibility, strip
         * away unit dimensions from the left of 'src'
         */
        while (ndim_tmp > PyArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    ndim_tmp, src_shape_tmp,
                    src_strides_tmp, "input array",
                    src_strides) < 0) {
            return 0;
        }
    }
    else {
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(src), PyArray_DIMS(src),
                    PyArray_STRIDES(src), "input array",
                    src_strides) < 0) {
            return 0;
        }
    }

    return 1;
}


/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array (On device memory).
 * src: The source array (On device memory).
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_AssignArray(PyMicArrayObject *dst, PyMicArrayObject *src,
                    PyMicArrayObject *wheremask,
                    NPY_CASTING casting)
{
    int copied_src = 0;

    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyMicArray_NDIM(src) == 0) {
        return PyMicArray_AssignRawScalar(
                            dst, PyMicArray_DESCR(src), PyMicArray_DATA(src),
                            PyMicArray_DEVICE(src) ,wheremask, casting);
    }

    /* Use assign from device if device if different */
    if (PyMicArray_DEVICE(dst) != PyMicArray_DEVICE(src)) {
        /* TODO: consider where mask case */
        return PyMicArray_AssignArrayFromDevice(dst, src, casting);
    }

    /*
     * Performance fix for expressions like "a[1000:6000] += x".  In this
     * case, first an in-place add is done, followed by an assignment,
     * equivalently expressed like this:
     *
     *   tmp = a[1000:6000]   # Calls array_subscript in mapping.c
     *   np.add(tmp, x, tmp)
     *   a[1000:6000] = tmp   # Calls array_assign_subscript in mapping.c
     *
     * In the assignment the underlying data type, shape, strides, and
     * data pointers are identical, but src != dst because they are separately
     * generated slices.  By detecting this and skipping the redundant
     * copy of values to themselves, we potentially give a big speed boost.
     *
     * Note that we don't call EquivTypes, because usually the exact same
     * dtype object will appear, and we don't want to slow things down
     * with a complicated comparison.  The comparisons are ordered to
     * try and reject this with as little work as possible.
     */
    if (PyMicArray_DATA(src) == PyMicArray_DATA(dst) &&
                        PyMicArray_DESCR(src) == PyMicArray_DESCR(dst) &&
                        PyMicArray_NDIM(src) == PyMicArray_NDIM(dst) &&
                        PyArray_CompareLists(PyMicArray_DIMS(src),
                                             PyMicArray_DIMS(dst),
                                             PyMicArray_NDIM(src)) &&
                        PyArray_CompareLists(PyMicArray_STRIDES(src),
                                             PyMicArray_STRIDES(dst),
                                             PyMicArray_NDIM(src))) {
        /*printf("Redundant copy operation detected\n");*/
        return 0;
    }

    if (PyMicArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        goto fail;
    }

    /* Check the casting rule */
    if (!PyArray_CanCastTypeTo(PyMicArray_DESCR(src),
                                PyMicArray_DESCR(dst), casting)) {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast scalar from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyMicArray_DESCR(src)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyMicArray_DESCR(dst)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        goto fail;
    }

    /*
     * When ndim is 1 and the strides point in the same direction,
     * the lower-level inner loop handles copying
     * of overlapping data. For bigger ndim and opposite-strided 1D
     * data, we make a temporary copy of 'src' if 'src' and 'dst' overlap.'
     */
    if (((PyMicArray_NDIM(dst) == 1 && PyMicArray_NDIM(src) >= 1 &&
                    PyMicArray_STRIDES(dst)[0] *
                            PyMicArray_STRIDES(src)[PyMicArray_NDIM(src) - 1] < 0) ||
                    PyMicArray_NDIM(dst) > 1) && arrays_overlap(src, dst)) {
        PyMicArrayObject *tmp;

        /*
         * Allocate a temporary copy array.
         */
        tmp = (PyMicArrayObject *)PyMicArray_NewLikeArray(
                                        PyMicArray_DEVICE(dst),
                                        (PyArrayObject *) dst,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            goto fail;
        }

        if (PyMicArray_AssignArray(tmp, src, NULL, NPY_UNSAFE_CASTING) < 0) {
            Py_DECREF(tmp);
            goto fail;
        }

        src = tmp;
        copied_src = 1;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (PyMicArray_NDIM(src) > PyMicArray_NDIM(dst)) {
        int ndim_tmp = PyMicArray_NDIM(src);
        npy_intp *src_shape_tmp = PyMicArray_DIMS(src);
        npy_intp *src_strides_tmp = PyMicArray_STRIDES(src);
        /*
         * As a special case for backwards compatibility, strip
         * away unit dimensions from the left of 'src'
         */
        while (ndim_tmp > PyMicArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                    ndim_tmp, src_shape_tmp,
                    src_strides_tmp, "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }
    else {
        if (broadcast_strides(PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                    PyMicArray_NDIM(src), PyMicArray_DIMS(src),
                    PyMicArray_STRIDES(src), "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }

    if (wheremask == NULL) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (raw_array_assign_array(PyMicArray_DEVICE(dst),
                PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                PyMicArray_DESCR(dst), PyMicArray_DATA(dst), PyMicArray_STRIDES(dst),
                PyMicArray_DESCR(src), PyMicArray_DATA(src), src_strides) < 0) {
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                    PyMicArray_NDIM(wheremask), PyMicArray_DIMS(wheremask),
                    PyMicArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* A straightforward where-masked assignment */
         /* Do the masked assignment with raw array iteration */
         if (raw_array_wheremasked_assign_array(
                 PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                 PyMicArray_DESCR(dst), PyMicArray_DATA(dst), PyMicArray_STRIDES(dst),
                 PyMicArray_DESCR(src), PyMicArray_DATA(src), src_strides,
                 PyMicArray_DESCR(wheremask), PyMicArray_DATA(wheremask),
                         wheremask_strides) < 0) {
             goto fail;
         }
    }

    if (copied_src) {
        Py_DECREF(src);
    }
    return 0;

fail:
    if (copied_src) {
        Py_DECREF(src);
    }
    return -1;
}

static int
_AssignArrayFromAnotherDevice(PyMicArrayObject *dst, PyArrayObject *src,
                                    int device, NPY_CASTING casting)
{
    int copied_src = 0;
    int host_device = CPU_DEVICE;

    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyArray_NDIM(src) == 0) {
        return PyMicArray_AssignRawScalar(
                            dst, PyArray_DESCR(src), PyArray_DATA(src),
                            device,
                            NULL, casting);
    }

    if (PyMicArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        goto fail;
    }

    /* Check the casting rule */
    if (!check_casting(PyMicArray_DESCR(dst), PyArray_DESCR(src), casting)){
        goto fail;
    }

    /*
     * When source dtype is not equal dest dtype,
     * we make a temporary copy of 'src'
     */
    if (PyArray_TYPE(src) != PyMicArray_TYPE(dst)) {
        if (device == host_device) {
            PyArrayObject *tmp;

            /*
            * Allocate a temporary copy array.
            */
            tmp = (PyArrayObject *)PyArray_NewLikeArray((PyArrayObject *) dst,
                                            NPY_KEEPORDER, NULL, 0);
            if (tmp == NULL) {
                goto fail;
            }

            if (PyArray_CopyInto(tmp, src) < 0) {
                Py_DECREF(tmp);
                goto fail;
            }
            src = tmp;
        }
        else {
            PyMicArrayObject *tmp;

            tmp = (PyMicArrayObject *)PyMicArray_NewLikeArray(
                                            device,
                                            (PyArrayObject *) dst,
                                            NPY_KEEPORDER, NULL, 0);
            if (PyMicArray_CopyInto(tmp, (PyMicArrayObject *)src) < 0) {
                Py_DECREF(tmp);
                goto fail;
            }
            src = (PyArrayObject *)tmp;
        }
        copied_src = 1;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (!broadcast_array_strides((PyArrayObject *) dst, src, src_strides)) {
        goto fail;
    }


    /* A straightforward value assignment */
    /* Do the assignment with raw array iteration */
    if (raw_array_assign_device_array(PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                PyMicArray_DESCR(dst),
                PyMicArray_DEVICE(dst), PyMicArray_BYTES(dst), PyMicArray_STRIDES(dst),
                device, PyArray_BYTES(src), src_strides) < 0) {
        goto fail;
    }

    if (copied_src) {
        Py_DECREF(src);
    }
    return 0;

fail:
    if (copied_src) {
        Py_DECREF(src);
    }
    return -1;
}

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array (On device memory).
 * src: The source array (On host memory).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_AssignArrayFromHost(PyMicArrayObject *dst, PyArrayObject *src,
                    NPY_CASTING casting)
{
    return _AssignArrayFromAnotherDevice(dst, src, CPU_DEVICE, casting);
}

NPY_NO_EXPORT int
PyMicArray_AssignArrayFromDevice(PyMicArrayObject *dst, PyMicArrayObject *src,
                    NPY_CASTING casting) {
    return _AssignArrayFromAnotherDevice(dst, (PyArrayObject *)src,
                    PyMicArray_DEVICE(src), casting);
}

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array (On host memory).
 * src: The source array (On device memory).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignArrayFromDevice(PyArrayObject *dst, PyMicArrayObject *src,
                    NPY_CASTING casting)
{
    int host_device;
    PyArrayObject *org_dst = NULL;

    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyMicArray_NDIM(src) == 0) {
        int ret;
        npy_intp itemsize = PyMicArray_DTYPE(src)->elsize;
        PyObject *tmp_scalar;

        /* Create tmp scalar */
        char tmp[itemsize];

        /* Copy scalar from device to host */
        host_device = omp_get_initial_device();
        if (omp_target_memcpy(tmp, PyMicArray_DATA(src),
                            itemsize,
                            0, 0, host_device, PyMicArray_DEVICE(src)) < 0){
            goto fail;
        }


        /* Create Numpy Scalar from tmp */
        tmp_scalar = PyArray_Scalar(&tmp, PyMicArray_DTYPE(src), NULL);
        if (tmp_scalar == NULL) {
            goto fail;
        }

        /* Fill 'dst' array with tmp scalar */
        ret = PyArray_FillWithScalar(dst, tmp_scalar);

        Py_DECREF(tmp_scalar);
        return ret;
    }

    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        goto fail;
    }

    /* Check the casting rule */
    if (!check_casting(PyArray_DESCR(dst), PyMicArray_DESCR(src), casting)){
        goto fail;
    }

    /*
     * When source dtype is not equal dest dtype,
     * we copy to an temporary array
     */
    if (PyMicArray_TYPE(src) != PyArray_TYPE(dst)) {
        PyArrayObject *tmp;

        /*
         * Allocate a temporary copy array.
         */
        tmp = (PyArrayObject *)PyArray_NewLikeArray((PyArrayObject *) src,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            goto fail;
        }

        org_dst = dst;
        dst = tmp;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (!broadcast_array_strides(dst,(PyArrayObject *) src, src_strides)) {
        goto fail;
    }


    /* A straightforward value assignment */
    /* Do the assignment with raw array iteration */
    host_device = omp_get_initial_device();
    if (raw_array_assign_device_array(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst),
                host_device, PyArray_BYTES(dst), PyArray_STRIDES(dst),
                PyMicArray_DEVICE(src), PyMicArray_BYTES(src), src_strides) < 0) {
        goto fail;
    }


    /*
     * Copy to back to original 'dst'
     */
    if (org_dst) {
        if (PyArray_CopyInto(org_dst, dst) < 0) {
            goto fail;
        }
        Py_DECREF(dst);
    }

    return 0;

fail:
    if (org_dst) {
        Py_DECREF(dst);
    }
    return -1;
}
