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
//#include "convert_datatype.h"
//#include "methods.h"
#include "common.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"

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
 * Prepares shape and strides for a simple raw array iteration.
 * This sorts the strides into FORTRAN order, reverses any negative
 * strides, then coalesces axes where possible. The results are
 * filled in the output parameters.
 *
 * This is intended for simple, lightweight iteration over arrays
 * where no buffering of any kind is needed, and the array may
 * not be stored as a PyArrayObject.
 *
 * The arrays shape, out_shape, strides, and out_strides must all
 * point to different data.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareOneRawArrayIter(int ndim, npy_intp *shape,
                            char *data, npy_intp *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_data = data;
        out_shape[0] = 1;
        out_strides[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entry = strides[0], shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride */
        if (stride_entry >= 0) {
            *out_data = data;
            out_strides[0] = stride_entry;
        }
        else {
            *out_data = data + stride_entry * (shape_entry - 1);
            out_strides[0] = -stride_entry;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, strides, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_strides[i] = strides[iperm];
    }

    /* Reverse any negative strides */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entry = out_strides[i], shape_entry = out_shape[i];

        if (stride_entry < 0) {
            data += stride_entry * (shape_entry - 1);
            out_strides[i] = -stride_entry;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_data = data;
            out_shape[0] = 0;
            out_strides[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_strides[i] * out_shape[i] == out_strides[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
    }
    ndim = i+1;

#if 0
    /* DEBUG */
    {
        printf("raw iter ndim %d\n", ndim);
        printf("shape: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_shape[i]);
        }
        printf("\n");
        printf("strides: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_strides[i]);
        }
        printf("\n");
    }
#endif

    *out_data = data;
    *out_ndim = ndim;
    return 0;
}

/*
 * The same as PyArray_PrepareOneRawArrayIter, but for two
 * operands instead of one. Any broadcasting of the two operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for both operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_TWO_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareTwoRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entryA = stridesA[0], stride_entryB = stridesB[0];
        npy_intp shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entryA = out_stridesA[i];
        npy_intp stride_entryB = out_stridesB[i];
        npy_intp shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                    out_stridesB[i] * out_shape[i] == out_stridesB[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_ndim = ndim;
    return 0;
}

/*
 * The same as PyArray_PrepareOneRawArrayIter, but for three
 * operands instead of one. Any broadcasting of the three operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for all operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_THREE_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_PrepareThreeRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            char *dataC, npy_intp *stridesC,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB,
                            char **out_dataC, npy_intp *out_stridesC)
{
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        *out_dataC = dataC;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        out_stridesC[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        npy_intp stride_entryA = stridesA[0];
        npy_intp stride_entryB = stridesB[0];
        npy_intp stride_entryC = stridesC[0];
        npy_intp shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
            out_stridesC[0] = stride_entryC;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            *out_dataC = dataC + stride_entryC * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
            out_stridesC[0] = -stride_entryC;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
        out_stridesC[i] = stridesC[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        npy_intp stride_entryA = out_stridesA[i];
        npy_intp stride_entryB = out_stridesB[i];
        npy_intp stride_entryC = out_stridesC[i];
        npy_intp shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            dataC += stride_entryC * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
            out_stridesC[i] = -stride_entryC;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            out_stridesC[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                    out_stridesB[i] * out_shape[i] == out_stridesB[j] &&
                    out_stridesC[i] * out_shape[i] == out_stridesC[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_dataC = dataC;
    *out_ndim = ndim;
    return 0;
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
raw_array_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data)
{
    //TODO: implement
    return -1;
}

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides)
{
    //TODO: implement
    return -1;
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
                        PyMicArrayObject *wheremask,
                        NPY_CASTING casting)
{
    //TODO: implement this
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
raw_array_assign_array(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp *src_strides)
{
    //TODO: implement
    return -1;
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
    if (PyArray_PrepareTwoRawArrayIter(
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
    //TODO: implement
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
    int copied_src = 0, host_device;

    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyArray_NDIM(src) == 0) {
        return PyMicArray_AssignRawScalar(
                            dst, PyArray_DESCR(src), PyArray_DATA(src),
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
        copied_src = 1;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (!broadcast_array_strides((PyArrayObject *) dst, src, src_strides)) {
        goto fail;
    }


    /* A straightforward value assignment */
    /* Do the assignment with raw array iteration */
    host_device = omp_get_initial_device();
    if (raw_array_assign_device_array(PyMicArray_NDIM(dst), PyMicArray_DIMS(dst),
                PyMicArray_DESCR(dst),
                PyMicArray_DEVICE(dst), PyMicArray_BYTES(dst), PyMicArray_STRIDES(dst),
                host_device, PyMicArray_BYTES(src), src_strides) < 0) {
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
