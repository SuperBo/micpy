/*
 * This file contains low-level loops for data type transfers.
 * In particular the function PyArray_GetDTypeTransferFunction is
 * implemented here.
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.

 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <numpy/npy_cpu.h>

#include "npy_pycompat.h"

#define _MICARRAYMODULE
#include "arrayobject.h"
#include "convert_datatype.h"
#include "creators.h"
//#include "_datetime.h"
//#include "datetime_strings.h"

#include "shape.h"
#include "lowlevel_strided_loops.h"

#define NPY_LOWLEVEL_BUFFER_BLOCKSIZE  128

/********** PRINTF DEBUG TRACING **************/
#define NPY_DT_DBG_TRACING 0
/* Tracing incref/decref can be very noisy */
#define NPY_DT_REF_DBG_TRACING 0

#if NPY_DT_REF_DBG_TRACING
#define NPY_DT_DBG_REFTRACE(msg, ref) \
    printf("%-12s %20p %s%d%s\n", msg, ref, \
                        ref ? "(refcnt " : "", \
                        ref ? (int)ref->ob_refcnt : 0, \
                        ref ? ((ref->ob_refcnt <= 0) ? \
                                        ") <- BIG PROBLEM!!!!" : ")") : ""); \
    fflush(stdout);
#else
#define NPY_DT_DBG_REFTRACE(msg, ref)
#endif
/**********************************************/

/*
 * Returns a transfer function which DECREFs any references in src_type.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
static int
get_decsrcref_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);

/*
 * Returns a transfer function which zeros out the dest values.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
static int
get_setdstzero_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);

/*
 * Returns a transfer function which sets a boolean type to ones.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
get_bool_setdstone_transfer_function(npy_intp dst_stride,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *NPY_UNUSED(out_needs_api));



/********************* DTYPE COPY SWAP FUNCTION ***********************/

NPY_NO_EXPORT int
PyMicArray_GetDTypeCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *dtype,
                            PyArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata)
{
    //TODO
    return NPY_FAIL;
}

/********************* MAIN DTYPE TRANSFER FUNCTION ***********************/

NPY_NO_EXPORT int
PyMicArray_GetDTypeTransferFunction(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    //TODO
    return NPY_FAIL;
}

NPY_NO_EXPORT int
PyMicArray_GetMaskedDTypeTransferFunction(int aligned,
                            npy_intp src_stride,
                            npy_intp dst_stride,
                            npy_intp mask_stride,
                            PyArray_Descr *src_dtype,
                            PyArray_Descr *dst_dtype,
                            PyArray_Descr *mask_dtype,
                            int move_references,
                            PyArray_MaskedStridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    //TODO
    return NPY_FAIL;
}

NPY_NO_EXPORT int
PyMicArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references)
{
    //TODO
   return NPY_FAIL;
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
PyMicArray_PrepareOneRawArrayIter(int ndim, npy_intp *shape,
                            char *data, npy_intp *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides)
{
    //TODO
    return -1;
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
PyMicArray_PrepareTwoRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB)
{
    //TODO
   return -1;
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
PyMicArray_PrepareThreeRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            char *dataC, npy_intp *stridesC,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB,
                            char **out_dataC, npy_intp *out_stridesC)
{
    //TODO
   return -1;
}
