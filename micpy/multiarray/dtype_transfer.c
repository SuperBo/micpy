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
#include "mpy_lowlevel_strided_loops.h"
#include "common.h"
#include "dtype_transfer.h"
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
/*static int
get_decsrcref_transfer_function(int aligned,
                            npy_intp src_stride,
                            PyArray_Descr *src_dtype,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);*/

/*
 * Returns a transfer function which zeros out the dest values.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
/*static int
get_setdstzero_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);
*/

/*
 * Returns a transfer function which sets a boolean type to ones.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
/*NPY_NO_EXPORT int
get_bool_setdstone_transfer_function(npy_intp dst_stride,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *NPY_UNUSED(out_needs_api));*/

/*************************** DEST SETZERO *******************************/

/* Sets dest to zero */
typedef struct {
    NpyAuxData base;
    npy_intp dst_itemsize;
} _dst_memset_zero_data;

/* zero-padded data copy function */
static NpyAuxData *_dst_memset_zero_data_clone(NpyAuxData *data)
{
    _dst_memset_zero_data *newdata =
            (_dst_memset_zero_data *)PyArray_malloc(
                                    sizeof(_dst_memset_zero_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_dst_memset_zero_data));

    return (NpyAuxData *)newdata;
}

static void
_null_to_strided_memset_zero(void *_dst,
                        npy_intp dst_stride,
                        void *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data, int device)
{
    _dst_memset_zero_data *d = (_dst_memset_zero_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;

    #pragma omp target device(device) map(to: N, _dst, dst_stride, dst_itemsize)
    {
        char *dst = (char *) _dst;
        while (N--) {
            memset(dst, 0, dst_itemsize);
            dst += dst_stride;
        }
    }
}

static void
_null_to_contig_memset_zero(void *dst,
                        npy_intp dst_stride,
                        void *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp N, npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *data, int device)
{
    _dst_memset_zero_data *d = (_dst_memset_zero_data *)data;
    npy_intp dst_itemsize = d->dst_itemsize;

    target_memset(dst, 0, N*dst_itemsize, device);
}

NPY_NO_EXPORT int
get_setdstzero_transfer_function(int aligned,
                            npy_intp dst_stride,
                            PyArray_Descr *dst_dtype,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    _dst_memset_zero_data *data;

    /* If there are no references, just set the whole thing to zero */
    if (!PyDataType_REFCHK(dst_dtype)) {
        data = (_dst_memset_zero_data *)
                        PyArray_malloc(sizeof(_dst_memset_zero_data));
        if (data == NULL) {
            PyErr_NoMemory();
            return NPY_FAIL;
        }

        data->base.free = (NpyAuxData_FreeFunc *)(&PyArray_free);
        data->base.clone = &_dst_memset_zero_data_clone;
        data->dst_itemsize = dst_dtype->elsize;

        if (dst_stride == data->dst_itemsize) {
            *out_stransfer = &_null_to_contig_memset_zero;
        }
        else {
            *out_stransfer = &_null_to_strided_memset_zero;
        }
        *out_transferdata = (NpyAuxData *)data;
    }
    /* If it's exactly one reference, use the decref function */
    else if (dst_dtype->type_num == NPY_OBJECT) {
        if (out_needs_api) {
            *out_needs_api = 1;
        }

        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    /* If there are subarrays, need to wrap it */
    else if (PyDataType_HASSUBARRAY(dst_dtype)) {
        //TODO: implement later
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }
    /* If there are fields, need to do each field */
    else if (PyDataType_HASFIELDS(dst_dtype)) {
        *out_stransfer = NULL;
        *out_transferdata = NULL;
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}

static void
_dec_src_ref_nop(void *NPY_UNUSED(dst),
                        npy_intp NPY_UNUSED(dst_stride),
                        void *NPY_UNUSED(src), npy_intp NPY_UNUSED(src_stride),
                        npy_intp NPY_UNUSED(N),
                        npy_intp NPY_UNUSED(src_itemsize),
                        NpyAuxData *NPY_UNUSED(data), int device)
{
    /* NOP */
}

/***************** WRAP ALIGNED CONTIGUOUS TRANSFER FUNCTION **************/

/* Wraps a transfer function + data in alignment code */
typedef struct {
    NpyAuxData base;
    int device;
    PyMicArray_StridedUnaryOp *wrapped,
                *tobuffer, *frombuffer;
    NpyAuxData *wrappeddata, *todata, *fromdata;
    npy_intp src_itemsize, dst_itemsize;
    char *bufferin, *bufferout;
} _align_wrap_data;

/* transfer data free function */
static void _align_wrap_data_free(NpyAuxData *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    NPY_AUXDATA_FREE(d->wrappeddata);
    NPY_AUXDATA_FREE(d->todata);
    NPY_AUXDATA_FREE(d->fromdata);
    target_free(d->bufferin, d->device);
    PyArray_free(data);
}

/* transfer data copy function */
static NpyAuxData *_align_wrap_data_clone(NpyAuxData *data)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    _align_wrap_data *newdata;
    npy_intp basedatasize, buffersize, datasize;

    /* Round up the structure size to 16-byte boundary */
    basedatasize = (sizeof(_align_wrap_data)+15)&(-0x10);
    /* Add space for two low level buffers */
    buffersize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE*d->src_itemsize +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*d->dst_itemsize;
    datasize = basedatasize + buffersize;

    /* Allocate the data, and populate it */
    newdata = (_align_wrap_data *)PyArray_malloc(basedatasize);
    if (newdata == NULL) {
        return NULL;
    }
    memcpy(newdata, data, basedatasize);
    //newdata->bufferin = (char *)newdata + basedatasize;
    newdata->bufferin = (char *) target_alloc(buffersize, newdata->device);
    if (newdata->bufferin == NULL) {
        PyArray_free(newdata);
        return NULL;
    }
    newdata->bufferout = newdata->bufferin +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*newdata->src_itemsize;
    if (newdata->wrappeddata != NULL) {
        newdata->wrappeddata = NPY_AUXDATA_CLONE(d->wrappeddata);
        if (newdata->wrappeddata == NULL) {
            target_free(newdata->bufferin, newdata->device);
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->todata != NULL) {
        newdata->todata = NPY_AUXDATA_CLONE(d->todata);
        if (newdata->todata == NULL) {
            NPY_AUXDATA_FREE(newdata->wrappeddata);
            target_free(newdata->bufferin, newdata->device);
            PyArray_free(newdata);
            return NULL;
        }
    }
    if (newdata->fromdata != NULL) {
        newdata->fromdata = NPY_AUXDATA_CLONE(d->fromdata);
        if (newdata->fromdata == NULL) {
            NPY_AUXDATA_FREE(newdata->wrappeddata);
            NPY_AUXDATA_FREE(newdata->todata);
            target_free(newdata->bufferin, newdata->device);
            PyArray_free(newdata);
            return NULL;
        }
    }

    return (NpyAuxData *)newdata;
}

static void
_strided_to_strided_contig_align_wrap(void *dst, npy_intp dst_stride,
                        void *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data, int device)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    PyMicArray_StridedUnaryOp *wrapped = d->wrapped,
            *tobuffer = d->tobuffer,
            *frombuffer = d->frombuffer;
    npy_intp inner_src_itemsize = d->src_itemsize,
             dst_itemsize = d->dst_itemsize;
    NpyAuxData *wrappeddata = d->wrappeddata,
            *todata = d->todata,
            *fromdata = d->fromdata;
    char *bufferin = d->bufferin, *bufferout = d->bufferout;

    if (d->device != device) {
        return;
    }

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, todata, device);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    inner_src_itemsize, wrappeddata, device);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    dst_itemsize, fromdata, device);
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride, N,
                                            src_itemsize, todata, device);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize, N,
                                            inner_src_itemsize, wrappeddata, device);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize, N,
                                            dst_itemsize, fromdata, device);
            return;
        }
    }
}

static void
_strided_to_strided_contig_align_wrap_init_dest(void *dst, npy_intp dst_stride,
                        void *src, npy_intp src_stride,
                        npy_intp N, npy_intp src_itemsize,
                        NpyAuxData *data, int device)
{
    _align_wrap_data *d = (_align_wrap_data *)data;
    PyMicArray_StridedUnaryOp *wrapped = d->wrapped,
            *tobuffer = d->tobuffer,
            *frombuffer = d->frombuffer;
    npy_intp inner_src_itemsize = d->src_itemsize,
             dst_itemsize = d->dst_itemsize;
    NpyAuxData *wrappeddata = d->wrappeddata,
            *todata = d->todata,
            *fromdata = d->fromdata;
    char *bufferin = d->bufferin, *bufferout = d->bufferout;

    if (d->device != device) {
        return;
    }

    for(;;) {
        if (N > NPY_LOWLEVEL_BUFFER_BLOCKSIZE) {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    src_itemsize, todata, device);
            target_memset(bufferout, 0, dst_itemsize*NPY_LOWLEVEL_BUFFER_BLOCKSIZE, device);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    inner_src_itemsize, wrappeddata, device);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize,
                                    NPY_LOWLEVEL_BUFFER_BLOCKSIZE,
                                    dst_itemsize, fromdata, device);
            N -= NPY_LOWLEVEL_BUFFER_BLOCKSIZE;
            src += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_stride;
            dst += NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_stride;
        }
        else {
            tobuffer(bufferin, inner_src_itemsize, src, src_stride, N,
                                            src_itemsize, todata, device);
            target_memset(bufferout, 0, dst_itemsize*N, device);
            wrapped(bufferout, dst_itemsize, bufferin, inner_src_itemsize, N,
                                            inner_src_itemsize, wrappeddata, device);
            frombuffer(dst, dst_stride, bufferout, dst_itemsize, N,
                                            dst_itemsize, fromdata, device);
            return;
        }
    }
}

/*
 * Wraps an aligned contig to contig transfer function between either
 * copies or byte swaps to temporary buffers.
 *
 * src_itemsize/dst_itemsize - The sizes of the src and dst datatypes.
 * tobuffer - copy/swap function from src to an aligned contiguous buffer.
 * todata - data for tobuffer
 * frombuffer - copy/swap function from an aligned contiguous buffer to dst.
 * fromdata - data for frombuffer
 * wrapped - contig to contig transfer function being wrapped
 * wrappeddata - data for wrapped
 * init_dest - 1 means to memset the dest buffer to 0 before calling wrapped.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
NPY_NO_EXPORT int
wrap_aligned_contig_transfer_function(
            int device,
            npy_intp src_itemsize, npy_intp dst_itemsize,
            PyMicArray_StridedUnaryOp *tobuffer, NpyAuxData *todata,
            PyMicArray_StridedUnaryOp *frombuffer, NpyAuxData *fromdata,
            PyMicArray_StridedUnaryOp *wrapped, NpyAuxData *wrappeddata,
            int init_dest,
            PyMicArray_StridedUnaryOp **out_stransfer,
            NpyAuxData **out_transferdata)
{
    _align_wrap_data *data;
    npy_intp basedatasize, buffersize, datasize;

    /* Round up the structure size to 16-byte boundary */
    basedatasize = (sizeof(_align_wrap_data)+15)&(-0x10);
    /* Add space for two low level buffers */
    buffersize = NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_itemsize +
                 NPY_LOWLEVEL_BUFFER_BLOCKSIZE*dst_itemsize;
    datasize = basedatasize + buffersize;

    /* Allocate the data, and populate it */
    data = (_align_wrap_data *)PyArray_malloc(basedatasize);
    if (data == NULL) {
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->bufferin = (char *) target_alloc(buffersize, device);
    if (data->bufferin == NULL) {
        PyArray_free(data);
        PyErr_NoMemory();
        return NPY_FAIL;
    }
    data->base.free = &_align_wrap_data_free;
    data->base.clone = &_align_wrap_data_clone;
    data->device = device;
    data->tobuffer = tobuffer;
    data->todata = todata;
    data->frombuffer = frombuffer;
    data->fromdata = fromdata;
    data->wrapped = wrapped;
    data->wrappeddata = wrappeddata;
    data->src_itemsize = src_itemsize;
    data->dst_itemsize = dst_itemsize;
    //data->bufferin = (char *)data + basedatasize;
    data->bufferout = data->bufferin +
                NPY_LOWLEVEL_BUFFER_BLOCKSIZE*src_itemsize;

    /* Set the function and data */
    if (init_dest) {
        *out_stransfer = &_strided_to_strided_contig_align_wrap_init_dest;
    }
    else {
        *out_stransfer = &_strided_to_strided_contig_align_wrap;
    }
    *out_transferdata = (NpyAuxData *)data;

    return NPY_SUCCEED;
}

/*************************** DTYPE CAST FUNCTIONS *************************/

static int
get_nbo_cast_numeric_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            int src_type_num, int dst_type_num,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata)
{
    /* Emit a warning if complex imaginary is being cast away */
    if (PyTypeNum_ISCOMPLEX(src_type_num) &&
                    !PyTypeNum_ISCOMPLEX(dst_type_num) &&
                    !PyTypeNum_ISBOOL(dst_type_num)) {
        PyObject *cls = NULL, *obj = NULL;
        int ret;
        obj = PyImport_ImportModule("numpy.core");
        if (obj) {
            cls = PyObject_GetAttrString(obj, "ComplexWarning");
            Py_DECREF(obj);
        }
        ret = PyErr_WarnEx(cls,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        Py_XDECREF(cls);
        if (ret < 0) {
            return NPY_FAIL;
        }
    }

    *out_stransfer = PyMicArray_GetStridedNumericCastFn(aligned,
                                src_stride, dst_stride,
                                src_type_num, dst_type_num);
    *out_transferdata = NULL;
    if (*out_stransfer == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "unexpected error in GetStridedNumericCastFn");
        return NPY_FAIL;
    }

    return NPY_SUCCEED;
}

static int
get_nbo_cast_transfer_function(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api,
                            int *out_needs_wrap)
{
    //_strided_cast_data *data;
    //PyMicArray_VectorUnaryFunc *castfunc;
    //PyArray_Descr *tmp_dtype;
    npy_intp shape = 1, src_itemsize = src_dtype->elsize,
            dst_itemsize = dst_dtype->elsize;

    if (PyTypeNum_ISNUMBER(src_dtype->type_num) &&
                    PyTypeNum_ISNUMBER(dst_dtype->type_num)) {
        *out_needs_wrap = !PyArray_ISNBO(src_dtype->byteorder) ||
                          !PyArray_ISNBO(dst_dtype->byteorder);
        return get_nbo_cast_numeric_transfer_function(aligned,
                                    src_stride, dst_stride,
                                    src_dtype->type_num, dst_dtype->type_num,
                                    out_stransfer, out_transferdata);
    }

    *out_stransfer = NULL;
    *out_transferdata = NULL;
    return NPY_FAIL;
}

static int
get_cast_transfer_function(int device, int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    PyMicArray_StridedUnaryOp *caststransfer;
    NpyAuxData *castdata, *todata = NULL, *fromdata = NULL;
    int needs_wrap = 0;
    npy_intp src_itemsize = src_dtype->elsize,
            dst_itemsize = dst_dtype->elsize;

    if (get_nbo_cast_transfer_function(aligned,
                            src_stride, dst_stride,
                            src_dtype, dst_dtype,
                            move_references,
                            &caststransfer,
                            &castdata,
                            out_needs_api,
                            &needs_wrap) != NPY_SUCCEED) {
        return NPY_FAIL;
    }

    /*
     * If all native byte order and doesn't need alignment wrapping,
     * return the function
     */
    if (!needs_wrap) {
        *out_stransfer = caststransfer;
        *out_transferdata = castdata;

        return NPY_SUCCEED;
    }
    /* Otherwise, we have to copy and/or swap to aligned temporaries */
    else {
        PyMicArray_StridedUnaryOp *tobuffer, *frombuffer;

        /* Get the copy/swap operation from src */
        PyMicArray_GetDTypeCopySwapFn(aligned,
                                src_stride, src_itemsize,
                                src_dtype,
                                &tobuffer, &todata);


        /* Get the copy/swap operation to dst */
        PyMicArray_GetDTypeCopySwapFn(aligned,
                                dst_itemsize, dst_stride,
                                dst_dtype,
                                &frombuffer, &fromdata);

        if (frombuffer == NULL || tobuffer == NULL) {
            NPY_AUXDATA_FREE(castdata);
            NPY_AUXDATA_FREE(todata);
            NPY_AUXDATA_FREE(fromdata);
            return NPY_FAIL;
        }

        *out_stransfer = caststransfer;

        /* Wrap it all up in a new transfer function + data */
        if (wrap_aligned_contig_transfer_function(
                            device,
                            src_itemsize, dst_itemsize,
                            tobuffer, todata,
                            frombuffer, fromdata,
                            caststransfer, castdata,
                            PyDataType_FLAGCHK(dst_dtype, NPY_NEEDS_INIT),
                            out_stransfer, out_transferdata) != NPY_SUCCEED) {
            NPY_AUXDATA_FREE(castdata);
            NPY_AUXDATA_FREE(todata);
            NPY_AUXDATA_FREE(fromdata);
            return NPY_FAIL;
        }

        return NPY_SUCCEED;
    }
}

/********************* DTYPE COPY SWAP FUNCTION ***********************/

NPY_NO_EXPORT int
PyMicArray_GetDTypeCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *dtype,
                            PyMicArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata)
{
   npy_intp itemsize = dtype->elsize;

    /* If it's a custom data type, wrap its copy swap function */
    if (dtype->type_num >= NPY_NTYPES) {
        *outstransfer = NULL;
        *outtransferdata = NULL;
    }
    /* A straight copy */
    else if (itemsize == 1 || PyArray_ISNBO(dtype->byteorder)) {
        *outstransfer = PyMicArray_GetStridedCopyFn(aligned,
                                    src_stride, dst_stride,
                                    itemsize);
        *outtransferdata = NULL;
    }
    else if (dtype->kind == 'U') {
        *outstransfer = NULL;
        *outtransferdata = NULL;
    }
    /* If it's not complex, one swap */
    else if (dtype->kind != 'c') {
        *outstransfer = PyMicArray_GetStridedCopySwapFn(aligned,
                                    src_stride, dst_stride,
                                    itemsize);
        *outtransferdata = NULL;
    }
    /* If complex, a paired swap */
    else {
        *outstransfer = PyMicArray_GetStridedCopySwapPairFn(aligned,
                                    src_stride, dst_stride,
                                    itemsize);
        *outtransferdata = NULL;
    }

    return (*outstransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
}

/********************* MAIN DTYPE TRANSFER FUNCTION ***********************/

NPY_NO_EXPORT int
PyMicArray_GetDTypeTransferFunction(int device, int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api)
{
    npy_intp src_itemsize, dst_itemsize;
    int src_type_num, dst_type_num;

#if NPY_DT_DBG_TRACING
    printf("Calculating dtype transfer from ");
    PyObject_Print((PyObject *)src_dtype, stdout, 0);
    printf(" to ");
    PyObject_Print((PyObject *)dst_dtype, stdout, 0);
    printf("\n");
#endif

    /*
     * If one of the dtypes is NULL, we give back either a src decref
     * function or a dst setzero function
     */
    if (dst_dtype == NULL) {
        if (move_references) {
            return NPY_FAIL;
        }
        else {
            *out_stransfer = &_dec_src_ref_nop;
            *out_transferdata = NULL;
            return NPY_SUCCEED;
        }
    }
    else if (src_dtype == NULL) {
        return get_setdstzero_transfer_function(aligned,
                                dst_dtype->elsize,
                                dst_dtype,
                                out_stransfer, out_transferdata,
                                out_needs_api);
    }

    src_itemsize = src_dtype->elsize;
    dst_itemsize = dst_dtype->elsize;
    src_type_num = src_dtype->type_num;
    dst_type_num = dst_dtype->type_num;

    /* Common special case - number -> number NBO cast */
    if (PyTypeNum_ISNUMBER(src_type_num) &&
                    PyTypeNum_ISNUMBER(dst_type_num) &&
                    PyArray_ISNBO(src_dtype->byteorder) &&
                    PyArray_ISNBO(dst_dtype->byteorder)) {

        if (PyArray_EquivTypenums(src_type_num, dst_type_num)) {
            *out_stransfer = PyMicArray_GetStridedCopyFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
        else {
            return get_nbo_cast_numeric_transfer_function (aligned,
                                        src_stride, dst_stride,
                                        src_type_num, dst_type_num,
                                        out_stransfer, out_transferdata);
        }
    }

    /*
     * If there are no references and the data types are equivalent,
     * return a simple copy
     */
    if (!PyDataType_REFCHK(src_dtype) && !PyDataType_REFCHK(dst_dtype) &&
                            PyArray_EquivTypes(src_dtype, dst_dtype)) {
        /*
         * We can't pass through the aligned flag because it's not
         * appropriate. Consider a size-8 string, it will say it's
         * aligned because strings only need alignment 1, but the
         * copy function wants to know if it's alignment 8.
         *
         * TODO: Change align from a flag to a "best power of 2 alignment"
         *       which holds the strongest alignment value for all
         *       the data which will be used.
         */
        *out_stransfer = PyMicArray_GetStridedCopyFn(0,
                                        src_stride, dst_stride,
                                        src_dtype->elsize);
        *out_transferdata = NULL;
        return NPY_SUCCEED;
    }

    /* First look at the possibilities of just a copy or swap */
    if (src_itemsize == dst_itemsize && src_dtype->kind == dst_dtype->kind &&
                !PyDataType_HASFIELDS(src_dtype) &&
                !PyDataType_HASFIELDS(dst_dtype) &&
                !PyDataType_HASSUBARRAY(src_dtype) &&
                !PyDataType_HASSUBARRAY(dst_dtype) &&
                src_type_num != NPY_DATETIME && src_type_num != NPY_TIMEDELTA) {
        /* A custom data type requires that we use its copy/swap */
        if (src_type_num >= NPY_NTYPES || dst_type_num >= NPY_NTYPES) {
            /*
             * If the sizes and kinds are identical, but they're different
             * custom types, then get a cast function
             */
            return NPY_FAIL;
        }

        /* The special types, which have no or subelement byte-order */
        switch (src_type_num) {
            case NPY_UNICODE:
            case NPY_VOID:
            case NPY_STRING:
            case NPY_OBJECT:
                return NPY_FAIL;
        }

        /* This is a straight copy */
        if (src_itemsize == 1 || PyArray_ISNBO(src_dtype->byteorder) ==
                                 PyArray_ISNBO(dst_dtype->byteorder)) {
            *out_stransfer = PyMicArray_GetStridedCopyFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
        /* This is a straight copy + byte swap */
        else if (!PyTypeNum_ISCOMPLEX(src_type_num)) {
            *out_stransfer = PyMicArray_GetStridedCopySwapFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
        /* This is a straight copy + element pair byte swap */
        else {
            *out_stransfer = PyMicArray_GetStridedCopySwapPairFn(aligned,
                                        src_stride, dst_stride,
                                        src_itemsize);
            *out_transferdata = NULL;
            return (*out_stransfer == NULL) ? NPY_FAIL : NPY_SUCCEED;
        }
    }

    /* Handle subarrays */
    if (PyDataType_HASSUBARRAY(src_dtype) ||
                                PyDataType_HASSUBARRAY(dst_dtype)) {
        return NPY_FAIL;
    }

    /* Handle fields */
    if ((PyDataType_HASFIELDS(src_dtype) || PyDataType_HASFIELDS(dst_dtype)) &&
            src_type_num != NPY_OBJECT && dst_type_num != NPY_OBJECT) {
        //TODO: figure out what field is
        return NPY_FAIL;
    }

    /* Check for different-sized strings, unicodes, or voids */
    if (src_type_num == dst_type_num) {
        switch (src_type_num) {
        case NPY_UNICODE:
        case NPY_STRING:
        case NPY_VOID:
            return NPY_FAIL;
        }
    }

    /* Otherwise a cast is necessary */
    return get_cast_transfer_function(device, aligned,
                    src_stride, dst_stride,
                    src_dtype, dst_dtype,
                    move_references,
                    out_stransfer, out_transferdata,
                    out_needs_api);
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
                            PyMicArray_MaskedStridedUnaryOp **out_stransfer,
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
PyMicArray_PrepareTwoRawArrayIter(int ndim, npy_intp *shape,
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
PyMicArray_PrepareThreeRawArrayIter(int ndim, npy_intp *shape,
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
