#ifndef MPY_DTYPE_TRANSFER_H_
#define MPY_DTYPE_TRANSFER_H_

NPY_NO_EXPORT int
PyMicArray_GetDTypeCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *dtype,
                            PyMicArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata);

NPY_NO_EXPORT int
PyMicArray_GetDTypeTransferFunction(int device, int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                            int move_references,
                            PyMicArray_StridedUnaryOp **out_stransfer,
                            NpyAuxData **out_transferdata,
                            int *out_needs_api);

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
                            int *out_needs_api);

NPY_NO_EXPORT int
PyMicArray_CastRawArrays(npy_intp count,
                      char *src, char *dst,
                      npy_intp src_stride, npy_intp dst_stride,
                      PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                      int move_references);

NPY_NO_EXPORT int
PyMicArray_PrepareOneRawArrayIter(int ndim, npy_intp *shape,
                            char *data, npy_intp *strides,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_data, npy_intp *out_strides);

NPY_NO_EXPORT int
PyMicArray_PrepareTwoRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB);

NPY_NO_EXPORT int
PyMicArray_PrepareThreeRawArrayIter(int ndim, npy_intp *shape,
                            char *dataA, npy_intp *stridesA,
                            char *dataB, npy_intp *stridesB,
                            char *dataC, npy_intp *stridesC,
                            int *out_ndim, npy_intp *out_shape,
                            char **out_dataA, npy_intp *out_stridesA,
                            char **out_dataB, npy_intp *out_stridesB,
                            char **out_dataC, npy_intp *out_stridesC);

#endif
