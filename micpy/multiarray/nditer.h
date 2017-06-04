#ifndef _MPY_NDITER_H_
#define _MPY_NDITER_H_

NPY_NO_EXPORT MpyIter_IterNextFunc *
MpyIter_GetIterNext(MpyIter *iter, char **errmsg);

NPY_NO_EXPORT MpyIter *
MpyIter_AdvancedNew(int nop, PyMicArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes,
                 int oa_ndim, int **op_axes, npy_intp *itershape,
                 npy_intp buffersize);

NPY_NO_EXPORT MpyIter *
MpyIter_MultiNew(int nop, PyMicArrayObject **op_in, npy_uint32 flags,
                 NPY_ORDER order, NPY_CASTING casting,
                 npy_uint32 *op_flags,
                 PyArray_Descr **op_request_dtypes);

NPY_NO_EXPORT MpyIter *
MpyIter_New(PyMicArrayObject *op, npy_uint32 flags,
                  NPY_ORDER order, NPY_CASTING casting,
                  PyArray_Descr* dtype);

NPY_NO_EXPORT int
MpyIter_Deallocate(MpyIter *iter);

NPY_NO_EXPORT int
MpyIter_RemoveAxis(MpyIter *iter, int axis);

NPY_NO_EXPORT int
MpyIter_RemoveMultiIndex(MpyIter *iter);

NPY_NO_EXPORT int
MpyIter_EnableExternalLoop(MpyIter *iter);

NPY_NO_EXPORT int
MpyIter_Reset(MpyIter *iter, char **errmsg);

NPY_NO_EXPORT int
MpyIter_ResetBasePointers(MpyIter *iter, char **baseptrs, char **errmsg);

NPY_NO_EXPORT int
MpyIter_ResetToIterIndexRange(MpyIter *iter,
                              npy_intp istart, npy_intp iend, char **errmsg);

NPY_NO_EXPORT int
MpyIter_GotoMultiIndex(MpyIter *iter, npy_intp *multi_index);

NPY_NO_EXPORT int
MpyIter_GotoIndex(MpyIter *iter, npy_intp flat_index);

NPY_NO_EXPORT int
MpyIter_GotoIterIndex(MpyIter *iter, npy_intp iterindex);

NPY_NO_EXPORT npy_intp
MpyIter_GetIterIndex(MpyIter *iter);

NPY_NO_EXPORT npy_bool
MpyIter_HasDelayedBufAlloc(MpyIter *iter);

NPY_NO_EXPORT npy_bool
MpyIter_HasExternalLoop(MpyIter *iter);

NPY_NO_EXPORT npy_bool
MpyIter_HasMultiIndex(MpyIter *iter);

NPY_NO_EXPORT npy_bool
MpyIter_HasIndex(MpyIter *iter);

NPY_NO_EXPORT npy_bool
MpyIter_RequiresBuffering(MpyIter *iter);

NPY_NO_EXPORT int
MpyIter_GetNDim(MpyIter *iter);

NPY_NO_EXPORT int
MpyIter_GetNOp(MpyIter *iter);

NPY_NO_EXPORT npy_intp
MpyIter_GetIterSize(MpyIter *iter);

NPY_NO_EXPORT npy_bool
MpyIter_IsBuffered(MpyIter *iter);

NPY_NO_EXPORT npy_intp
MpyIter_GetBufferSize(MpyIter *iter);

NPY_NO_EXPORT void
MpyIter_GetIterIndexRange(MpyIter *iter,
                          npy_intp *istart, npy_intp *iend);

NPY_NO_EXPORT int
MpyIter_GetShape(MpyIter *iter, npy_intp *outshape);

NPY_NO_EXPORT npy_intp *
MpyIter_GetDataPtrArray(MpyIter *iter);

NPY_NO_EXPORT npy_intp *
MpyIter_GetInitialDataPtrArray(MpyIter *iter);

NPY_NO_EXPORT PyArray_Descr **
MpyIter_GetDescrArray(MpyIter *iter);

NPY_NO_EXPORT PyMicArrayObject **
MpyIter_GetOperandArray(MpyIter *iter);

NPY_NO_EXPORT PyMicArrayObject *
MpyIter_GetIterView(MpyIter *iter, npy_intp i);

NPY_NO_EXPORT npy_intp *
MpyIter_GetIndexPtr(MpyIter *iter);

NPY_NO_EXPORT void
MpyIter_GetReadFlags(MpyIter *iter, char *outreadflags);

NPY_NO_EXPORT void
MpyIter_GetReadFlags(MpyIter *iter, char *outreadflags);

NPY_NO_EXPORT void
MpyIter_GetReadFlags(MpyIter *iter, char *outreadflags);

NPY_NO_EXPORT void
MpyIter_GetWriteFlags(MpyIter *iter, char *outwriteflags);

NPY_NO_EXPORT npy_intp *
MpyIter_GetInnerStrideArray(MpyIter *iter);

NPY_NO_EXPORT npy_intp *
MpyIter_GetAxisStrideArray(MpyIter *iter, int axis);

NPY_NO_EXPORT void
MpyIter_GetInnerFixedStrideArray(MpyIter *iter, npy_intp *out_strides);

NPY_NO_EXPORT npy_intp *
MpyIter_GetInnerLoopSizePtr(MpyIter *iter);

NPY_NO_EXPORT void
MpyIter_DebugPrint(MpyIter *iter);

#endif
