#ifndef _MPY_ARRAY_API_CREATOR_H_
#define _MPY_ARRAY_API_CREATOR_H_

#define PyMicArray_API _PyMicArray_CAPI

#if ((PY_MAJOR_VERSION < 3 && PY_VERSION_HEX >= 0x02070000 ) \
        || PY_VERSION_HEX >= 0x03010000)
#define PyMicArray_API_USE_CAPSULE
#endif

#define init_PyMicArray_API() \
    static void *PyMicArray_API[] = {\
        (void *) &PyMicArray_Type,\
        (void *) &PyMicArray_New,\
        (void *) &PyMicArray_NewFromDescr,\
        (void *) &PyMicArray_NewLikeArray,\
        (void *) &PyMicArray_Empty,\
        (void *) &PyMicArray_Zeros,\
        (void *) &PyMicArray_FromAny,\
        (void *) &PyMicArray_FromArray,\
        (void *) &PyMicArray_CopyAnyInto,\
        (void *) &PyMicArray_CopyInto,\
        (void *) &PyMicArray_CopyIntoHost,\
        (void *) &PyMicArray_CopyIntoFromHost,\
        (void *) &PyMicArray_CopyAsFlat,\
        (void *) PyMicArray_MoveInto,\
        (void *) &PyMicArray_NewCopy,\
        (void *) &PyMicArray_View,\
        (void *) &PyMicArray_Resize,\
        (void *) &PyMicArray_Newshape,\
        (void *) &PyMicArray_Reshape,\
        (void *) &PyMicArray_Transpose,\
        (void *) &PyMicArray_SwapAxes,\
        (void *) &PyMicArray_Ravel,\
        (void *) &PyMicArray_Flatten,\
        (void *) &PyMicArray_RemoveAxesInPlace,\
        (void *) &PyMicArray_Return,\
        (void *) &PyMicArray_ToScalar,\
        (void *) &PyMicArray_FillWithScalar,\
        (void *) &PyMicArray_FailUnlessWriteable,\
        (void *) &PyMicArray_GetNumDevices,\
        (void *) &PyMicArray_GetCurrentDevice,\
        (void *) &MpyIter_AdvancedNew,\
        (void *) &MpyIter_MultiNew,\
        (void *) &MpyIter_New,\
        (void *) &MpyIter_Deallocate,\
        (void *) &MpyIter_Reset,\
        (void *) &MpyIter_ResetBasePointers,\
        (void *) &MpyIter_GetIterIndex,\
        (void *) &MpyIter_GetIterIndexRange,\
        (void *) &MpyIter_GetIterSize,\
        (void *) &MpyIter_GetNDim,\
        (void *) &MpyIter_GetNOp,\
        (void *) &MpyIter_GetShape,\
        (void *) &MpyIter_GetDescrArray,\
        (void *) &MpyIter_GetOperandArray,\
        (void *) &MpyIter_GetDataPtrArray,\
        (void *) &MpyIter_GetInnerLoopSizePtr,\
        (void *) &MpyIter_GetInnerStrideArray,\
        (void *) &MpyIter_GetIterNext,\
        (void *) &MpyIter_GetDevice,\
        (void *) &MpyIter_IterationNeedsAPI,\
        (void *) &MpyIter_GetOffDataPtrArray,\
        (void *) &MpyIter_GetOffInnerLoopSizePtr,\
        (void *) &MpyIter_GetOffInnerStrideArray,\
        (void *) &PyMicArray_SetNumericOps,\
        &PyMicArray_FromScalar,\
        NULL\
    }

#endif
