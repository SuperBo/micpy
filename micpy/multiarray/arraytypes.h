#ifndef _MPY_ARRAYTYPES_H_
#define _MPY_ARRAYTYPES_H_

//#include "common.h"
#include "mpyndarraytypes.h"


/* needed for blasfuncs */
NPY_NO_EXPORT void
FLOAT_dot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT void
CFLOAT_dot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT void
DOUBLE_dot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT void
CDOUBLE_dot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT void
CFLOAT_vdot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT void
CDOUBLE_vdot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT void
CLONGDOUBLE_vdot(void *, npy_intp, void *, npy_intp, void *, npy_intp, int);

NPY_NO_EXPORT PyMicArray_ArrFuncs *
PyMicArray_GetArrFuncs(int typenum);

/* for _pyarray_correlate */
NPY_NO_EXPORT int
small_correlate(const char * d_, npy_intp dstride,
                npy_intp nd, enum NPY_TYPES dtype,
                const char * k_, npy_intp kstride,
                npy_intp nk, enum NPY_TYPES ktype,
                char * out_, npy_intp ostride);

#endif
