#ifndef _MPY_ARRAYTYPES_H_
#define _MPY_ARRAYTYPES_H_

//#include "common.h"


/* needed for blasfuncs */
NPY_NO_EXPORT void
FLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, int);

NPY_NO_EXPORT void
CFLOAT_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, int);

NPY_NO_EXPORT void
DOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, int);

NPY_NO_EXPORT void
CDOUBLE_dot(char *, npy_intp, char *, npy_intp, char *, npy_intp, int);

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
