#ifndef MPYNDARRAYTYPES_H
#define MPYNDARRAYTYPES_H

#include "mpy_common.h"

/*
 * These assume aligned and notswapped data -- a buffer will be used
 * before or contiguous data will be obtained
 */

typedef int (PyMicArray_CompareFunc)(const void *, const void *, void *);
typedef int (PyMicArray_ArgFunc)(void*, npy_intp, npy_intp*, void *);

typedef void (PyMicArray_DotFunc)(void *, npy_intp, void *, npy_intp, void *,
                                  npy_intp, int);

typedef void (PyMicArray_VectorUnaryFunc)(void *, void *, npy_intp, int);

/*
 * XXX the ignore argument should be removed next time the API version
 * is bumped. It used to be the separator.
 */

typedef int (PyMicArray_FillFunc)(void *, npy_intp, int);

typedef int (PyMicArray_SortFunc)(void *, npy_intp, void *);
typedef int (PyMicArray_ArgSortFunc)(void *, npy_intp *, npy_intp, void *);
typedef int (PyMicArray_PartitionFunc)(void *, npy_intp, npy_intp,
                                    npy_intp *, npy_intp *,
                                    void *);
typedef int (PyMicArray_ArgPartitionFunc)(void *, npy_intp *, npy_intp, npy_intp,
                                       npy_intp *, npy_intp *,
                                       void *);

typedef int (PyMicArray_FillWithScalarFunc)(void *, npy_intp, void *, void *);

typedef int (PyMicArray_ScalarKindFunc)(void *);

typedef void (PyMicArray_FastClipFunc)(void *in, npy_intp n_in, void *min,
                                    void *max, void *out, int device);
typedef void (PyMicArray_FastPutmaskFunc)(void *in, void *mask, npy_intp n_in,
                                       void *values, npy_intp nv, int device);
typedef int  (PyMicArray_FastTakeFunc)(void *dest, void *src, npy_intp *indarray,
                                       npy_intp nindarray, npy_intp n_outer,
                                       npy_intp m_middle, npy_intp nelem,
                                       NPY_CLIPMODE clipmode, int device);

typedef PyObject * (PyMicArray_GetItemFunc) (void *, void *);
typedef int (PyMicArray_SetItemFunc)(PyObject *, void *, void *);

typedef void (PyMicArray_CopySwapNFunc)(void *, npy_intp, void *, npy_intp,
                                     npy_intp, int, int);

typedef void (PyMicArray_CopySwapFunc)(void *, void *, int, int);

typedef npy_bool (PyMicArray_NonzeroFunc)(void *, void *);

typedef struct {
        /*
         * Functions to cast to most other standard types
         * Can have some NULL entries. The types
         * DATETIME, TIMEDELTA, and HALF go into the castdict
         * even though they are built-in.
         */
        PyMicArray_VectorUnaryFunc *cast[NPY_NTYPES_ABI_COMPATIBLE];

        /* The next four functions *cannot* be NULL */

        /*
         * Functions to get and set items with standard Python types
         * -- not array scalars
         */
        PyMicArray_GetItemFunc *getitem;
        PyMicArray_SetItemFunc *setitem;

        /*
         * Copy and/or swap data.  Memory areas may not overlap
         * Use memmove first if they might
         */
        PyMicArray_CopySwapNFunc *copyswapn;
        PyMicArray_CopySwapFunc *copyswap;

        /*
         * Function to compare items
         * Can be NULL
         */
        PyMicArray_CompareFunc *compare;

        /*
         * Function to select largest
         * Can be NULL
         */
        PyMicArray_ArgFunc *argmax;

        /*
         * Function to compute dot product
         * Can be NULL
         */
        PyMicArray_DotFunc *dotfunc;


        /*
         * Function to determine if data is zero or not
         * If NULL a default version is
         * used at Registration time.
         */
        PyMicArray_NonzeroFunc *nonzero;

        /*
         * Used for arange.
         * Can be NULL.
         */
        PyMicArray_FillFunc *fill;

        /*
         * Function to fill arrays with scalar values
         * Can be NULL
         */
        PyMicArray_FillWithScalarFunc *fillwithscalar;

        /*
         * Sorting functions
         * Can be NULL
         */
        PyMicArray_SortFunc *sort[NPY_NSORTS];
        PyMicArray_ArgSortFunc *argsort[NPY_NSORTS];


        PyMicArray_FastClipFunc *fastclip;
        PyMicArray_FastPutmaskFunc *fastputmask;
        PyMicArray_FastTakeFunc *fasttake;

        /*
         * Function to select smallest
         * Can be NULL
         */
        PyMicArray_ArgFunc *argmin;

} PyMicArray_ArrFuncs;

#endif