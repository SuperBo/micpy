#ifndef _MPY_EXTERNAL_COMMON_H
#define _MPY_EXTERNAL_COMMON_H

#include <numpy/npy_os.h>

/* Some usefull macros */

#ifdef NPY_OS_WIN32
#define MPY_TARGET_MIC __declspec(target(mic))
#else
#define MPY_TARGET_MIC __attribute__((target(mic)))
#endif


/* Array iter part */
typedef struct MpyIter_InternalOnly MpyIter;
typedef int (MpyIter_IterNextFunc)(MpyIter *iter);
typedef void (MpyIter_GetMultiIndexFunc)(MpyIter *iter,
                                      npy_intp *outcoords);

#ifdef NPY_ALLOW_THREADS
#define MPY_BEGIN_THREADS_NDITER(iter) \
        do { \
            if (!MpyIter_IterationNeedsAPI(iter)) { \
                NPY_BEGIN_THREADS_THRESHOLDED(MpyIter_GetIterSize(iter)); \
            } \
        } while(0)
#else
#define MPY_BEGIN_THREADS_NDITER(iter)
#endif

/* Backward compatibility for numpy 1.12 */
#ifndef NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE
#define NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE 0x40000000
#endif

/* End array iter */

#endif
