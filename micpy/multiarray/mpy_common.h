#ifndef _MPY_EXTERNAL_COMMON_H
#define _MPY_EXTERNAL_COMMON_H

#include <omp.h>
#include <offload.h>
#include <numpy/npy_os.h>


/* Some usefull macros */

#ifdef NPY_OS_WIN32
#define MPY_TARGET_MIC __declspec(target(mic))
#else
#define MPY_TARGET_MIC __attribute__((target(mic)))
#endif

/* Memset on target */
static NPY_INLINE void *
target_memset(void *ptr, int value, size_t num, int device_num)
{
    #pragma omp target device(device_num) map(to: ptr, value, num)
    memset(ptr, value, num);
    return ptr;
}

#define target_alloc omp_target_alloc
#define target_malloc omp_target_alloc
#define target_free omp_target_free
#define target_memcpy(dst, src, len, dst_dev, src_dev) \
                omp_target_memcpy(dst, src, len, 0, 0, dst_dev, src_dev)


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
