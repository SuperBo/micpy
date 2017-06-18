#ifndef _MPY_COMMON_H
#define _MPY_COMMON_H
#include <omp.h>
#include <offload.h>

#include "arrayobject.h"
#include "mpy_common.h"

#ifndef NMAXDEVICES
#define NMAXDEVICES 2
#endif

NPY_NO_EXPORT int PyMicArray_GetCurrentDevice(void);
NPY_NO_EXPORT int PyMicArray_GetNumDevices(void);

#define NDEVICES (PyMicArray_GetNumDevices())
#define N_DEVICES (PyMicArray_GetNumDevices())

#define CURRENT_DEVICE (PyMicArray_GetCurrentDevice())
#define DEFAULT_DEVICE (PyMicArray_GetCurrentDevice())

/*
 * Define a chunksize for CBLAS. CBLAS counts in integers.
 */
#if NPY_MAX_INTP > INT_MAX
# define MPY_CBLAS_CHUNK  (INT_MAX / 2 + 1)
#else
# define MPY_CBLAS_CHUNK  NPY_MAX_INTP
#endif

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

NPY_NO_EXPORT int
_zerofill(PyMicArrayObject *ret);

/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype);

NPY_NO_EXPORT int
_IsAligned(PyMicArrayObject *ap);

NPY_NO_EXPORT npy_bool
_IsWriteable(PyMicArrayObject *ap);

NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending);

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

NPY_NO_EXPORT void
dot_alignment_error(PyMicArrayObject *a, int i, PyMicArrayObject *b, int j);

NPY_NO_EXPORT int
get_common_device2(PyObject *op1, PyObject *op2);

NPY_NO_EXPORT int
get_common_device(PyObject **ops, int nop);

/*
 * return true if pointer is aligned to 'alignment'
 * borrow from numpy
 */
static NPY_INLINE MPY_TARGET_MIC int
mpy_is_aligned(const void * p, const npy_uintp alignment)
{
    /*
     * alignment is usually a power of two
     * the test is faster than a direct modulo
     */
    if (NPY_LIKELY((alignment & (alignment - 1)) == 0)) {
        return ((npy_uintp)(p) & ((alignment) - 1)) == 0;
    }
    else {
        return ((npy_uintp)(p) % alignment) == 0;
    }
}

/*
 * Returns -1 and sets an exception if *index is an invalid index for
 * an array of size max_item, otherwise adjusts it in place to be
 * 0 <= *index < max_item, and returns 0.
 * 'axis' should be the array axis that is being indexed over, if known. If
 * unknown, use -1.
 * If _save is NULL it is assumed the GIL is taken
 * If _save is not NULL it is assumed the GIL is not taken and it
 * is acquired in the case of an error
 */
static NPY_INLINE int
check_and_adjust_index(npy_intp *index, npy_intp max_item, int axis,
                       PyThreadState * _save)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*index < -max_item) || (*index >= max_item))) {
        NPY_END_THREADS;
        /* Try to be as clear as possible about what went wrong. */
        if (axis >= 0) {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for axis %d with size %"NPY_INTP_FMT,
                         *index, axis, max_item);
        } else {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for size %"NPY_INTP_FMT, *index, max_item);
        }
        return -1;
    }
    /* adjust negative indices */
    if (*index < 0) {
        *index += max_item;
    }
    return 0;
}

/*
 * Convert NumPy stride to BLAS stride. Returns 0 if conversion cannot be done
 * (BLAS won't handle negative or zero strides the way we want).
 */
static NPY_INLINE int
blas_stride(npy_intp stride, unsigned itemsize)
{
    /*
     * Should probably check pointer alignment also, but this may cause
     * problems if we require complex to be 16 byte aligned.
     */
    if (stride > 0 && mpy_is_aligned((void *)stride, itemsize)) {
        stride /= itemsize;
        if (stride <= INT_MAX) {
            return stride;
        }
    }
    return 0;
}

/* Memset on target */
static NPY_INLINE void *
target_memset(void *ptr, int value, size_t num, int device_num)
{
    #pragma omp target device(device_num) map(to: ptr, value, num)
    memset(ptr, value, num);
    return ptr;
}

#define target_alloc omp_target_alloc
#define target_free omp_target_free

#endif