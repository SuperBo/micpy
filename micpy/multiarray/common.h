#ifndef _MPY_COMMON_H
#define _MPY_COMMON_H
#include <omp.h>
#include <offload.h>

#include "arrayobject.h"

#ifndef NMAXDEVICES
#define NMAXDEVICES 2
#endif

extern int num_devices;
#define NDEVICES num_devices
#define N_DEVICES num_devices

extern int current_device;
#define CURRENT_DEVICE current_device
#define DEFAULT_DEVICE current_device

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

NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending);

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

/*
 * return true if pointer is aligned to 'alignment'
 * borrow from numpy
 */
static NPY_INLINE int
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

#endif
