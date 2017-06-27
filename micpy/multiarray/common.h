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

#define CPU_DEVICE (omp_get_initial_device())

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
 * Returns -1 and sets an exception if *axis is an invalid axis for
 * an array of dimension ndim, otherwise adjusts it in place to be
 * 0 <= *axis < ndim, and returns 0.
 *
 * msg_prefix: borrowed reference, a string to prepend to the message
 */
static NPY_INLINE int
check_and_adjust_axis_msg(int *axis, int ndim, PyObject *msg_prefix)
{
    /* Check that index is valid, taking into account negative indices */
    if (NPY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        /*
         * Load the exception type, if we don't already have it. Unfortunately
         * we don't have access to npy_cache_import here
         */
        static PyObject *AxisError_cls = NULL;
        PyObject *exc;

        if (AxisError_cls == NULL) {
            PyObject *mod = PyImport_ImportModule("numpy.core._internal");

            if (mod != NULL) {
                AxisError_cls = PyObject_GetAttrString(mod, "AxisError");
                Py_DECREF(mod);
            }
        }

        /* Invoke the AxisError constructor */
        exc = PyObject_CallFunction(AxisError_cls, "iiO",
                                    *axis, ndim, msg_prefix);
        if (exc == NULL) {
            return -1;
        }
        PyErr_SetObject(AxisError_cls, exc);
        Py_DECREF(exc);

        return -1;
    }
    /* adjust negative indices */
    if (*axis < 0) {
        *axis += ndim;
    }
    return 0;
}
static NPY_INLINE int
check_and_adjust_axis(int *axis, int ndim)
{
    return check_and_adjust_axis_msg(axis, ndim, Py_None);
}

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

#endif