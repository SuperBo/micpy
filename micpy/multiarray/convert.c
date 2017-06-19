#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/npy_3kcompat.h>

#include "npy_config.h"

#define _MICARRAYMODULE
#include "common.h"
#include "arrayobject.h"
#include "scalar.h"
#include "creators.h"
#include "array_assign.h"
//#include "mapping.h"
#include "convert.h"
#include "lowlevel_strided_loops.h"

int
fallocate(int fd, int mode, off_t offset, off_t len);

/*
 * allocate nbytes of diskspace for file fp
 * this allows the filesystem to make smarter allocation decisions and gives a
 * fast exit on not enough free space
 * returns -1 and raises exception on no space, ignores all other errors
 */
static int
npy_fallocate(npy_intp nbytes, FILE * fp)
{
    /*
     * unknown behavior on non-linux so don't try it
     * we don't want explicit zeroing to happen
     */
#if defined(HAVE_FALLOCATE) && defined(__linux__)
    int r;
    /* small files not worth the system call */
    if (nbytes < 16 * 1024 * 1024) {
        return 0;
    }

    /* btrfs can take a while to allocate making release worthwhile */
    NPY_BEGIN_ALLOW_THREADS;
    /*
     * flush in case there might be some unexpected interactions between the
     * fallocate call and unwritten data in the descriptor
     */
    fflush(fp);
    /*
     * the flag "1" (=FALLOC_FL_KEEP_SIZE) is needed for the case of files
     * opened in append mode (issue #8329)
     */
    r = fallocate(fileno(fp), 1, npy_ftell(fp), nbytes);
    NPY_END_ALLOW_THREADS;

    /*
     * early exit on no space, other errors will also get found during fwrite
     */
    if (r == -1 && errno == ENOSPC) {
        PyErr_Format(PyExc_IOError, "Not enough free space to write "
                     "%"NPY_INTP_FMT" bytes", nbytes);
        return -1;
    }
#endif
    return 0;
}


/*NUMPY_API
 * To List
 */
NPY_NO_EXPORT PyObject *
PyMicArray_ToList(PyMicArrayObject *self)
{
    /*return PyArray_ToList((PyArrayObject *)self);*/
    return NULL;
}

/* XXX: FIXME --- add ordering argument to
   Allow Fortran ordering on write
   This will need the addition of a Fortran-order iterator.
 */


/*NUMPY_API*/
NPY_NO_EXPORT int
PyMicArray_FillWithScalar(PyMicArrayObject *arr, PyObject *obj)
{
    PyArray_Descr *dtype = NULL;
    npy_longlong value_buffer[4];
    char *value = NULL;
    int retcode = 0, device = CPU_DEVICE;

    /*
     * If 'arr' is an object array, copy the object as is unless
     * 'obj' is a zero-dimensional array, in which case we copy
     * the element in that array instead.
     */
    if (PyMicArray_DESCR(arr)->type_num == NPY_OBJECT) {
        //TODO(superbo): print error here
        return -1;
    }
    /* NumPy scalar */
    else if (PyArray_IsScalar(obj, Generic)) {
        dtype = PyArray_DescrFromScalar(obj);
        if (dtype == NULL) {
            return -1;
        }
        value = scalar_value(obj, dtype);
        if (value == NULL) {
            Py_DECREF(dtype);
            return -1;
        }
    }
    /* MicArray */
    else if (PyMicArray_Check(obj)) {
        PyMicArrayObject *src_arr = (PyMicArrayObject *) obj;
        if (PyMicArray_NDIM(src_arr) != 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Input object to FillWithScalar is not a scalar");
            return -1;
        }

        value = PyMicArray_BYTES(src_arr);
        device = PyMicArray_DEVICE(src_arr);
        dtype = PyMicArray_DESCR(src_arr);
        Py_INCREF(dtype);
    }
    /* Python boolean */
    else if (PyBool_Check(obj)) {
        value = (char *)value_buffer;
        *value = (obj == Py_True);

        dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            return -1;
        }
    }
    /* Python integer */
    else if (PyLong_Check(obj) || PyInt_Check(obj)) {
        /* Try long long before unsigned long long */
        npy_longlong ll_v = PyLong_AsLongLong(obj);
        if (ll_v == -1 && PyErr_Occurred()) {
            /* Long long failed, try unsigned long long */
            npy_ulonglong ull_v;
            PyErr_Clear();
            ull_v = PyLong_AsUnsignedLongLong(obj);
            if (ull_v == (unsigned long long)-1 && PyErr_Occurred()) {
                return -1;
            }
            value = (char *)value_buffer;
            *(npy_ulonglong *)value = ull_v;

            dtype = PyArray_DescrFromType(NPY_ULONGLONG);
            if (dtype == NULL) {
                return -1;
            }
        }
        else {
            /* Long long succeeded */
            value = (char *)value_buffer;
            *(npy_longlong *)value = ll_v;

            dtype = PyArray_DescrFromType(NPY_LONGLONG);
            if (dtype == NULL) {
                return -1;
            }
        }
    }
    /* Python float */
    else if (PyFloat_Check(obj)) {
        npy_double v = PyFloat_AsDouble(obj);
        if (v == -1 && PyErr_Occurred()) {
            return -1;
        }
        value = (char *)value_buffer;
        *(npy_double *)value = v;

        dtype = PyArray_DescrFromType(NPY_DOUBLE);
        if (dtype == NULL) {
            return -1;
        }
    }
    /* Python complex */
    else if (PyComplex_Check(obj)) {
        npy_double re, im;

        re = PyComplex_RealAsDouble(obj);
        if (re == -1 && PyErr_Occurred()) {
            return -1;
        }
        im = PyComplex_ImagAsDouble(obj);
        if (im == -1 && PyErr_Occurred()) {
            return -1;
        }
        value = (char *)value_buffer;
        ((npy_double *)value)[0] = re;
        ((npy_double *)value)[1] = im;

        dtype = PyArray_DescrFromType(NPY_CDOUBLE);
        if (dtype == NULL) {
            return -1;
        }
    }

    /* Use the value pointer we got if possible */
    if (value != NULL) {
        /* TODO: switch to SAME_KIND casting */
        retcode = PyMicArray_AssignRawScalar(arr, dtype, value,
                                device, NULL, NPY_UNSAFE_CASTING);
        Py_DECREF(dtype);
        return retcode;
    }
    /* Otherwise convert to an array to do the assignment */
    else {
        PyArrayObject *src_arr;

        /**
         * The dtype of the destination is used when converting
         * from the pyobject, so that for example a tuple gets
         * recognized as a struct scalar of the required type.
         */
        Py_INCREF(PyMicArray_DTYPE(arr));
        src_arr = (PyArrayObject *)PyArray_FromAny(obj,
                        PyMicArray_DTYPE(arr), 0, 0, 0, NULL);
        if (src_arr == NULL) {
            return -1;
        }

        if (PyArray_NDIM(src_arr) != 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Input object to FillWithScalar is not a scalar");
            Py_DECREF(src_arr);
            return -1;
        }

        retcode = PyMicArray_CopyIntoFromHost(arr, src_arr);

        Py_DECREF(src_arr);
        return retcode;
    }
}

/*
 * Fills an array with zeros.
 *
 * dst: The destination array.
 * wheremask: If non-NULL, a boolean mask specifying where to set the values.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_AssignZero(PyMicArrayObject *dst,
                      PyMicArrayObject *wheremask)
{
    npy_bool value;
    PyArray_Descr *bool_dtype;
    int retcode;

    /* Create a raw bool scalar with the value False */
    bool_dtype = PyArray_DescrFromType(NPY_BOOL);
    if (bool_dtype == NULL) {
        return -1;
    }
    value = 0;

    retcode = PyMicArray_AssignRawScalar(dst, bool_dtype, (char *)&value,
                                      CPU_DEVICE, wheremask, NPY_SAFE_CASTING);

    Py_DECREF(bool_dtype);
    return retcode;
}

/*
 * Fills an array with ones.
 *
 * dst: The destination array.
 * wheremask: If non-NULL, a boolean mask specifying where to set the values.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_AssignOne(PyMicArrayObject *dst,
                     PyMicArrayObject *wheremask)
{
    npy_bool value;
    PyArray_Descr *bool_dtype;
    int retcode;

    /* Create a raw bool scalar with the value True */
    bool_dtype = PyArray_DescrFromType(NPY_BOOL);
    if (bool_dtype == NULL) {
        return -1;
    }
    value = 1;

    retcode = PyMicArray_AssignRawScalar(dst, bool_dtype, (char *)&value,
                                      CPU_DEVICE, wheremask, NPY_SAFE_CASTING);

    Py_DECREF(bool_dtype);
    return retcode;
}

/*NUMPY_API
 * Copy an array.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_NewCopy(PyMicArrayObject *obj, NPY_ORDER order)
{
    PyMicArrayObject *ret;

    ret = (PyMicArrayObject *)PyMicArray_NewLikeArray(PyMicArray_DEVICE(obj),
                                    (PyArrayObject *) obj, order, NULL, 1);
    if (ret == NULL) {
        return NULL;
    }

    if (PyMicArray_AssignArray(ret, obj, NULL, NPY_UNSAFE_CASTING) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject *)ret;
}

/*NUMPY_API
 * View
 * steals a reference to type -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyMicArray_View(PyMicArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
{
    PyMicArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    PyTypeObject *subtype;
    int flags;

    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = Py_TYPE(self);
    }

    if (type != NULL && (PyMicArray_FLAGS(self) & NPY_ARRAY_WARN_ON_WRITE)) {
        const char *msg =
            "Numpy has detected that you may be viewing or writing to an array "
            "returned by selecting multiple fields in a structured array. \n\n"
            "This code may break in numpy 1.13 because this will return a view "
            "instead of a copy -- see release notes for details.";
        /* 2016-09-19, 1.12 */
        if (DEPRECATE_FUTUREWARNING(msg) < 0) {
            return NULL;
        }
        /* Only warn once per array */
        PyMicArray_CLEARFLAGS(self, NPY_ARRAY_WARN_ON_WRITE);
    }

    flags = PyMicArray_FLAGS(self);

    dtype = PyMicArray_DESCR(self);
    Py_INCREF(dtype);
    ret = (PyMicArrayObject *)PyMicArray_NewFromDescr_int(
                               PyMicArray_DEVICE(self),
                               subtype,
                               dtype,
                               PyMicArray_NDIM(self), PyMicArray_DIMS(self),
                               PyMicArray_STRIDES(self),
                               PyMicArray_DATA(self),
                               flags,
                               (PyObject *)self, 0, 1);
    if (ret == NULL) {
        Py_XDECREF(type);
        return NULL;
    }

    /* Set the base object */
    Py_INCREF(self);
    if (PyMicArray_SetBaseObject(ret, (PyObject *)self) < 0) {
        Py_DECREF(ret);
        Py_XDECREF(type);
        return NULL;
    }

    if (type != NULL) {
        if (PyObject_SetAttrString((PyObject *)ret, "dtype",
                                   (PyObject *)type) < 0) {
            Py_DECREF(ret);
            Py_DECREF(type);
            return NULL;
        }
        Py_DECREF(type);
    }
    return (PyObject *)ret;
}
