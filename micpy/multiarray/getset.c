/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "npy_import.h"
#include "mem_overlap.h"

#define _MICARRAYMODULE
#include "common.h"
#include "arrayobject.h"
#include "creators.h"
#include "getset.h"
#include "shape.h"

/*******************  array attribute get and set routines ******************/

static PyObject *
array_device_get(PyMicArrayObject *self)
{
    return PyInt_FromLong(PyMicArray_DEVICE(self));
}

static PyObject *
array_ndim_get(PyMicArrayObject *self)
{
    return PyInt_FromLong(PyMicArray_NDIM(self));
}

static PyObject *
array_flags_get(PyMicArrayObject *self)
{
    return PyArray_NewFlagsObject((PyObject *)self);
}

static PyObject *
array_shape_get(PyMicArrayObject *self)
{
    return PyArray_IntTupleFromIntp(PyMicArray_NDIM(self), PyMicArray_DIMS(self));
}

static int
array_shape_set(PyMicArrayObject *self, PyObject *val)
{
    int nd;
    PyMicArrayObject *ret;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array shape");
        return -1;
    }
    /* Assumes C-order */
    ret = (PyMicArrayObject *)PyMicArray_Reshape(self, val);
    if (ret == NULL) {
        return -1;
    }
    if (PyMicArray_DATA(ret) != PyMicArray_DATA(self)) {
        Py_DECREF(ret);
        PyErr_SetString(PyExc_AttributeError,
                        "incompatible shape for a non-contiguous "\
                        "array");
        return -1;
    }

    /* Free old dimensions and strides */
    PyDimMem_FREE(PyMicArray_DIMS(self));
    nd = PyMicArray_NDIM(ret);
    ((PyMicArrayObject *)self)->nd = nd;
    if (nd > 0) {
        /* create new dimensions and strides */
        ((PyMicArrayObject *)self)->dimensions = PyDimMem_NEW(3*nd);
        if (PyMicArray_DIMS(self) == NULL) {
            Py_DECREF(ret);
            PyErr_SetString(PyExc_MemoryError,"");
            return -1;
        }
        ((PyMicArrayObject *)self)->strides = PyMicArray_DIMS(self) + nd;
        memcpy(PyMicArray_DIMS(self), PyMicArray_DIMS(ret), nd*sizeof(npy_intp));
        memcpy(PyMicArray_STRIDES(self), PyMicArray_STRIDES(ret), nd*sizeof(npy_intp));
    }
    else {
        ((PyMicArrayObject *)self)->dimensions = NULL;
        ((PyMicArrayObject *)self)->strides = NULL;
    }
    Py_DECREF(ret);
    PyMicArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return 0;
}


static PyObject *
array_strides_get(PyMicArrayObject *self)
{
    return PyArray_IntTupleFromIntp(PyMicArray_NDIM(self), PyMicArray_STRIDES(self));
}

static int
array_strides_set(PyMicArrayObject *self, PyObject *obj)
{
    PyArray_Dims newstrides = {NULL, 0};
    PyMicArrayObject *new;
    npy_intp numbytes = 0;
    npy_intp offset = 0;
    npy_intp lower_offset = 0;
    npy_intp upper_offset = 0;
    Py_ssize_t buf_len;
    char *buf;

    if (obj == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array strides");
        return -1;
    }
    if (!PyArray_IntpConverter(obj, &newstrides) ||
        newstrides.ptr == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid strides");
        return -1;
    }
    if (newstrides.len != PyMicArray_NDIM(self)) {
        PyErr_Format(PyExc_ValueError, "strides must be "       \
                     " same length as shape (%d)", PyMicArray_NDIM(self));
        goto fail;
    }
    new = self;
    while(PyMicArray_BASE(new) && PyMicArray_Check(PyMicArray_BASE(new))) {
        new = (PyMicArrayObject *)(PyMicArray_BASE(new));
    }
    /*
     * Get the available memory through the buffer interface on
     * PyArray_BASE(new) or if that fails from the current new
     */
    if (PyMicArray_BASE(new) && PyObject_AsReadBuffer(PyMicArray_BASE(new),
                                           (const void **)&buf,
                                           &buf_len) >= 0) {
        offset = PyMicArray_BYTES(self) - buf;
        numbytes = buf_len + offset;
    }
    else {
        PyErr_Clear();
        offset_bounds_from_strides(PyMicArray_ITEMSIZE(new), PyMicArray_NDIM(new),
                                   PyMicArray_DIMS(new), PyMicArray_STRIDES(new),
                                   &lower_offset, &upper_offset);

        offset = PyMicArray_BYTES(self) - (PyMicArray_BYTES(new) + lower_offset);
        numbytes = upper_offset - lower_offset;
    }

    /* numbytes == 0 is special here, but the 0-size array case always works */
    if (!PyArray_CheckStrides(PyMicArray_ITEMSIZE(self), PyMicArray_NDIM(self),
                              numbytes, offset,
                              PyMicArray_DIMS(self), newstrides.ptr)) {
        PyErr_SetString(PyExc_ValueError, "strides is not "\
                        "compatible with available memory");
        goto fail;
    }
    memcpy(PyMicArray_STRIDES(self), newstrides.ptr, sizeof(npy_intp)*newstrides.len);
    PyMicArray_UpdateFlags(self, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS |
                              NPY_ARRAY_ALIGNED);
    PyDimMem_FREE(newstrides.ptr);
    return 0;

 fail:
    PyDimMem_FREE(newstrides.ptr);
    return -1;
}

static PyObject *
array_priority_get(PyMicArrayObject *self)
{
    if (PyArray_CheckExact(self)) {
        return PyFloat_FromDouble(NPY_PRIORITY);
    }
    else {
        return PyFloat_FromDouble(NPY_PRIORITY);
    }
}

static PyObject *
array_descr_get(PyMicArrayObject *self)
{
    Py_INCREF(PyMicArray_DESCR(self));
    return (PyObject *)PyMicArray_DESCR(self);
}

static PyObject *
array_itemsize_get(PyMicArrayObject *self)
{
    return PyInt_FromLong((long) PyMicArray_DESCR(self)->elsize);
}

static PyObject *
array_size_get(PyMicArrayObject *self)
{
    npy_intp size = PyMicArray_SIZE(self);
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) size);
#else
    if (size > NPY_MAX_LONG || size < NPY_MIN_LONG) {
        return PyLong_FromLongLong(size);
    }
    else {
        return PyInt_FromLong((long) size);
    }
#endif
}

static PyObject *
array_nbytes_get(PyMicArrayObject *self)
{
    npy_intp nbytes = PyMicArray_NBYTES(self);
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) nbytes);
#else
    if (nbytes > NPY_MAX_LONG || nbytes < NPY_MIN_LONG) {
        return PyLong_FromLongLong(nbytes);
    }
    else {
        return PyInt_FromLong((long) nbytes);
    }
#endif
}


/*
 * If the type is changed.
 * Also needing change: strides, itemsize
 *
 * Either itemsize is exactly the same or the array is single-segment
 * (contiguous or fortran) with compatibile dimensions The shape and strides
 * will be adjusted in that case as well.
 */
static int
array_descr_set(PyMicArrayObject *self, PyObject *arg)
{
    PyArray_Descr *newtype = NULL;
    npy_intp newdim;
    int i;
    char *msg = "new type not compatible with array.";
    PyObject *safe;
    static PyObject *checkfunc = NULL;


    if (arg == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array dtype");
        return -1;
    }

    if (!(PyArray_DescrConverter(arg, &newtype)) ||
        newtype == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "invalid data-type for array");
        return -1;
    }

    /* check that we are not reinterpreting memory containing Objects. */
    if (_may_have_objects(PyMicArray_DESCR(self)) || _may_have_objects(newtype)) {
        npy_cache_import("numpy.core._internal", "_view_is_safe", &checkfunc);
        if (checkfunc == NULL) {
            return -1;
        }

        safe = PyObject_CallFunction(checkfunc, "OO",
                                     PyMicArray_DESCR(self), newtype);
        if (safe == NULL) {
            Py_DECREF(newtype);
            return -1;
        }
        Py_DECREF(safe);
    }

    if (newtype->elsize == 0) {
        /* Allow a void view */
        if (newtype->type_num == NPY_VOID) {
            PyArray_DESCR_REPLACE(newtype);
            if (newtype == NULL) {
                return -1;
            }
            newtype->elsize = PyMicArray_DESCR(self)->elsize;
        }
        /* But no other flexible types */
        else {
            PyErr_SetString(PyExc_TypeError,
                    "data-type must not be 0-sized");
            Py_DECREF(newtype);
            return -1;
        }
    }


    if ((newtype->elsize != PyMicArray_DESCR(self)->elsize) &&
            (PyMicArray_NDIM(self) == 0 ||
             !PyMicArray_ISONESEGMENT(self) ||
             PyDataType_HASSUBARRAY(newtype))) {
        goto fail;
    }

    /* Deprecate not C contiguous and a dimension changes */
    if (newtype->elsize != PyMicArray_DESCR(self)->elsize &&
            !PyMicArray_IS_C_CONTIGUOUS(self)) {
        /* 11/27/2015 1.11.0 */
        if (DEPRECATE("Changing the shape of non-C contiguous array by\n"
                      "descriptor assignment is deprecated. To maintain\n"
                      "the Fortran contiguity of a multidimensional Fortran\n"
                      "array, use 'a.T.view(...).T' instead") < 0) {
            return -1;
        }
    }

    if (PyMicArray_IS_C_CONTIGUOUS(self)) {
        i = PyMicArray_NDIM(self) - 1;
    }
    else {
        i = 0;
    }
    if (newtype->elsize < PyMicArray_DESCR(self)->elsize) {
        /*
         * if it is compatible increase the size of the
         * dimension at end (or at the front for NPY_ARRAY_F_CONTIGUOUS)
         */
        if (PyMicArray_DESCR(self)->elsize % newtype->elsize != 0) {
            goto fail;
        }
        newdim = PyMicArray_DESCR(self)->elsize / newtype->elsize;
        PyMicArray_DIMS(self)[i] *= newdim;
        PyMicArray_STRIDES(self)[i] = newtype->elsize;
    }
    else if (newtype->elsize > PyMicArray_DESCR(self)->elsize) {
        /*
         * Determine if last (or first if NPY_ARRAY_F_CONTIGUOUS) dimension
         * is compatible
         */
        newdim = PyMicArray_DIMS(self)[i] * PyMicArray_DESCR(self)->elsize;
        if ((newdim % newtype->elsize) != 0) {
            goto fail;
        }
        PyMicArray_DIMS(self)[i] = newdim / newtype->elsize;
        PyMicArray_STRIDES(self)[i] = newtype->elsize;
    }

    /* fall through -- adjust type*/
    Py_DECREF(PyMicArray_DESCR(self));
    if (PyDataType_HASSUBARRAY(newtype)) {
        /*
         * create new array object from data and update
         * dimensions, strides and descr from it
         * TODO(superbo):re implement
         */
        PyMicArrayObject *temp;
        /*
         * We would decref newtype here.
         * temp will steal a reference to it
         */
        temp = (PyMicArrayObject *)
            PyMicArray_NewFromDescr(PyMicArray_DEVICE(self),
                                 &PyMicArray_Type, newtype, PyMicArray_NDIM(self),
                                 PyMicArray_DIMS(self), PyMicArray_STRIDES(self),
                                 PyMicArray_DATA(self), PyMicArray_FLAGS(self), NULL);
        if (temp == NULL) {
            return -1;
        }
        PyDimMem_FREE(PyMicArray_DIMS(self));
        ((PyMicArrayObject *)self)->dimensions = PyMicArray_DIMS(temp);
        ((PyMicArrayObject *)self)->nd = PyMicArray_NDIM(temp);
        ((PyMicArrayObject *)self)->strides = PyMicArray_STRIDES(temp);
        newtype = PyMicArray_DESCR(temp);
        Py_INCREF(PyMicArray_DESCR(temp));
        /* Fool deallocator not to delete these*/
        ((PyMicArrayObject *)temp)->nd = 0;
        ((PyMicArrayObject *)temp)->dimensions = NULL;
        Py_DECREF(temp);
    }

    ((PyMicArrayObject *)self)->descr = newtype;
    PyMicArray_UpdateFlags(self, NPY_ARRAY_UPDATE_ALL);
    return 0;

 fail:
    PyErr_SetString(PyExc_ValueError, msg);
    Py_DECREF(newtype);
    return -1;
}

static PyObject *
array_base_get(PyMicArrayObject *self)
{
    if (PyMicArray_BASE(self) == NULL) {
        Py_RETURN_NONE;
    }
    else {
        Py_INCREF(PyMicArray_BASE(self));
        return PyMicArray_BASE(self);
    }
}

/*
 * Create a view of a complex array with an equivalent data-type
 * except it is real instead of complex.
 */
static PyMicArrayObject *
_get_part(PyMicArrayObject *self, int imag)
{
    int float_type_num;
    PyArray_Descr *type;
    PyMicArrayObject *ret;
    int offset;

    switch (PyMicArray_TYPE(self)) {
        case NPY_CFLOAT:
            float_type_num = NPY_FLOAT;
            break;
        case NPY_CDOUBLE:
            float_type_num = NPY_DOUBLE;
            break;
        case NPY_CLONGDOUBLE:
            float_type_num = NPY_LONGDOUBLE;
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                     "Cannot convert complex type number %d to float",
                     PyMicArray_TYPE(self));
            return NULL;

    }
    type = PyArray_DescrFromType(float_type_num);

    offset = (imag ? type->elsize : 0);

    if (!PyArray_ISNBO(PyMicArray_DESCR(self)->byteorder)) {
        PyArray_Descr *new;
        new = PyArray_DescrNew(type);
        new->byteorder = PyMicArray_DESCR(self)->byteorder;
        Py_DECREF(type);
        type = new;
    }
    ret = (PyMicArrayObject *)
        PyMicArray_NewFromDescr(PyMicArray_DEVICE(self),
                             Py_TYPE(self),
                             type,
                             PyMicArray_NDIM(self),
                             PyMicArray_DIMS(self),
                             PyMicArray_STRIDES(self),
                             PyMicArray_BYTES(self) + offset,
                             PyMicArray_FLAGS(self), (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyMicArray_SetBaseObject(ret, (PyObject *)self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    PyMicArray_CLEARFLAGS(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return ret;
}

/* For Object arrays, we need to get and set the
   real part of each element.
 */

static PyObject *
array_real_get(PyMicArrayObject *self)
{
    PyMicArrayObject *ret;

    if (PyMicArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 0);
        return (PyObject *)ret;
    }
    else {
        Py_INCREF(self);
        return (PyObject *)self;
    }
}


static int
array_real_set(PyMicArrayObject *self, PyObject *val)
{
    PyMicArrayObject *ret;
    PyMicArrayObject *new;
    int retcode, device;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array real part");
        return -1;
    }
    if (PyMicArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 0);
        if (ret == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(self);
        ret = self;
    }
    device = PyMicArray_DEVICE(self);
    new = (PyMicArrayObject *)PyMicArray_FromAny(device, val, NULL, 0, 0, 0, NULL);
    if (new == NULL) {
        Py_DECREF(ret);
        return -1;
    }
    retcode = PyMicArray_MoveInto(ret, new);
    Py_DECREF(ret);
    Py_DECREF(new);
    return retcode;
}

/* For Object arrays we need to get
   and set the imaginary part of
   each element
*/

static PyObject *
array_imag_get(PyMicArrayObject *self)
{
    PyMicArrayObject *ret;

    if (PyMicArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 1);
    }
    else {
        Py_INCREF(PyMicArray_DESCR(self));
        ret = (PyMicArrayObject *)PyMicArray_NewFromDescr(
                                                    PyMicArray_DEVICE(self),
                                                    Py_TYPE(self),
                                                    PyMicArray_DESCR(self),
                                                    PyMicArray_NDIM(self),
                                                    PyMicArray_DIMS(self),
                                                    NULL, NULL,
                                                    PyMicArray_ISFORTRAN(self),
                                                    (PyObject *)self);
        if (ret == NULL) {
            return NULL;
        }
        if (_zerofill(ret) < 0) {
            return NULL;
        }
        PyMicArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    return (PyObject *) ret;
}

static int
array_imag_set(PyMicArrayObject *self, PyObject *val)
{
    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete array imaginary part");
        return -1;
    }
    if (PyMicArray_ISCOMPLEX(self)) {
        PyMicArrayObject *ret;
        PyMicArrayObject *new;
        int retcode, device;

        ret = _get_part(self, 1);
        if (ret == NULL) {
            return -1;
        }
        device = PyMicArray_DEVICE(self);
        new = (PyMicArrayObject *)PyMicArray_FromAny(device, val, NULL, 0, 0, 0, NULL);
        if (new == NULL) {
            Py_DECREF(ret);
            return -1;
        }
        retcode = PyMicArray_MoveInto(ret, new);
        Py_DECREF(ret);
        Py_DECREF(new);
        return retcode;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "array does not have imaginary part to set");
        return -1;
    }
}

static PyObject *
array_transpose_get(PyMicArrayObject *self)
{
    return PyMicArray_Transpose(self, NULL);
}

/* If this is None, no function call is made
   --- default sub-class behavior
*/
static PyObject *
array_finalize_get(PyMicArrayObject *NPY_UNUSED(self))
{
    Py_RETURN_NONE;
}

NPY_NO_EXPORT PyGetSetDef array_getsetlist[] = {
    {"device",
        (getter)array_device_get,
        NULL,
        NULL, NULL},
    {"ndim",
        (getter)array_ndim_get,
        NULL,
        NULL, NULL},
    {"flags",
        (getter)array_flags_get,
        NULL,
        NULL, NULL},
    {"shape",
        (getter)array_shape_get,
        (setter)array_shape_set,
        NULL, NULL},
    {"strides",
        (getter)array_strides_get,
        (setter)array_strides_set,
        NULL, NULL},
    /*Not need anymore
    {"data",
        (getter)array_data_get,
        NULL,
        NULL, NULL},*/
    {"itemsize",
        (getter)array_itemsize_get,
        NULL,
        NULL, NULL},
    {"size",
        (getter)array_size_get,
        NULL,
        NULL, NULL},
    {"nbytes",
        (getter)array_nbytes_get,
        NULL,
        NULL, NULL},
    {"base",
        (getter)array_base_get,
        NULL,
        NULL, NULL},
    {"dtype",
        (getter)array_descr_get,
        (setter)array_descr_set,
        NULL, NULL},
    {"real",
        (getter)array_real_get,
        (setter)array_real_set,
        NULL, NULL},
    {"imag",
        (getter)array_imag_get,
        (setter)array_imag_set,
        NULL, NULL},
    /* TODO: keep or delete ?
    {"flat",
        (getter)array_flat_get,
        (setter)array_flat_set,
        NULL, NULL},*/
    /* TODO: keep or delete?
    {"ctypes",
        (getter)array_ctypes_get,
        NULL,
        NULL, NULL},*/
    {"T",
        (getter)array_transpose_get,
        NULL,
        NULL, NULL},
    /*TODO: keep or delete ?
    {"__array_interface__",
        (getter)array_interface_get,
        NULL,
        NULL, NULL},
    {"__array_struct__",
        (getter)array_struct_get,
        NULL,
        NULL, NULL},*/
    {"__array_priority__",
        (getter)array_priority_get,
        NULL,
        NULL, NULL},
    {"__array_finalize__",
        (getter)array_finalize_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};

/****************** end of attribute get and set routines *******************/
