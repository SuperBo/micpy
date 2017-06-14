#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/npy_3kcompat.h>

#include "templ_common.h"

#define _MICARRAYMODULE
#include "common.h"
#include "arrayobject.h"
#include "multiarraymodule.h"
#include "creators.h"
#include "convert.h"
#include "array_assign.h"
//#include "numpymemoryview.h"
//#include "lowlevel_strided_loops.h"
#include "methods.h"
#include "alloc.h"


/*
 * Change a sub-array field to the base descriptor
 * and update the dimensions and strides
 * appropriately.  Dimensions and strides are added
 * to the end.
 *
 * Strides are only added if given (because data is given).
 */
static int
_update_descr_and_dimensions(PyArray_Descr **des, npy_intp *newdims,
                             npy_intp *newstrides, int oldnd)
{
    PyArray_Descr *old;
    int newnd;
    int numnew;
    npy_intp *mydim;
    int i;
    int tuple;

    old = *des;
    *des = old->subarray->base;


    mydim = newdims + oldnd;
    tuple = PyTuple_Check(old->subarray->shape);
    if (tuple) {
        numnew = PyTuple_GET_SIZE(old->subarray->shape);
    }
    else {
        numnew = 1;
    }


    newnd = oldnd + numnew;
    if (newnd > NPY_MAXDIMS) {
        goto finish;
    }
    if (tuple) {
        for (i = 0; i < numnew; i++) {
            mydim[i] = (npy_intp) PyInt_AsLong(
                    PyTuple_GET_ITEM(old->subarray->shape, i));
        }
    }
    else {
        mydim[0] = (npy_intp) PyInt_AsLong(old->subarray->shape);
    }

    if (newstrides) {
        npy_intp tempsize;
        npy_intp *mystrides;

        mystrides = newstrides + oldnd;
        /* Make new strides -- alwasy C-contiguous */
        tempsize = (*des)->elsize;
        for (i = numnew - 1; i >= 0; i--) {
            mystrides[i] = tempsize;
            tempsize *= mydim[i] ? mydim[i] : 1;
        }
    }

 finish:
    Py_INCREF(*des);
    Py_DECREF(old);
    return newnd;
}

/*
 * Generic new array creation routine.
 * Internal variant with calloc argument for PyArray_Zeros.
 *
 * steals a reference to descr. On failure or descr->subarray, descr will
 * be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_NewFromDescr_int(int device, PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                         npy_intp *dims, npy_intp *strides, void *data,
                         int flags, PyObject *obj, int zeroed,
                         int allow_emptystring)
{
    PyMicArrayObject *fa;
    int i, is_empty;
    npy_intp nbytes;

    if (descr->subarray) {
        PyObject *ret;
        npy_intp newdims[2*NPY_MAXDIMS];
        npy_intp *newstrides = NULL;
        memcpy(newdims, dims, nd*sizeof(npy_intp));
        if (strides) {
            newstrides = newdims + NPY_MAXDIMS;
            memcpy(newstrides, strides, nd*sizeof(npy_intp));
        }
        nd =_update_descr_and_dimensions(&descr, newdims,
                                         newstrides, nd);
        ret = PyMicArray_NewFromDescr_int(device, subtype, descr, nd, newdims,
                                          newstrides,
                                          data, flags, obj, zeroed,
                                          allow_emptystring);
        return ret;
    }

    /* Check device number */
    if (!(device >= 0  && device < N_DEVICES)) {
        PyErr_Format(PyExc_ValueError,
                        "device number must be within [0, %d]",
                        N_DEVICES);
        Py_DECREF(descr);
        return NULL;
    }

    if ((unsigned int)nd > (unsigned int)NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                     "number of dimensions must be within [0, %d]",
                     NPY_MAXDIMS);
        Py_DECREF(descr);
        return NULL;
    }

    /* Check datatype element size */
    nbytes = descr->elsize;
    if (nbytes == 0) {
        if (!PyDataType_ISFLEXIBLE(descr)) {
            PyErr_SetString(PyExc_TypeError, "Empty data-type");
            Py_DECREF(descr);
            return NULL;
        } else if (PyDataType_ISSTRING(descr) && !allow_emptystring) {
            PyArray_DESCR_REPLACE(descr);
            if (descr == NULL) {
                return NULL;
            }
            if (descr->type_num == NPY_STRING) {
                nbytes = descr->elsize = 1;
            }
            else {
                nbytes = descr->elsize = sizeof(npy_ucs4);
            }
        }
    }

    /* Check dimensions and multiply them to nbytes */
    is_empty = 0;
    for (i = 0; i < nd; i++) {
        npy_intp dim = dims[i];

        if (dim == 0) {
            /*
             * Compare to PyArray_OverflowMultiplyList that
             * returns 0 in this case.
             */
            is_empty = 1;
            continue;
        }

        if (dim < 0) {
            PyErr_SetString(PyExc_ValueError,
                "negative dimensions are not allowed");
            Py_DECREF(descr);
            return NULL;
        }

        /*
         * Care needs to be taken to avoid integer overflow when
         * multiplying the dimensions together to get the total size of the
         * array.
         */
        if (npy_mul_with_overflow_intp(&nbytes, nbytes, dim)) {
            PyErr_SetString(PyExc_ValueError,
                "array is too big; `arr.size * arr.dtype.itemsize` "
                "is larger than the maximum possible size.");
            Py_DECREF(descr);
            return NULL;
        }
    }

    fa = (PyMicArrayObject *) subtype->tp_alloc(subtype, 0);
    if (fa == NULL) {
        Py_DECREF(descr);
        return NULL;
    }
    fa->device = device;
    fa->nd = nd;
    fa->dimensions = NULL;
    fa->data = NULL;
    if (data == NULL) {
        fa->flags = NPY_ARRAY_DEFAULT;
        if (flags) {
            fa->flags |= NPY_ARRAY_F_CONTIGUOUS;
            if (nd > 1) {
                fa->flags &= ~NPY_ARRAY_C_CONTIGUOUS;
            }
            flags = NPY_ARRAY_F_CONTIGUOUS;
        }
    }
    else {
        fa->flags = (flags & ~NPY_ARRAY_UPDATEIFCOPY);
    }
    fa->descr = descr;
    fa->base = (PyObject *)NULL;
    fa->weakreflist = (PyObject *)NULL;

    if (nd > 0) {
        fa->dimensions = mpy_alloc_cache_dim(2 * nd);
        if (fa->dimensions == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        fa->strides = fa->dimensions + nd;
        memcpy(fa->dimensions, dims, sizeof(npy_intp)*nd);
        if (strides == NULL) {  /* fill it in */
            _array_fill_strides(fa->strides, dims, nd, descr->elsize,
                                flags, &(fa->flags));
        }
        else {
            /*
             * we allow strides even when we create
             * the memory, but be careful with this...
             */
            memcpy(fa->strides, strides, sizeof(npy_intp)*nd);
        }
    }
    else {
        fa->dimensions = fa->strides = NULL;
        fa->flags |= NPY_ARRAY_F_CONTIGUOUS;
    }

    if (data == NULL) {
        /*
         * Allocate something even for zero-space arrays
         * e.g. shape=(0,) -- otherwise buffer exposure
         * (a.data) doesn't work as it should.
         * Could probably just allocate a few bytes here. -- Chuck
         */
        if (is_empty) {
            nbytes = descr->elsize;
        }
        /*
         * It is bad to have uninitialized OBJECT pointers
         * which could also be sub-fields of a VOID array
         */
        if (zeroed || PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            data = mpy_alloc_cache_zero(nbytes, device);
        }
        else {
            data = mpy_alloc_cache(nbytes, device);
        }
        if (data == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        fa->flags |= NPY_ARRAY_OWNDATA;

    }
    else {
        /*
         * If data is passed in, this object won't own it by default.
         * Caller must arrange for this to be reset if truly desired
         */
        fa->flags &= ~NPY_ARRAY_OWNDATA;
    }
    fa->data = data;

    /*
     * always update the flags to get the right CONTIGUOUS, ALIGN properties
     * not owned data and input strides may not be aligned and on some
     * platforms (debian sparc) malloc does not provide enough alignment for
     * long double types
     */
    PyArray_UpdateFlags((PyArrayObject *)fa, NPY_ARRAY_UPDATE_ALL);

    /*
     * call the __array_finalize__
     * method if a subtype.
     * If obj is NULL, then call method with Py_None
     */
    if ((subtype != &PyMicArray_Type)) {
        PyObject *res, *func, *args;

        func = PyObject_GetAttr((PyObject *)fa, mpy_ma_str_array_finalize);
        if (func && func != Py_None) {
            if (NpyCapsule_Check(func)) {
                /* A C-function is stored here */
                PyMicArray_FinalizeFunc *cfunc;
                cfunc = NpyCapsule_AsVoidPtr(func);
                Py_DECREF(func);
                if (cfunc((PyMicArrayObject *)fa, obj) < 0) {
                    goto fail;
                }
            }
            else {
                args = PyTuple_New(1);
                if (obj == NULL) {
                    obj=Py_None;
                }
                Py_INCREF(obj);
                PyTuple_SET_ITEM(args, 0, obj);
                res = PyObject_Call(func, args, NULL);
                Py_DECREF(args);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
        else Py_XDECREF(func);
    }
    return (PyObject *)fa;

 fail:
    Py_DECREF(fa);
    return NULL;
}


/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr. On failure or when dtype->subarray is
 * true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_NewFromDescr(int device, PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                     npy_intp *dims, npy_intp *strides, void *data,
                     int flags, PyObject *obj)
{
    return PyMicArray_NewFromDescr_int(device, subtype, descr, nd,
                                    dims, strides, data,
                                    flags, obj, 0, 0);
}

/*NUMPY_API
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order and data type changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * dtype     - If not NULL, overrides the data type of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * dtype->subarray is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_NewLikeArray(int device, PyArrayObject *prototype, NPY_ORDER order,
                     PyArray_Descr *dtype, int subok)
{
    PyObject *ret = NULL;
    int ndim = PyArray_NDIM(prototype);

    /* If no override data type, use the one from the prototype */
    if (dtype == NULL) {
        dtype = PyArray_DESCR(prototype);
        Py_INCREF(dtype);
    }

    /* Handle ANYORDER and simple KEEPORDER cases */
    switch (order) {
        case NPY_ANYORDER:
            order = PyArray_ISFORTRAN(prototype) ?
                                    NPY_FORTRANORDER : NPY_CORDER;
            break;
        case NPY_KEEPORDER:
            if (PyArray_IS_C_CONTIGUOUS(prototype) || ndim <= 1) {
                order = NPY_CORDER;
                break;
            }
            else if (PyArray_IS_F_CONTIGUOUS(prototype)) {
                order = NPY_FORTRANORDER;
                break;
            }
            break;
        default:
            break;
    }

    /* If it's not KEEPORDER, this is simple */
    if (order != NPY_KEEPORDER) {
        ret = PyMicArray_NewFromDescr(device,
                                      subok ? Py_TYPE(prototype) : &PyMicArray_Type,
                                      dtype,
                                      ndim,
                                      PyArray_DIMS(prototype),
                                      NULL,
                                      NULL,
                                      order,
                                      subok ? (PyObject *)prototype : NULL);
    }
    /* KEEPORDER needs some analysis of the strides */
    else {
        npy_intp strides[NPY_MAXDIMS], stride;
        npy_intp *shape = PyArray_DIMS(prototype);
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        int idim;

        PyArray_CreateSortedStridePerm(PyArray_NDIM(prototype),
                                        PyArray_STRIDES(prototype),
                                        strideperm);

        /* Build the new strides */
        stride = dtype->elsize;
        for (idim = ndim-1; idim >= 0; --idim) {
            npy_intp i_perm = strideperm[idim].perm;
            strides[i_perm] = stride;
            stride *= shape[i_perm];
        }

        /* Finally, allocate the array */
        ret = PyMicArray_NewFromDescr(device,
                                      subok ? Py_TYPE(prototype) : &PyMicArray_Type,
                                      dtype,
                                      ndim,
                                      shape,
                                      strides,
                                      NULL,
                                      0,
                                      subok ? (PyObject *)prototype : NULL);
    }

    return ret;
}

/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_New(int device, PyTypeObject *subtype, int nd, npy_intp *dims, int type_num,
            npy_intp *strides, void *data, int itemsize, int flags,
            PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (descr->elsize == 0) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        descr->elsize = itemsize;
    }
    new = PyMicArray_NewFromDescr(device, subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}

/*NUMPY_API
 * Does not check for NPY_ARRAY_ENSURECOPY and NPY_ARRAY_NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
NPY_NO_EXPORT PyObject *
PyMicArray_FromAny(int device, PyObject *op, PyArray_Descr *newtype, int min_depth,
                   int max_depth, int flags, PyObject *context)
{
    /*
     * This is the main code to make a MicPy array from a Python
     * Object.  It is called from many different places.
     */

    PyArrayObject *arr;
    PyObject *ret;

    if (PyArray_Check(op) || PyMicArray_Check(op)) {
        return PyMicArray_FromArray((PyArrayObject *)op, newtype, device, flags);
    }

    arr = (PyArrayObject *) PyArray_FromAny(op, newtype, min_depth, max_depth,
                                            flags, context);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyMicArray_FromArray(arr, newtype, device, flags);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * steals reference to newtype --- acc. NULL
 * arr can be PyMicArray or PyArray
 */
NPY_NO_EXPORT PyObject *
PyMicArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int device, int flags)
{
    PyMicArrayObject *ret = NULL;
    int itemsize;
    int copy = 0;
    int arrflags;
    PyArray_Descr *oldtype;
    NPY_CASTING casting = NPY_SAFE_CASTING;

    oldtype = PyArray_DESCR(arr);
    if (newtype == NULL) {
        /*
         * Check if object is of array with Null newtype.
         * If so return it directly instead of checking for casting.
         */
        if (flags == 0) {
            Py_INCREF(arr);
            return (PyObject *)arr;
        }
        newtype = oldtype;
        Py_INCREF(oldtype);
    }
    itemsize = newtype->elsize;
    if (itemsize == 0) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
        itemsize = newtype->elsize;
    }

    /* If the casting if forced, use the 'unsafe' casting rule */
    if (flags & NPY_ARRAY_FORCECAST) {
        casting = NPY_UNSAFE_CASTING;
    }

    /* Raise an error if the casting rule isn't followed */
    if (!PyArray_CanCastArrayTo(arr, newtype, casting)) {
        PyObject *errmsg;
        PyArray_Descr *arr_descr = NULL;
        PyObject *arr_descr_repr = NULL;
        PyObject *newtype_repr = NULL;

        PyErr_Clear();
        errmsg = PyUString_FromString("Cannot cast array data from ");
        arr_descr = PyArray_DESCR(arr);
        if (arr_descr == NULL) {
            Py_DECREF(newtype);
            Py_DECREF(errmsg);
            return NULL;
        }
        arr_descr_repr = PyObject_Repr((PyObject *)arr_descr);
        if (arr_descr_repr == NULL) {
            Py_DECREF(newtype);
            Py_DECREF(errmsg);
            return NULL;
        }
        PyUString_ConcatAndDel(&errmsg, arr_descr_repr);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        newtype_repr = PyObject_Repr((PyObject *)newtype);
        if (newtype_repr == NULL) {
            Py_DECREF(newtype);
            Py_DECREF(errmsg);
            return NULL;
        }
        PyUString_ConcatAndDel(&errmsg, newtype_repr);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);

        Py_DECREF(newtype);
        return NULL;
    }

    arrflags = PyArray_FLAGS(arr);
           /* If a guaranteed copy was requested */
    copy = !PyMicArray_Check(arr) ||
           (PyMicArray_DEVICE(arr) != device) ||
           (flags & NPY_ARRAY_ENSURECOPY) ||
           /* If C contiguous was requested, and arr is not */
           ((flags & NPY_ARRAY_C_CONTIGUOUS) &&
                   (!(arrflags & NPY_ARRAY_C_CONTIGUOUS))) ||
           /* If an aligned array was requested, and arr is not */
           ((flags & NPY_ARRAY_ALIGNED) &&
                   (!(arrflags & NPY_ARRAY_ALIGNED))) ||
           /* If a Fortran contiguous array was requested, and arr is not */
           ((flags & NPY_ARRAY_F_CONTIGUOUS) &&
                   (!(arrflags & NPY_ARRAY_F_CONTIGUOUS))) ||
           /* If a writeable array was requested, and arr is not */
           ((flags & NPY_ARRAY_WRITEABLE) &&
                   (!(arrflags & NPY_ARRAY_WRITEABLE))) ||
           !PyArray_EquivTypes(oldtype, newtype);

    if (copy) {
        NPY_ORDER order = NPY_KEEPORDER;
        int subok = 1;
        int copy_ret = -1;

        /* Set the order for the copy being made based on the flags */
        if (flags & NPY_ARRAY_F_CONTIGUOUS) {
            order = NPY_FORTRANORDER;
        }
        else if (flags & NPY_ARRAY_C_CONTIGUOUS) {
            order = NPY_CORDER;
        }

        if ((flags & NPY_ARRAY_ENSUREARRAY)) {
            subok = 0;
        }

        ret = (PyMicArrayObject *)PyMicArray_NewLikeArray(device, arr, order,
                                                    newtype, subok);

        if (ret == NULL) {
            return NULL;
        }

        if (PyArray_CheckExact(arr)) {
            copy_ret = PyMicArray_CopyIntoFromHost(ret, arr);
        }
        else if (PyMicArray_Check(arr)) {
            copy_ret = PyMicArray_CopyInto(ret, (PyMicArrayObject *)arr);
        }

        if (copy_ret < 0) {
            Py_DECREF(ret);
            return NULL;
        }

        if (flags & NPY_ARRAY_UPDATEIFCOPY)  {
            Py_INCREF(arr);
            if (PyArray_SetUpdateIfCopyBase((PyArrayObject *)ret, arr) < 0) {
                Py_DECREF(ret);
                return NULL;
            }
        }
    }
    /*
     * If no copy then take an appropriate view if necessary, or
     * just return a reference to ret itself.
     */
    else {
        int needview = ((flags & NPY_ARRAY_ENSUREARRAY) &&
                        !PyMicArray_CheckExact(arr));
        PyMicArrayObject *mic_arr = (PyMicArrayObject *) arr;

        Py_DECREF(newtype);
        if (needview) {
            PyArray_Descr *dtype = PyMicArray_DESCR(mic_arr);
            PyTypeObject *subtype = NULL;

            if (flags & NPY_ARRAY_ENSUREARRAY) {
                subtype = &PyMicArray_Type;
            }

            Py_INCREF(dtype);
            ret = (PyMicArrayObject *)PyMicArray_View(mic_arr, NULL, subtype);
            if (ret == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(mic_arr);
            ret = mic_arr;
        }
    }

    return (PyObject *)ret;
}

/*NUMPY_API */
NPY_NO_EXPORT PyObject *
PyMicArray_FromStructInterface(PyObject *input)
{
    //TODO: implement
    return NULL;
}

#define PyIntOrLong_Check(obj) (PyInt_Check(obj) || PyLong_Check(obj))

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyMicArray_FromInterface(PyObject *origin)
{
    //TODO: implement
    return NULL;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyMicArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
{
    //TODO(superbo): implement this
    return NULL;
}

/* TODO: Put the order parameter in PyArray_CopyAnyInto and remove this */
NPY_NO_EXPORT int
PyMicArray_CopyAsFlat(PyMicArrayObject *dst, PyMicArrayObject *src, NPY_ORDER order)
{
    //TODO(superbo): implement this
    return -1;
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 *
 * TODO: For NumPy 2.0, this could accept an order parameter which
 *       only allows NPY_CORDER and NPY_FORDER.  Could also rename
 *       this to CopyAsFlat to make the name more intuitive.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyMicArray_CopyAnyInto(PyMicArrayObject *dst, PyMicArrayObject *src)
{
    return PyMicArray_CopyAsFlat(dst, src, NPY_CORDER);
}

/*
 * Copy an Device array into another device array.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_CopyInto(PyMicArrayObject *dst, PyMicArrayObject *src)
{
    return PyMicArray_AssignArray(dst, src, NULL, NPY_UNSAFE_CASTING);
}

/*
 * Copy an host array into another device array.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_CopyIntoFromHost(PyMicArrayObject *dst, PyArrayObject *src)
{
    return PyMicArray_AssignArrayFromHost(dst, src, NPY_UNSAFE_CASTING);
}

/*
 * Copy an device array into another host array.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_CopyIntoHost(PyArrayObject *dst, PyMicArrayObject *src)
{
    return PyArray_AssignArrayFromDevice(dst, src, NPY_UNSAFE_CASTING);
}

/*NUMPY_API
 * Move the memory of one array into another, allowing for overlapping data.
 *
 * Returns 0 on success, negative on failure.
 */
NPY_NO_EXPORT int
PyMicArray_MoveInto(PyMicArrayObject *dst, PyMicArrayObject *src)
{
    return PyMicArray_AssignArray(dst, src, NULL, NPY_UNSAFE_CASTING);
}


/*NUMPY_API
 * Zeros
 *
 * steals a reference to type. On failure or when dtype->subarray is
 * true, dtype will be decrefed.
 * accepts NULL type
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Zeros(int device, int nd, npy_intp *dims, PyArray_Descr *type, int is_f_order)
{
    PyMicArrayObject *ret;

    if (!type) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    Py_INCREF(type);

    ret = (PyMicArrayObject *)PyMicArray_NewFromDescr_int(device,
                                                    &PyMicArray_Type,
                                                    type,
                                                    nd, dims,
                                                    NULL, NULL,
                                                    is_f_order, NULL, 1, 0);

    Py_DECREF(type);
    return (PyObject *)ret;
}

/*NUMPY_API
 * Empty
 *
 * accepts NULL type
 * steals referenct to type
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Empty(int device, int nd, npy_intp *dims, PyArray_Descr *type, int is_f_order)
{
    PyMicArrayObject *ret;

    if (!type) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    /*
     * PyMicArray_NewFromDescr steals a ref,
     * but we need to look at type later.
     * */
    Py_INCREF(type);

    ret = (PyMicArrayObject *)PyMicArray_NewFromDescr(device, &PyMicArray_Type,
                                                type, nd, dims,
                                                NULL, NULL,
                                                is_f_order, NULL);

    Py_DECREF(type);
    return (PyObject *)ret;
}

/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure. In case of failure, set a PyExc_Overflow
 * exception
 */
static int _safe_ceil_to_intp(double value, npy_intp* ret)
{
    double ivalue;

    ivalue = npy_ceil(value);
    if (ivalue < NPY_MIN_INTP || ivalue > NPY_MAX_INTP) {
        return -1;
    }

    *ret = (npy_intp)ivalue;
    return 0;
}


/*NUMPY_API
  Arange,
*/
NPY_NO_EXPORT PyObject *
PyMicArray_Arange(double start, double stop, double step, int type_num)
{
    //TODO: implement
    return NULL;
}

/*
 * the formula is len = (intp) ceil((start - stop) / step);
 */
static npy_intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step, PyObject **next, int cmplx)
{
    npy_intp len, tmp;
    PyObject *val;
    double value;

    *next = PyNumber_Subtract(stop, start);
    if (!(*next)) {
        if (PyTuple_Check(stop)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "\
                            "instead of a tuple.");
        }
        return -1;
    }
    val = PyNumber_TrueDivide(*next, step);
    Py_DECREF(*next);
    *next = NULL;
    if (!val) {
        return -1;
    }
    if (cmplx && PyComplex_Check(val)) {
        value = PyComplex_RealAsDouble(val);
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            Py_DECREF(val);
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        value = PyComplex_ImagAsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &tmp)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        len = PyArray_MIN(len, tmp);
    }
    else {
        value = PyFloat_AsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
    }
    if (len > 0) {
        *next = PyNumber_Add(start, step);
        if (!*next) {
            return -1;
        }
    }
    return len;
}

#undef FROM_BUFFER_SIZE


/*
 * This is the main array creation routine.
 *
 * Flags argument has multiple related meanings
 * depending on data and strides:
 *
 * If data is given, then flags is flags associated with data.
 * If strides is not given, then a contiguous strides array will be created
 * and the NPY_ARRAY_C_CONTIGUOUS bit will be set.  If the flags argument
 * has the NPY_ARRAY_F_CONTIGUOUS bit set, then a FORTRAN-style strides array will be
 * created (and of course the NPY_ARRAY_F_CONTIGUOUS flag bit will be set).
 *
 * If data is not given but created here, then flags will be NPY_ARRAY_DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 *
 * Dimensions and itemsize must have been checked for validity.
 */

NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
#if NPY_RELAXED_STRIDES_CHECKING
    npy_bool not_cf_contig = 0;
    npy_bool nod = 0; /* A dim != 1 was found */

    /* Check if new array is both F- and C-contiguous */
    for (i = 0; i < nd; i++) {
        if (dims[i] != 1) {
            if (nod) {
                not_cf_contig = 1;
                break;
            }
            nod = 1;
        }
    }
#endif /* NPY_RELAXED_STRIDES_CHECKING */

    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS)) ==
                                            NPY_ARRAY_F_CONTIGUOUS) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
#if NPY_RELAXED_STRIDES_CHECKING
            else {
                not_cf_contig = 0;
            }
#endif /* NPY_RELAXED_STRIDES_CHECKING */
        }
#if NPY_RELAXED_STRIDES_CHECKING
        if (not_cf_contig) {
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if ((nd > 1) && ((strides[0] != strides[nd-1]) || (dims[nd-1] > 1))) {
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
            *objflags = ((*objflags)|NPY_ARRAY_F_CONTIGUOUS) &
                                            ~NPY_ARRAY_C_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS);
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
#if NPY_RELAXED_STRIDES_CHECKING
            else {
                not_cf_contig = 0;
            }
#endif /* NPY_RELAXED_STRIDES_CHECKING */
        }
#if NPY_RELAXED_STRIDES_CHECKING
        if (not_cf_contig) {
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if ((nd > 1) && ((strides[0] != strides[nd-1]) || (dims[0] > 1))) {
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
            *objflags = ((*objflags)|NPY_ARRAY_C_CONTIGUOUS) &
                                            ~NPY_ARRAY_F_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    return;
}


/*
 * Calls arr_of_subclass.__array_wrap__(towrap), in order to make 'towrap'
 * have the same ndarray subclass as 'arr_of_subclass'.
 */
NPY_NO_EXPORT PyMicArrayObject *
PyMicArray_SubclassWrap(PyMicArrayObject *arr_of_subclass, PyMicArrayObject *towrap)
{
    PyObject *wrapped = PyObject_CallMethod((PyObject *)arr_of_subclass,
                                        "__array_wrap__", "O", towrap);
    if (wrapped == NULL) {
        return NULL;
    }
    if (!PyMicArray_Check(wrapped)) {
        PyErr_SetString(PyExc_RuntimeError,
                "micpy.ndarray subclass __array_wrap__ method returned an "
                "object which was not an instance of an micpy.ndarray subclass");
        Py_DECREF(wrapped);
        return NULL;
    }

    return (PyMicArrayObject *)wrapped;
}
