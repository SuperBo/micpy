/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/

/* $Id: multiarraymodule.c,v 1.36 2005/09/14 00:14:00 teoliphant Exp $ */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/npy_common.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <numpy/npy_math.h>
#include <npy_config.h>

#define _MICARRAYMODULE
/* Internal APIs */
#include "arrayobject.h"
//#include "calculation.h"
#include "number.h"
//#include "numpymemoryview.h"
#include "mpyndarraytypes.h"
#include "arraytypes.h"
#include "array_assign.h"
#include "conversion_utils.h"
#include "methods.h"
#include "creators.h"
#include "convert.h"
#include "common.h"
#include "multiarraymodule.h"
#include "multiarray_api_creator.h"
#include "shape.h"
#include "scalar.h"
#include "nditer.h"
#include "cblasfuncs.h"
#include "mpymem_overlap.h"
#include "convert_datatype.h"

static int num_devices;
static int current_device;

NPY_NO_EXPORT int PyMicArray_GetCurrentDevice(void){
    return current_device;
}

NPY_NO_EXPORT int PyMicArray_SetCurrentDevice(int device_id){
    if (device_id >= 0 && device_id < num_devices) {
        current_device = device_id;
        return 0;
    }
    return -1;
}

NPY_NO_EXPORT int PyMicArray_GetNumDevices(void){
    return num_devices;
}

static PyObject *
get_current_device(PyObject *NPY_UNUSED(ignored), PyObject *args){
    return (PyObject *) PyInt_FromLong(current_device);
}

static PyObject *
set_current_device(PyObject *NPY_UNUSED(ignored), PyObject *device_id)
{
    int device = -1;

    if (!PyMicArray_DeviceConverter(device_id, &device)) {
        return NULL;
    }

    current_device = device;
    Py_RETURN_NONE;
}

static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the npy_byte to test */
    char byteorder;
    int elsize;

    elsize = PyArray_DESCR(arr)->elsize;
    byteorder = PyArray_DESCR(arr)->byteorder;
    ptr = PyArray_DATA(arr);
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          PyArray_ISNBO(NPY_LITTLE)))) {
        ptr += elsize - 1;
    }
    return ((*ptr & bitmask) != 0);
}


/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 * Assume that all arrays are on one device
 */
static PyMicArrayObject *
new_array_for_sum(PyMicArrayObject *ap1, PyMicArrayObject *ap2, PyMicArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyMicArrayObject **result)
{
    PyMicArrayObject *out_buf;
    PyTypeObject *subtype;
    double prior1, prior2;
    int device = PyMicArray_DEVICE(ap1);
    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    if (Py_TYPE(ap2) != Py_TYPE(ap1)) {
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
    }
    else {
        prior1 = prior2 = 0.0;
        subtype = Py_TYPE(ap1);
    }
    if (out) {
        int d;

        /* verify that out is usable */
        if (Py_TYPE(out) != subtype ||
            PyMicArray_NDIM(out) != nd ||
            PyMicArray_TYPE(out) != typenum ||
            !PyMicArray_ISCARRAY(out)) {
            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable "
                "(must have the right type, nr dimensions, and be a C-Array)");
            return 0;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyMicArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                return 0;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = (PyMicArrayObject *)PyMicArray_NewLikeArray(device, (PyArrayObject *)out,
                                                            NPY_CORDER, NULL, 0);
            if (out_buf == NULL) {
                return NULL;
            }

            /* set copy-back */
            Py_INCREF(out);
            if (PyMicArray_SetUpdateIfCopyBase(out_buf, out) < 0) {
                Py_DECREF(out);
                Py_DECREF(out_buf);
                return NULL;
            }
        }
        else {
            Py_INCREF(out);
            out_buf = out;
        }

        if (result) {
            Py_INCREF(out);
            *result = out;
        }

        return out_buf;
    }

    out_buf = (PyMicArrayObject *)PyMicArray_New(device, subtype, nd, dimensions,
                                           typenum, NULL, NULL, 0, 0,
                                           (PyObject *)
                                           (prior2 > prior1 ? ap2 : ap1));

    if (out_buf != NULL && result) {
        Py_INCREF(out_buf);
        *result = out_buf;
    }

    return out_buf;
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyMicArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyMicArrayObject* out)
{
    PyMicArrayObject *ap1, *ap2, *out_buf = NULL, *result = NULL;
    MpyIter *it1, *it2;
    MpyIter_IterNextFunc *it1_next, *it2_next;
    char **it1_dataptr, **it2_dataptr;
    npy_uint32 iterflags;
    npy_intp i, j, l, itersize;
    int typenum, nd, axis, matchDim, device, usecache = 0;
    npy_intp is1, is2, os, elesize;
    char *op, *cache;
    npy_intp dimensions[NPY_MAXDIMS];
    PyMicArray_DotFunc *dot;
    PyMicArray_CopySwapNFunc *copyn;
    PyArray_Descr *typec = NULL;
    NPY_BEGIN_THREADS_DEF;

    device = get_common_device2(op1, op2);

    typenum = PyMicArray_ObjectType(op1, 0);
    typenum = PyMicArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot find a common data type.");
        return NULL;
    }

    ap1 = (PyMicArrayObject *)PyMicArray_FromAny(device, op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        return NULL;
    }
    Py_INCREF(typec);
    ap2 = (PyMicArrayObject *)PyMicArray_FromAny(device, op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }

    if (PyMicArray_NDIM(ap1) <= 2 && PyMicArray_NDIM(ap2) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out);
    }

    if (PyMicArray_NDIM(ap1) == 0 || PyMicArray_NDIM(ap2) == 0) {
        //TODO: error may occur here
        result = (PyMicArray_NDIM(ap1) == 0 ? ap1 : ap2);
        result = (PyMicArrayObject *)Py_TYPE(result)->tp_as_number->nb_multiply(
                                        (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)result;
    }
    l = PyMicArray_DIMS(ap1)[PyMicArray_NDIM(ap1) - 1];
    if (PyMicArray_NDIM(ap2) > 1) {
        matchDim = PyMicArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }
    if (PyMicArray_DIMS(ap2)[matchDim] != l) {
        dot_alignment_error(ap1, PyMicArray_NDIM(ap1) - 1, ap2, matchDim);
        goto fail;
    }
    nd = PyMicArray_NDIM(ap1) + PyMicArray_NDIM(ap2) - 2;
    if (nd > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < PyMicArray_NDIM(ap1) - 1; i++) {
        dimensions[j++] = PyMicArray_DIMS(ap1)[i];
    }
    for (i = 0; i < PyMicArray_NDIM(ap2) - 2; i++) {
        dimensions[j++] = PyMicArray_DIMS(ap2)[i];
    }
    if(PyMicArray_NDIM(ap2) > 1) {
        dimensions[j++] = PyMicArray_DIMS(ap2)[PyMicArray_NDIM(ap2)-1];
    }

    is1 = PyMicArray_STRIDES(ap1)[PyMicArray_NDIM(ap1)-1];
    is2 = PyMicArray_STRIDES(ap2)[matchDim];
    /* Choose which subtype to return */
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyMicArray_SIZE(ap1) == 0 && PyMicArray_SIZE(ap2) == 0) {
        target_memset(PyMicArray_DATA(out_buf), 0, PyMicArray_NBYTES(out_buf),
                      PyMicArray_DEVICE(out_buf));
    }

    dot = PyMicArray_GetArrFuncs(typenum)->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }

    /* Switch to use nditer */
    elesize = PyMicArray_DESCR(out_buf)->elsize;
    op = PyMicArray_DATA(out_buf);
    os = PyMicArray_DESCR(out_buf)->elsize;
    iterflags = NPY_ITER_READONLY |  NPY_ITER_MULTI_INDEX;
    axis = PyMicArray_NDIM(ap1)-1;
    it1 = MpyIter_New(ap1, iterflags, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (it1 == NULL) {
        goto fail;
    }
    if (MpyIter_RemoveAxis(it1, axis) != NPY_SUCCEED) {
        MpyIter_Deallocate(it1);
        goto fail;
    }
    it1_next = MpyIter_GetIterNext(it1, NULL);
    if (it1_next == NULL) {
        MpyIter_Deallocate(it1);
        goto fail;
    }
    it2 = MpyIter_New(ap2, iterflags, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (it2 == NULL) {
        MpyIter_Deallocate(it1);
        goto fail;
    }
    if (MpyIter_RemoveAxis(it2, matchDim) != NPY_SUCCEED) {
        MpyIter_Deallocate(it1);
        MpyIter_Deallocate(it2);
        goto fail;
    }
    it2_next = MpyIter_GetIterNext(it2, NULL);
    if (it2_next == NULL) {
        MpyIter_Deallocate(it1);
        MpyIter_Deallocate(it2);
        goto fail;
    }

    itersize = MpyIter_GetIterSize(it1);
    it1_dataptr = MpyIter_GetDataPtrArray(it1);
    it2_dataptr = MpyIter_GetDataPtrArray(it2);

    /* Use cache if itersize larger than 1 */
    /* TODO: can we usecache for ndim > 2 ? */
    if (nd == 2 && itersize > 1 && is2 != elesize) {
        copyn = PyMicArray_GetArrFuncs(typenum)->copyswapn;
        if (copyn != NULL) {
            usecache = 1;
            cache = (char *) omp_target_alloc(elesize*l, device);
        }
    }

    NPY_BEGIN_THREADS_DESCR(PyMicArray_DESCR(ap2));
    if (usecache) {
        char *op0 = op;
        npy_intp k = PyMicArray_DIM(out_buf, 1);
        do {
            copyn(cache, elesize, *it2_dataptr, is2, l, 0, device);
            do {
                dot(*it1_dataptr, is1, cache, elesize, op, l, device);
                op += os*k;
            } while (it1_next(it1));
            op0 += os;
            op = op0;
            MpyIter_Reset(it1, NULL);
        } while (it2_next(it2));
    }
    else {
        do {
            do {
                dot(*it1_dataptr, is1, *it2_dataptr, is2, op, l, device);
                op += os;
            } while (it2_next(it2));
            MpyIter_Reset(it2, NULL);
        } while (it1_next(it1));
    }
    NPY_END_THREADS_DESCR(PyMicArray_DESCR(ap2));
    MpyIter_Deallocate(it1);
    MpyIter_Deallocate(it2);
    if (usecache) {
        omp_target_free(cache, device);
        cache = NULL;
    }
    if (PyErr_Occurred()) {
        /* only for OBJECT arrays */
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copy-back into `result` */
    Py_DECREF(out_buf);

    return (PyObject *)result;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}

/*NUMPY_API
 * Copy and Transpose
 *
 * Could deprecate this function, as there isn't a speed benefit over
 * calling Transpose and then Copy.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_CopyAndTranspose(PyObject *op)
{
    //TODO: implement
    return NULL;
}


/*
 * Revert a one dimensional array in-place
 *
 * Return 0 on success, other value on failure
 */
static int
_pyarray_revert(PyArrayObject *ret)
{
    //TODO: implement
    return -1;
}

/*** END C-API FUNCTIONS **/

/*
 * NPY_RELAXED_STRIDES_CHECKING: If the strides logic is changed, the
 * order specific stride setting is not necessary.
 */
static NPY_STEALS_REF_TO_ARG(1) PyObject *
_prepend_ones(PyMicArrayObject *arr, int nd, int ndmin, NPY_ORDER order)
{
    npy_intp newdims[NPY_MAXDIMS];
    npy_intp newstrides[NPY_MAXDIMS];
    npy_intp newstride;
    int i, k, num;
    PyMicArrayObject *ret;
    PyArray_Descr *dtype;

    if (order == NPY_FORTRANORDER || PyMicArray_ISFORTRAN(arr) || PyMicArray_NDIM(arr) == 0) {
        newstride = PyMicArray_DESCR(arr)->elsize;
    }
    else {
        newstride = PyMicArray_STRIDES(arr)[0] * PyMicArray_DIMS(arr)[0];
    }

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = newstride;
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyMicArray_DIMS(arr)[k];
        newstrides[i] = PyMicArray_STRIDES(arr)[k];
    }
    dtype = PyMicArray_DESCR(arr);
    Py_INCREF(dtype);
    ret = (PyMicArrayObject *)PyMicArray_NewFromDescr(
                        PyMicArray_DEVICE(arr),
                        Py_TYPE(arr),
                        dtype, ndmin, newdims, newstrides,
                        PyMicArray_DATA(arr),
                        PyMicArray_FLAGS(arr),
                        (PyObject *)arr);
    if (ret == NULL) {
        Py_DECREF(arr);
        return NULL;
    }
    /* steals a reference to arr --- so don't increment here */
    if (PyMicArray_SetBaseObject(ret, (PyObject *)arr) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject *)ret;
}


#define STRIDING_OK(op, order) \
                ((order) == NPY_ANYORDER || \
                 (order) == NPY_KEEPORDER || \
                 ((order) == NPY_CORDER && PyMicArray_IS_C_CONTIGUOUS(op)) || \
                 ((order) == NPY_FORTRANORDER && PyMicArray_IS_F_CONTIGUOUS(op)))

static PyObject *
_array_fromobject(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws)
{
    PyObject *op;
    PyMicArrayObject *oparr = NULL, *ret = NULL;
    npy_bool subok = NPY_FALSE;
    npy_bool copy = NPY_TRUE;
    int ndmin = 0, nd;
    PyArray_Descr *type = NULL;
    PyArray_Descr *oldtype = NULL;
    NPY_ORDER order = NPY_KEEPORDER;
    int flags = 0;
    int device;

    static char *kwd[]= {"object", "dtype", "copy", "order", "subok",
                         "ndmin", "device", NULL};

    if (PyTuple_GET_SIZE(args) > 2) {
        PyErr_SetString(PyExc_ValueError,
                        "only 2 non-keyword arguments accepted");
        return NULL;
    }

    /* super-fast path for ndarray argument calls */
    if (PyTuple_GET_SIZE(args) == 0) {
        goto full_path;
    }
    op = PyTuple_GET_ITEM(args, 0);
    if (PyMicArray_CheckExact(op)) {
        PyObject * dtype_obj = Py_None;
        oparr = (PyMicArrayObject *)op;
        /* get dtype which can be positional */
        if (PyTuple_GET_SIZE(args) == 2) {
            dtype_obj = PyTuple_GET_ITEM(args, 1);
        }
        else if (kws) {
            dtype_obj = PyDict_GetItem(kws, mpy_ma_str_dtype);
            if (dtype_obj == NULL) {
                dtype_obj = Py_None;
            }
        }
        if (dtype_obj != Py_None) {
            goto full_path;
        }

        /* array(ndarray) */
        if (kws == NULL) {
            ret = (PyMicArrayObject *)PyMicArray_NewCopy(oparr, order);
            goto finish;
        }
        else {
            /* fast path for copy=False rest default (np.asarray) */
            PyObject * copy_obj, * order_obj, *ndmin_obj;
            copy_obj = PyDict_GetItem(kws, mpy_ma_str_copy);
            if (copy_obj != Py_False) {
                goto full_path;
            }
            copy = NPY_FALSE;

            /* order does not matter for contiguous 1d arrays */
            if (PyMicArray_NDIM((PyMicArrayObject*)op) > 1 ||
                !PyMicArray_IS_C_CONTIGUOUS((PyMicArrayObject*)op)) {
                order_obj = PyDict_GetItem(kws, mpy_ma_str_order);
                if (order_obj != Py_None && order_obj != NULL) {
                    goto full_path;
                }
            }

            ndmin_obj = PyDict_GetItem(kws, mpy_ma_str_ndmin);
            if (ndmin_obj) {
                ndmin = PyLong_AsLong(ndmin_obj);
                if (ndmin == -1 && PyErr_Occurred()) {
                    goto clean_type;
                }
                else if (ndmin > NPY_MAXDIMS) {
                    goto full_path;
                }
            }

            /* copy=False with default dtype, order and ndim */
            if (STRIDING_OK(oparr, order)) {
                ret = oparr;
                Py_INCREF(ret);
                goto finish;
            }
        }
    }

full_path:
    device = PyMicArray_GetCurrentDevice();

    if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O&O&O&O&iO&:array", kwd,
                &op,
                PyArray_DescrConverter2, &type,
                PyArray_BoolConverter, &copy,
                PyArray_OrderConverter, &order,
                PyArray_BoolConverter, &subok,
                &ndmin,
                PyMicArray_DeviceConverter, &device)) {
        goto clean_type;
    }

    if (ndmin > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        goto clean_type;
    }
    /* fast exit if simple call */
    if ((subok && PyMicArray_Check(op)) ||
        (!subok && PyMicArray_CheckExact(op))) {
        oparr = (PyMicArrayObject *)op;
        if (type == NULL) {
            if (!copy && STRIDING_OK(oparr, order)) {
                ret = oparr;
                Py_INCREF(ret);
                goto finish;
            }
            else {
                ret = (PyMicArrayObject *)PyMicArray_NewCopy(oparr, order);
                goto finish;
            }
        }
        /* One more chance */
        oldtype = PyMicArray_DESCR(oparr);
        if (PyArray_EquivTypes(oldtype, type)) {
            if (!copy && STRIDING_OK(oparr, order)) {
                Py_INCREF(op);
                ret = oparr;
                goto finish;
            }
            else {
                ret = (PyMicArrayObject *)PyMicArray_NewCopy(oparr, order);
                if (oldtype == type || ret == NULL) {
                    goto finish;
                }
                Py_INCREF(oldtype);
                Py_DECREF(PyMicArray_DESCR(ret));
                ((PyMicArrayObject *)ret)->descr = oldtype;
                goto finish;
            }
        }
    }

    if (copy) {
        flags = NPY_ARRAY_ENSURECOPY;
    }
    if (order == NPY_CORDER) {
        flags |= NPY_ARRAY_C_CONTIGUOUS;
    }
    else if ((order == NPY_FORTRANORDER)
                 /* order == NPY_ANYORDER && */
                 || (PyMicArray_Check(op) &&
                     PyMicArray_ISFORTRAN((PyMicArrayObject *)op))) {
        flags |= NPY_ARRAY_F_CONTIGUOUS;
    }
    if (!subok) {
        flags |= NPY_ARRAY_ENSUREARRAY;
    }

    flags |= NPY_ARRAY_FORCECAST;
    Py_XINCREF(type);
    ret = (PyMicArrayObject *)PyMicArray_CheckFromAny(device, op, type,
                                                0, 0, flags, NULL);

 finish:
    Py_XDECREF(type);
    if (ret == NULL) {
        return NULL;
    }

    nd = PyMicArray_NDIM(ret);
    if (nd >= ndmin) {
        return (PyObject *)ret;
    }
    /*
     * create a new array from the same data with ones in the shape
     * steals a reference to ret
     */
    return _prepend_ones(ret, nd, ndmin, order);

clean_type:
    Py_XDECREF(type);
    return NULL;
}



static PyObject *
array_copyto(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dst","src","casting","where",NULL};
    PyObject *wheremask_in = NULL;
    PyMicArrayObject *dst = NULL, *src = NULL, *wheremask = NULL;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O&|O&O:copyto", kwlist,
                &PyMicArray_Type, &dst,
                &PyMicArray_Converter, &src,
                &PyArray_CastingConverter, &casting,
                &wheremask_in)) {
        goto fail;
    }

    if (wheremask_in != NULL) {
        /* Get the boolean where mask */
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            goto fail;
        }
        wheremask = (PyMicArrayObject *)PyMicArray_FromAny(
                                        PyMicArray_DEVICE(dst),
                                        wheremask_in,
                                        dtype, 0, 0, 0, NULL);
        if (wheremask == NULL) {
            goto fail;
        }
    }

    if (PyMicArray_AssignArray(dst, src, wheremask, casting) < 0) {
        goto fail;
    }

    Py_XDECREF(src);
    Py_XDECREF(wheremask);

    Py_RETURN_NONE;

fail:
    Py_XDECREF(src);
    Py_XDECREF(wheremask);
    return NULL;
}

static PyObject *
array_empty(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape","dtype","order","device",NULL};
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order;
    int device = DEFAULT_DEVICE;
    PyMicArrayObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&O&", kwlist,
                &PyArray_IntpConverter, &shape,
                &PyArray_DescrConverter, &typecode,
                &PyArray_OrderConverter, &order,
                &PyMicArray_DeviceConverter, &device)) {
        goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    ret = (PyMicArrayObject *)PyMicArray_Empty(device, shape.len, shape.ptr,
                                            typecode, is_f_order);

    PyDimMem_FREE(shape.ptr);
    return (PyObject *)ret;

fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return NULL;
}

static PyObject *
array_empty_like(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"prototype","dtype","order","subok","device",NULL};
    PyArrayObject *prototype = NULL;
    PyArray_Descr *dtype = NULL;
    NPY_ORDER order = NPY_KEEPORDER;
    PyMicArrayObject *ret = NULL;
    int device = DEFAULT_DEVICE;
    int subok = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&iO&", kwlist,
                &PyMicArray_GeneralConverter, &prototype,
                &PyArray_DescrConverter2, &dtype,
                &PyArray_OrderConverter, &order,
                &subok,
                &PyMicArray_DeviceConverter, &device)) {
        goto fail;
    }

    /* steals the reference to dtype if it's not NULL */
    ret = (PyMicArrayObject *)PyMicArray_NewLikeArray(device, prototype,
                                            order, dtype, subok);
    Py_DECREF(prototype);

    return (PyObject *)ret;

fail:
    Py_XDECREF(prototype);
    Py_XDECREF(dtype);
    return NULL;
}

static PyObject *
array_zeros(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape","dtype","order","device",NULL};
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order = NPY_FALSE;
    PyMicArrayObject *ret = NULL;
    int device = DEFAULT_DEVICE;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&O&", kwlist,
                &PyArray_IntpConverter, &shape,
                &PyArray_DescrConverter, &typecode,
                &PyArray_OrderConverter, &order,
                &PyMicArray_DeviceConverter, &device)) {
        goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    ret = (PyMicArrayObject *)PyMicArray_Zeros(device, shape.len, shape.ptr,
                                        typecode, (int) is_f_order);

    PyDimMem_FREE(shape.ptr);
    return (PyObject *)ret;

fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return (PyObject *)ret;
}

static PyObject *
array_zeros_like(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"prototype","dtype","order","device",NULL};
    PyArrayObject *prototype = NULL;
    PyArray_Descr *dtype = NULL;
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order = NPY_FALSE;
    PyMicArrayObject *ret = NULL;
    int device = DEFAULT_DEVICE;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&O&", kwlist,
                &PyMicArray_GeneralConverter, &prototype,
                &PyArray_DescrConverter2, &dtype,
                &PyArray_OrderConverter, &order,
                &PyMicArray_DeviceConverter, &device)) {
        goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    /* If no override data type, use the one from the prototype */
    if (dtype == NULL) {
        dtype = PyArray_DESCR(prototype);
        Py_INCREF(dtype);
    }

    /* steals the reference to dtype if it's not NULL */
    ret = (PyMicArrayObject *)PyMicArray_Zeros(device, PyArray_NDIM(prototype),
                                                PyArray_DIMS(prototype),
                                                dtype, is_f_order);
    Py_DECREF(prototype);

    return (PyObject *)ret;

fail:
    Py_XDECREF(prototype);
    Py_XDECREF(dtype);
    return NULL;
}

static PyObject *
array_ones(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    PyMicArrayObject *ret = NULL;

    ret = (PyMicArrayObject *) array_empty(NULL, args, kwds);
    if (ret != NULL) {
        if (PyMicArray_AssignOne(ret, NULL) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "failed to set array to one");
            Py_DECREF(ret);
            return NULL;
        }
    }

    return (PyObject *)ret;
}

static PyObject *
array_ones_like(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    PyMicArrayObject *ret;

    ret = (PyMicArrayObject *) array_empty_like(NULL, args, kwds);
    if (ret != NULL) {
        if (PyMicArray_AssignOne(ret, NULL) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "failed to set array to one");
            Py_DECREF(ret);
            return NULL;
        }
    }

    return (PyObject *)ret;
}

static PyObject *
array_count_nonzero(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    //TODO: implement
    return NULL;
}

static PyObject *
array_matrixproduct(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject* kwds)
{
    PyObject *v, *a, *o = NULL;
    PyMicArrayObject *ret;
    char* kwlist[] = {"a", "b", "out", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:matrixproduct",
                                     kwlist, &a, &v, &o)) {
        return NULL;
    }
    if (o != NULL) {
        if (o == Py_None) {
            o = NULL;
        }
        else if (!PyMicArray_Check(o)) {
            PyErr_SetString(PyExc_TypeError, "'out' must be an array");
            return NULL;
        }
    }
    ret = (PyMicArrayObject *)PyMicArray_MatrixProduct2(a, v, (PyMicArrayObject *)o);
    return PyMicArray_Return(ret);
}

static PyObject *
array_vdot(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int typenum, device;
    char *ip1, *ip2, *op;
    npy_intp n, stride1, stride2;
    PyObject *op1, *op2;
    npy_intp newdimptr[1] = {-1};
    PyArray_Dims newdims = {newdimptr, 1};
    PyMicArrayObject *ap1 = NULL, *ap2  = NULL, *ret = NULL;
    PyArray_Descr *type;
    PyMicArray_DotFunc *vdot;
    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, "OO:vdot", &op1, &op2)) {
        return NULL;
    }

    device = get_common_device2(op1, op2);

    /*
     * Conjugating dot product using the BLAS for vectors.
     * Flattens both op1 and op2 before dotting.
     */
    typenum = PyMicArray_ObjectType(op1, 0);
    typenum = PyMicArray_ObjectType(op2, typenum);
    type = PyArray_DescrFromType(typenum);
    if (type == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot find a common data type.");
        return NULL;
    }

    ap1 = (PyMicArrayObject *)PyMicArray_FromAny(device, op1, type, 0, 0, 0, NULL);
    if (ap1 == NULL) {
        goto fail;
    }

    op1 = PyMicArray_Newshape(ap1, &newdims, NPY_CORDER);
    if (op1 == NULL) {
        goto fail;
    }
    Py_DECREF(ap1);
    ap1 = (PyMicArrayObject *)op1;

    Py_INCREF(type);
    ap2 = (PyMicArrayObject *)PyMicArray_FromAny(device, op2, type, 0, 0, 0, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    op2 = PyMicArray_Newshape(ap2, &newdims, NPY_CORDER);
    if (op2 == NULL) {
        goto fail;
    }
    Py_DECREF(ap2);
    ap2 = (PyMicArrayObject *)op2;

    if (PyMicArray_DIM(ap2, 0) != PyMicArray_DIM(ap1, 0)) {
        PyErr_SetString(PyExc_ValueError,
                "vectors have different lengths");
        goto fail;
    }

    /* array scalar output */
    ret = new_array_for_sum(ap1, ap2, NULL, 0, (npy_intp *)NULL, typenum, NULL);
    if (ret == NULL) {
        goto fail;
    }

    n = PyMicArray_DIM(ap1, 0);
    stride1 = PyMicArray_STRIDE(ap1, 0);
    stride2 = PyMicArray_STRIDE(ap2, 0);
    ip1 = PyMicArray_DATA(ap1);
    ip2 = PyMicArray_DATA(ap2);
    op = PyMicArray_DATA(ret);

    switch (typenum) {
        case NPY_CFLOAT:
            vdot = (PyMicArray_DotFunc *)CFLOAT_vdot;
            break;
        case NPY_CDOUBLE:
            vdot = (PyMicArray_DotFunc *)CDOUBLE_vdot;
            break;
        case NPY_CLONGDOUBLE:
            vdot = (PyMicArray_DotFunc *)CLONGDOUBLE_vdot;
            break;
        default:
            vdot = PyMicArray_GetArrFuncs(typenum)->dotfunc;
            if (vdot == NULL) {
                PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
                goto fail;
            }
    }

    if (n < 500) {
        vdot(ip1, stride1, ip2, stride2, op, n, device);
    }
    else {
        NPY_BEGIN_THREADS_DESCR(type);
        vdot(ip1, stride1, ip2, stride2, op, n, device);
        NPY_END_THREADS_DESCR(type);
    }

    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    return PyMicArray_Return(ret);
fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

static PyObject *
array_fastCopyAndTranspose(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    //TODO: implement this
    return NULL;
}


static PyObject *
array_tohost(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *array = NULL;
    PyArrayObject *ret = NULL;

    if (!PyArg_ParseTuple(args, "O&",
                &PyMicArray_GeneralConverter, &array)) {
        goto fail;
    }

    /* If array is numpy ndarray, return itself */
    if (PyArray_Check(array)) {
        return array;
    }

    ret = (PyArrayObject *) PyArray_NewLikeArray((PyArrayObject *) array,
                                            NPY_KEEPORDER, NULL, 0);
    if (PyMicArray_CopyIntoHost(ret, (PyMicArrayObject * ) array) < 0){
        Py_XDECREF(ret);
        goto fail;
    }

    Py_DECREF(array);
    return (PyObject *)ret;

fail:
    Py_XDECREF(array);
    return NULL;
}


static PyObject *
array_todevice(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"arr","device",NULL};
    PyArrayObject *array = NULL;
    PyMicArrayObject *ret = NULL;
    int device = DEFAULT_DEVICE;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
                &PyArray_Converter, &array,
                &PyMicArray_DeviceConverter, &device)) {
        goto fail;
    }

    ret = (PyMicArrayObject *)PyMicArray_NewLikeArray(device, array,
                                            NPY_KEEPORDER, NULL, 0);
    if (PyMicArray_CopyIntoFromHost(ret, array) < 0){
        Py_XDECREF(ret);
        goto fail;
    }

    Py_DECREF(array);
    return (PyObject *)ret;

fail:
    Py_XDECREF(array);
    return NULL;
}


static PyObject *
format_longfloat(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    //TODO: keep or delete
    return NULL;
}

static PyObject *
compare_chararrays(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    //TODO: implement this
    return NULL;
}


static PyObject *
test_interrupt(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int kind = 0;
    int a = 0;

    if (!PyArg_ParseTuple(args, "|i", &kind)) {
        return NULL;
    }
    if (kind) {
        Py_BEGIN_ALLOW_THREADS;
        while (a >= 0) {
            if ((a % 1000 == 0) && PyOS_InterruptOccurred()) {
                break;
            }
            a += 1;
        }
        Py_END_ALLOW_THREADS;
    }
    else {
        NPY_SIGINT_ON
        while(a >= 0) {
            a += 1;
        }
        NPY_SIGINT_OFF
    }
    return PyInt_FromLong(a);
}

static struct PyMethodDef array_module_methods[] = {
    /*{"set_typeDict",
        (PyCFunction)array_set_typeDict,
        METH_VARARGS, NULL},*/
    {"array",
        (PyCFunction)_array_fromobject,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"copyto",
        (PyCFunction)array_copyto,
        METH_VARARGS|METH_KEYWORDS, NULL},
    /*{"arange",
        (PyCFunction)array_arange,
        METH_VARARGS|METH_KEYWORDS, NULL},
    */
    {"ones",
        (PyCFunction)array_ones,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"ones_like",
        (PyCFunction)array_ones_like,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros",
        (PyCFunction)array_zeros,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros_like",
        (PyCFunction)array_zeros_like,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"count_nonzero",
        (PyCFunction)array_count_nonzero,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty",
        (PyCFunction)array_empty,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty_like",
        (PyCFunction)array_empty_like,
        METH_VARARGS|METH_KEYWORDS, NULL},
    /*{"scalar",
        (PyCFunction)array_scalar,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"where",
        (PyCFunction)array_where,
        METH_VARARGS, NULL},
    {"lexsort",
        (PyCFunction)array_lexsort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"concatenate",
        (PyCFunction)array_concatenate,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"inner",
        (PyCFunction)array_innerproduct,
        METH_VARARGS, NULL},*/
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"vdot",
        (PyCFunction)array_vdot,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /*{"matmul",
        (PyCFunction)array_matmul,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"c_einsum",
        (PyCFunction)array_einsum,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"_fastCopyAndTranspose",
        (PyCFunction)array_fastCopyAndTranspose,
        METH_VARARGS, NULL},
    {"correlate",
        (PyCFunction)array_correlate,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"correlate2",
        (PyCFunction)array_correlate2,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"frombuffer",
        (PyCFunction)array_frombuffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromfile",
        (PyCFunction)array_fromfile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"can_cast",
        (PyCFunction)array_can_cast_safely,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"promote_types",
        (PyCFunction)array_promote_types,
        METH_VARARGS, NULL},
    {"result_type",
        (PyCFunction)array_result_type,
        METH_VARARGS, NULL},*/
    {"to_cpu",
        (PyCFunction)array_tohost,
        METH_VARARGS, NULL},
    {"to_mic",
        (PyCFunction)array_todevice,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"device",
        (PyCFunction)get_current_device,
        METH_NOARGS, NULL},
    {"set_device",
        (PyCFunction)set_current_device,
        METH_O, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_array = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_array_prepare = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_array_wrap = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_array_finalize = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_buffer = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_ufunc = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_order = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_copy = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_dtype = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_ma_str_ndmin = NULL;

static int
intern_strings(void)
{
    mpy_ma_str_array = PyUString_InternFromString("__array__");
    mpy_ma_str_array_prepare = PyUString_InternFromString("__array_prepare__");
    mpy_ma_str_array_wrap = PyUString_InternFromString("__array_wrap__");
    mpy_ma_str_array_finalize = PyUString_InternFromString("__array_finalize__");
    mpy_ma_str_buffer = PyUString_InternFromString("__buffer__");
    mpy_ma_str_ufunc = PyUString_InternFromString("__numpy_ufunc__");
    mpy_ma_str_order = PyUString_InternFromString("order");
    mpy_ma_str_copy = PyUString_InternFromString("copy");
    mpy_ma_str_dtype = PyUString_InternFromString("dtype");
    mpy_ma_str_ndmin = PyUString_InternFromString("ndmin");

    return mpy_ma_str_array && mpy_ma_str_array_prepare &&
           mpy_ma_str_array_wrap && mpy_ma_str_array_finalize &&
           mpy_ma_str_buffer && mpy_ma_str_ufunc &&
           mpy_ma_str_order && mpy_ma_str_copy && mpy_ma_str_dtype &&
           mpy_ma_str_ndmin;
}

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "multiarray",
        NULL,
        -1,
        array_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit_multiarray(void) {
#else
#define RETVAL
PyMODINIT_FUNC initmultiarray(void) {
#endif
    PyObject *m, *d, *s, *obj_ndevices;
    PyObject *c_api = NULL;

    /* Init some variable */
    num_devices = omp_get_num_devices();
    current_device = omp_get_default_device();

    /* Create the module and add the functions */
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("multiarray", array_module_methods);
#endif
    if (!m) {
        goto err;
    }


    /* Import Numpy Array module */
    import_array();

#if defined(MS_WIN64) && defined(__GNUC__)
  PyErr_WarnEx(PyExc_Warning,
        "Numpy built with MINGW-W64 on Windows 64 bits is experimental, " \
        "and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    if (!d) {
        goto err;
    }

    /*
     * Before calling PyType_Ready, initialize the tp_hash slot in
     * PyArray_Type to work around mingw32 not being able initialize
     * static structure slots with functions from the Python C_API.
     */
    PyMicArray_Type.tp_hash = PyObject_HashNotImplemented;
    if (PyType_Ready(&PyMicArray_Type) < 0) {
        return RETVAL;
    }

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "multiarray.error"

     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString (d, "error", PyExc_Exception);

    s = PyUString_FromString("0.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

#define ADDCONST(NAME)                          \
    s = PyInt_FromLong(NPY_##NAME);             \
    PyDict_SetItemString(d, #NAME, s);          \
    Py_DECREF(s)


    ADDCONST(ALLOW_THREADS);
    ADDCONST(BUFSIZE);
    ADDCONST(CLIP);

    ADDCONST(ITEM_HASOBJECT);
    ADDCONST(LIST_PICKLE);
    ADDCONST(ITEM_IS_POINTER);
    ADDCONST(NEEDS_INIT);
    ADDCONST(NEEDS_PYAPI);

    ADDCONST(RAISE);
    ADDCONST(WRAP);
    ADDCONST(MAXDIMS);

#undef ADDCONST

    /* Initialize C-API */
    init_PyMicArray_API();
#ifdef PyMicArray_API_USE_CAPSULE
    c_api = PyCapsule_New((void *)PyMicArray_API, NULL, NULL);
#else
    c_api = PyCObject_FromVoidPtr((void *)PyMicArray_API, NULL);
#endif
    if (c_api != NULL) {
        PyDict_SetItemString(d, "_MICARRAY_CAPI", c_api);
    }
    Py_XDECREF(c_api);

    //Py_INCREF(&PyMicArray_Type);
    PyDict_SetItemString(d, "ndarray", (PyObject *)&PyMicArray_Type);

    /* Add some other constants */
    obj_ndevices = (PyObject *) PyInt_FromLong(num_devices);
    if (obj_ndevices != NULL) {
        PyDict_SetItemString(d, "ndevices", obj_ndevices);
        Py_DECREF(obj_ndevices);
    }

    if (!intern_strings()) {
        goto err;
    }

    return RETVAL;

 err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load multiarray module.");
    }
    return RETVAL;
}
