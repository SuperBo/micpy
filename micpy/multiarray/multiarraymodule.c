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
//#include "number.h"
//#include "numpymemoryview.h"
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

static int num_devices;
static int current_device;

NPY_NO_EXPORT int PyMicArray_GetCurrentDevice(void){
    return current_device;
}

NPY_NO_EXPORT int PyMicArray_GetNumDevices(void){
    return num_devices;
}



static PyObject *
get_current_device(PyObject *ignored, PyObject *args){
    return (PyObject *) PyInt_FromLong(current_device);
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
 */
static PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum)
{
    //TODO: implement
    return NULL;
}

/* Could perhaps be redone to not make contiguous arrays */


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

#define STRIDING_OK(op, order) \
                ((order) == NPY_ANYORDER || \
                 (order) == NPY_KEEPORDER || \
                 ((order) == NPY_CORDER && PyArray_IS_C_CONTIGUOUS(op)) || \
                 ((order) == NPY_FORTRANORDER && PyArray_IS_F_CONTIGUOUS(op)))



static PyObject *
array_copyto(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    //TODO: implement
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
array_count_nonzero(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    //TODO: implement
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
        METH_VARARGS, NULL},
    {"array",
        (PyCFunction)_array_fromobject,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"copyto",
        (PyCFunction)array_copyto,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"arange",
        (PyCFunction)array_arange,
        METH_VARARGS|METH_KEYWORDS, NULL},
    */
    {"zeros",
        (PyCFunction)array_zeros,
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
        METH_VARARGS, NULL},
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"vdot",
        (PyCFunction)array_vdot,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"matmul",
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
