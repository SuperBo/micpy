/* -*- c -*- */

/*
 * vim:syntax=c
 */

/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */

/*
 * _UMATHMODULE IS needed in __ufunc_api.h, included from numpy/ufuncobject.h.
 * This is a mess and it would be nice to fix it. It has nothing to do with
 * __ufunc_api.c
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL _mpy_umathmodule_ARRAY_API

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/npy_math.h>
#include <npy_config.h>

#define PyMicArray_API_UNIQUE_NAME _mpy_umathmodule_MICARRAY_API
#include <multiarray/multiarray_api.h>
#include <multiarray/arrayobject.h>

#define _MICARRAY_UMATHMODULE
#include "mufunc_object.h"
//#include "reducion.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
//#include "funcs.inc"
//#include "loops.h"
//#include "__umath_generated.c"
//#include "ufunc_api_creator.h"

//NPY_NO_EXPORT int initscalarmath(PyObject *);

/*
 *****************************************************************************
 **                            SETUP UFUNCS                                 **
 *****************************************************************************
 */

NPY_VISIBILITY_HIDDEN PyObject * mpy_um_str_out = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_um_str_subok = NULL;
NPY_VISIBILITY_HIDDEN PyObject * mpy_um_str_pyvals_name = NULL;

/* intern some strings used in ufuncs */
static int
intern_strings(void)
{
    mpy_um_str_out = PyUString_InternFromString("out");
    mpy_um_str_subok = PyUString_InternFromString("subok");
    mpy_um_str_pyvals_name = PyUString_InternFromString(MUFUNC_PYVALS_NAME);

    return mpy_um_str_out && mpy_um_str_subok;
}

/* Setup the umath module */
/* Remove for time being, it is declared in __ufunc_api.h */
/*static PyTypeObject PyUFunc_Type;*/

static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}                /* sentinel */
};


#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "umath",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#include <stdio.h>

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit_umath(void)
#else
#define RETVAL
PyMODINIT_FUNC initumath(void)
#endif
{
    PyObject *m, *d, *s, *s2, *c_api;
    int UFUNC_FLOATING_POINT_SUPPORT = 1;

#ifdef NO_UFUNC_FLOATING_POINT_SUPPORT
    UFUNC_FLOATING_POINT_SUPPORT = 0;
#endif
    /* Create the module and add the functions */
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("umath", methods);
#endif
    if (!m) {
        return RETVAL;
    }

    /* Import the array */
    if (_import_array() < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ImportError,
                            "mumath failed: Could not import array core.");
        }
        return RETVAL;
    }

    /* Import micarray */
    if (_import_pymicarray() < 0) {
        if (!PyErr_Occurred()) {
             PyErr_SetString(PyExc_ImportError,
                            "mumath failed: Could not import micarray core.");
        }
        return RETVAL;
    }

    /* Initialize the types */
    if (PyType_Ready(&PyMUFunc_Type) < 0)
        return RETVAL;

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    /*
     * TODO: generate and init PyMUFunc_API
    c_api = NpyCapsule_FromVoidPtr((void *)PyMUFunc_API, NULL);
    if (PyErr_Occurred()) {
        goto err;
    }
    PyDict_SetItemString(d, "_UFUNC_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        goto err;
    }
    */

    s = PyString_FromString("0.1.0");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* Load the ufunc operators into the array module's namespace */
    /* TODO: finish this work */
    //InitOperators(d);


#define ADDCONST(str) PyModule_AddIntConstant(m, #str, UFUNC_##str)
#define ADDSCONST(str) PyModule_AddStringConstant(m, "UFUNC_" #str, UFUNC_##str)

    ADDCONST(ERR_IGNORE);
    ADDCONST(ERR_WARN);
    ADDCONST(ERR_CALL);
    ADDCONST(ERR_RAISE);
    ADDCONST(ERR_PRINT);
    ADDCONST(ERR_LOG);
    ADDCONST(ERR_DEFAULT);

    ADDCONST(SHIFT_DIVIDEBYZERO);
    ADDCONST(SHIFT_OVERFLOW);
    ADDCONST(SHIFT_UNDERFLOW);
    ADDCONST(SHIFT_INVALID);

    ADDCONST(FPE_DIVIDEBYZERO);
    ADDCONST(FPE_OVERFLOW);
    ADDCONST(FPE_UNDERFLOW);
    ADDCONST(FPE_INVALID);

    ADDCONST(FLOATING_POINT_SUPPORT);

    ADDSCONST(PYVALS_NAME);

#undef ADDCONST
#undef ADDSCONST
    PyModule_AddIntConstant(m, "UFUNC_BUFSIZE_DEFAULT", (long)NPY_BUFSIZE);

#if defined(NPY_PY3K)
    s = PyDict_GetItemString(d, "true_divide");
    PyDict_SetItemString(d, "divide", s);
#endif

    //s = PyDict_GetItemString(d, "conjugate");
    //s2 = PyDict_GetItemString(d, "remainder");
    /* Setup the array object's numerical structures with appropriate
       ufuncs in d*/
    /* TODO: finish this work */
    //PyArray_SetNumericOps(d);

    //PyDict_SetItemString(d, "conj", s);
    //PyDict_SetItemString(d, "mod", s2);

    /* TODO: finish this work */
    //initscalarmath(m);

    if (!intern_strings()) {
        goto err;
    }

    return RETVAL;

 err:
    /* Check for errors */
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load umath module.");
    }
    return RETVAL;
}
