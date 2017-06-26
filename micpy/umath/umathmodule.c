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

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _mpy_umathmodule_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL _mpy_umathmodule_UFUNC_API
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/npy_math.h>
#include <npy_config.h>

#define PyMicArray_API_UNIQUE_NAME _mpy_umathmodule_MICARRAY_API
#include <multiarray/multiarray_api.h>
#include <multiarray/arrayobject.h>
#include <mpymath/mpy_math.h>
#include <non_standards.h>

#define _MICARRAY_UMATHMODULE
#include "mufunc_object.h"
//#include "reducion.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
#include "funcs.inc"
#include "loops.h"
//#include "ufunc_api_creator.h"

//NPY_NO_EXPORT int initscalarmath(PyObject *);

/*
 *****************************************************************************
 **                            INIT SOME SPECIAL MUFUNCS                    **
 *****************************************************************************
 */
static PyUFunc_TypeResolutionFunc *npy_type_resolvers[12];

#define PyUFunc_OnesLikeTypeResolver (*(npy_type_resolvers[0]))
#define PyUFunc_AbsoluteTypeResolver (*(npy_type_resolvers[1]))
#define PyUFunc_AdditionTypeResolver (*(npy_type_resolvers[2]))
#define PyUFunc_SubtractionTypeResolver (*(npy_type_resolvers[3]))
#define PyUFunc_MultiplicationTypeResolver (*(npy_type_resolvers[4]))
#define PyUFunc_DivisionTypeResolver (*(npy_type_resolvers[5]))
#define PyUFunc_SimpleBinaryComparisonTypeResolver (*(npy_type_resolvers[6]))
#define PyUFunc_NegativeTypeResolver (*(npy_type_resolvers[7]))
#define PyUFunc_SimpleUnaryOperationTypeResolver (*(npy_type_resolvers[8]))
#define PyUFunc_SimpleBinaryOperationTypeResolver (*(npy_type_resolvers[9]))
#define PyUFunc_MixedDivisionTypeResolver (*(npy_type_resolvers[10]))
#define PyUFunc_IsNaTTypeResolver (*(npy_type_resolvers[11]))

#include "__umath_generated.c"

#define get_set_resolvers(idx, name) \
    if (!PyObject_HasAttrString(umath_module, name)) {\
        return -1;\
    }\
    func = PyObject_GetAttrString(umath_module, name);\
    npy_type_resolvers[idx] = ((PyUFuncObject *)func)->type_resolver;\
    Py_DECREF(func)

static int
PyMUFunc_InitSpecialTypeResolvers(PyObject *umath_module)
{
    PyObject *func;

    get_set_resolvers(0, "_ones_like");
    get_set_resolvers(1, "absolute");
    get_set_resolvers(2, "add");
    get_set_resolvers(3, "subtract");
    get_set_resolvers(4, "multiply");
    get_set_resolvers(5, "floor_divide");
    get_set_resolvers(6, "equal");
    get_set_resolvers(7, "negative");
    get_set_resolvers(8, "sign");
    get_set_resolvers(9, "maximum");
    get_set_resolvers(10, "divide");

    if (PyObject_HasAttrString(umath_module, "isnat")) {
        func = PyObject_GetAttrString(umath_module, "isnat");
        npy_type_resolvers[11] = ((PyUFuncObject *)func)->type_resolver;
        Py_DECREF(func);
    }
    else {
        // Fall back to isnan if there is no isnat
        if (PyErr_WarnEx(PyExc_RuntimeWarning, "isnat is not supported by "
                "installed numpy, use isnan", 1) < 0)
            return -1;
        get_set_resolvers(11, "isnan");
    }

    return 0;
}


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
    PyObject *m, *d, *s, *s2, *c_api, *umath;
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

    /* Import NPY_ARRAY API */
    import_array();

    /* Import UFunc API */
    import_ufunc();

    /* Workaround to get numpy internal type resolver */
    umath = PyDict_GetItemString(PyImport_GetModuleDict(), "numpy.core.umath");
    if (umath == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                            "Can not get numpy.core.umath from sys.modules");
        return RETVAL;
    }
    if (PyMUFunc_InitSpecialTypeResolvers(umath) < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                            "Get internal type resolvers from numpy failed");
        return RETVAL;
    }

    /* Import micarray */
    import_micarray();

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
    InitOperators(d);


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

    s = PyDict_GetItemString(d, "conjugate");
    s2 = PyDict_GetItemString(d, "remainder");
    /* Setup the array object's numerical structures with appropriate
       ufuncs in d*/
    /* TODO: finish this work */
    PyMicArray_SetNumericOps(d);

    PyDict_SetItemString(d, "conj", s);
    PyDict_SetItemString(d, "mod", s2);

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
