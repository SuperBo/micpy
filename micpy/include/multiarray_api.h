#ifdef PyMicArray_API_UNIQUE_NAME
#define PyMicArray_API PyMicArray_API_UNIQUE_NAME
#else
#define PyMicArray_API _PyMicArray_CAPI
#endif

#ifdef PyMicArray_NO_IMPORT
extern void **PyMicArray_API;
#else

#ifndef PyMicArray_API_UNIQUE_NAME
static
#endif
void **PyMicArray_API;

static int _import_pymicarray() {
    int st;
    PyObject *micpy = PyImport_ImportModule("micpy.multiarray");
    PyObject *c_api = NULL;

    if (micpy == NULL) {
        PyErr_SetString(PyExc_ImportError, "micpy.multiarray failed to import");
        return -1;
    }

    c_api = PyObject_GetAttrString(micpy, "_MICARRAY_CAPI");
    Py_DECREF(micpy);
    if (c_api == NULL) {
        PyErr_SetString(PyExc_AttributeError, "_MICARRAY_CAPI not found");
        return -1;
    }

#if ((PY_MAJOR_VERSION < 3 && PY_VERSION_HEX >= 0x02070000 ) \
        || PY_VERSION_HEX >= 0x03010000)
    if (!PyCapsule_CheckExact(c_api)) {
        PyErr_SetString(PyExc_RuntimeError, "_MICARRAY_CAPI is not PyCapsule object");
        Py_DECREF(c_api);
        return -1;
    }
    PyMicArray_API = (void **) PyCapsule_GetPointer(c_api, NULL);
#else
    if (!PyCObject_Check(c_api)) {
        PyErr_SetString(PyExc_RuntimeError, "_MICARRAY_CAPI is not PyCObject object");
        Py_DECREF(c_api);
        return -1;
    }
    PyMicArray_API = (void **) PyCObject_AsVoidPtr(c_api);
#endif

    Py_DECREF(c_api);
    if (PyMicArray_API == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "_MICARRAY_CAPI is NULL pointer");
        return -1;
    }

    return 0;
}

#if PY_VERSION_HEX >= 0x03000000
#define MICPY_IMPORT_ARRAY_RETVAL NULL
#else
#define MICPY_IMPORT_ARRAY_RETVAL
#endif

#define import_micarray() {\
    if (_import_pymicarray() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "micpy.multiarray failed to import"); return MICPY_IMPORT_ARRAY_RETVAL; }\
}
#endif

#define PyMicArray_New \
    (*(PyObject * (*)(int, PyTypeObject *, int, npy_intp *, int, npy_intp *, void *, \
                       int, int, PyObject *)) \
     PyMicArray_API[0])
#define PyMicArray_NewFromDescr \
    (*(PyObject * (*)(int, PyTypeObject *, PyArray_Descr *, int, npy_intp *, \
                      npy_intp *, void *, int, PyObject *)) \
     PyMicArray_API[1])
#define PyMicArray_NewLikeArray \
    (*(PyObject * (*)(int, PyArrayObject *, NPY_ORDER, PyArray_Descr *, int)) \
     PyMicArray_API[2])
#define PyMicArray_Empty \
    (*(PyObject * (*)(int, int, npy_intp *, PyArray_Descr *, int)) \
     PyMicArray_API[3])
#define PyMicArray_Zeros \
    (*(PyObject * (*)(int, int, npy_intp *, PyArray_Descr *, int)) \
     PyMicArray_API[4])
