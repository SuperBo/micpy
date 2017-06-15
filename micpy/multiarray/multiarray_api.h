#ifndef _MPY_ARRAY_API_CLIENT_H_
#define _MPY_ARRAY_API_CLIENT_H_

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
    if (_import_pymicarray() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "micpy.multiarray failed to import"); return MICPY_IMPORT_ARRAY_RETVAL;}\
}

#endif

#define PyMicArray_Type (*(PyTypeObject *) PyMicArray_API[0])
#define PyMicArray_New \
    (*(PyObject * (*)(int, PyTypeObject *, int, npy_intp *, int, npy_intp *, void *, \
                       int, int, PyObject *)) \
     PyMicArray_API[1])
#define PyMicArray_NewFromDescr \
    (*(PyObject * (*)(int, PyTypeObject *, PyArray_Descr *, int, npy_intp *, \
                      npy_intp *, void *, int, PyObject *)) \
     PyMicArray_API[2])
#define PyMicArray_NewLikeArray \
    (*(PyObject * (*)(int, PyArrayObject *, NPY_ORDER, PyArray_Descr *, int)) \
     PyMicArray_API[3])
#define PyMicArray_Empty \
    (*(PyObject * (*)(int, int, npy_intp *, PyArray_Descr *, int)) \
     PyMicArray_API[4])
#define PyMicArray_Zeros \
    (*(PyObject * (*)(int, int, npy_intp *, PyArray_Descr *, int)) \
     PyMicArray_API[5])
#define PyMicArray_FromAny \
    (*(PyObject * (*)(PyObject *, PyArray_Descr *, int, int, int , PyObject *)) \
     PyMicArray_API[6])
#define PyMicArray_FromArray \
    (*(PyObject *(*)(PyArrayObject *, PyArray_Descr *, int, int)) \
     PyMicArray_API[7])
#define PyMicArray_CopyAnyInto \
    (*(int (*)(PyMicArrayObject *, PyMicArrayObject *)) \
     PyMicArray_API[8])
#define PyMicArray_CopyInto \
    (*(int (*)(PyMicArrayObject *, PyMicArrayObject *)) \
     PyMicArray_API[9])
#define PyMicArray_CopyIntoHost \
    (*(int (*)(PyArrayObject *, PyMicArrayObject *)) \
     PyMicArray_API[10])
#define PyMicArray_CopyIntoFromHost \
    (*(int (*)(PyMicArrayObject *, PyArrayObject *)) \
     PyMicArray_API[11])
#define PyMicArray_CopyAsFlat \
    (*(int (*)(PyMicArrayObject *, PyMicArrayObject *, NPY_ORDER)) \
     PyMicArray_API[12])
#define PyMicArray_MoveInto \
    (*(int (*)(PyMicArrayObject *, PyMicArrayObject *)) \
     PyMicArray_API[13])
#define PyMicArray_NewCopy \
    (*(PyObject * (*)(PyMicArrayObject *, NPY_ORDER)) \
     PyMicArray_API[14])
#define PyMicArray_View \
    (*(PyObject *(*)(PyMicArrayObject *, PyArray_Descr *, PyTypeObject *)) \
     PyMicArray_API[15])
#define PyMicArray_Resize \
    (*(PyObject * (*)(PyMicArrayObject *, PyArray_Dims *, int, NPY_ORDER)) \
     PyMicArray_API[16])
#define PyMicArray_Newshape \
    (*(PyObject * (*)(PyMicArrayObject *, PyArray_Dims *, NPY_ORDER)) \
     PyMicArray_API[17])
#define PyMicArray_Reshape \
    (*(PyObject * (*)(PyMicArrayObject *, PyObject *)) \
     PyMicArray_API[18];
#define PyMicArray_Transpose \
    (*(PyObject * (*)(PyMicArrayObject *, PyArray_Dims *)) \
     PyMicArray_API[19])
#define PyMicArray_SwapAxes \
    (*(PyObject * (*)(PyMicArrayObject *, int, int)) \
     PyMicArray_API[20])
#define PyMicArray_Ravel \
    (*(PyObject * (*)(PyMicArrayObject *, NPY_ORDER)) \
     PyMicArray_API[21])
#define PyMicArray_Flatten \
    (*(PyObject * (*)(PyMicArrayObject *, NPY_ORDER)) \
     PyMicArray_API[22])
#define PyMicArray_RemoveAxesInPlace \
    (*(void (*)(PyMicArrayObject *, npy_bool *)) \
     PyMicArray_API[23])
#define PyMicArray_Return \
    (*(PyObject * (*)(PyMicArrayObject *)) \
     PyMicArray_API[24])
#define PyMicArray_ToScalar \
    (*(PyObject * (*)(void *, PyMicArrayObject *) \
     PyMicArray_API[25])
#define PyMicArray_FillWithScalar \
    (*(int (*)(PyMicArrayObject *, PyObject *)) \
     PyMicArray_API[26])
#define PyMicArray_FailUnlessWriteable \
    (*(int (*)(PyMicArrayObject *, const char *)) \
     PyMicArray_API[27])
#define PyMicArray_GetNumDevices \
    (*(int (*)(void)) \
     PyMicArray_API[28])
#define PyMicArray_GetCurrentDevice \
    (*(int (*)(void)) \
     PyMicArray_API[29])
#define MpyIter_AdvancedNew \
    (*(MpyIter * (*)(int , PyMicArrayObject **, npy_uint32,\
                 NPY_ORDER, NPY_CASTING, npy_uint32 *,\
                 PyArray_Descr **, int , int **, npy_intp *, npy_intp)) \
     PyMicArray_API[30])
#define MpyIter_MultiNew \
    (*(MpyIter * (*)(int, PyMicArrayObject **, npy_uint32,\
                 NPY_ORDER, NPY_CASTING,\
                 npy_uint32 *, PyArray_Descr **)) \
     PyMicArray_API[31])
#define MpyIter_New \
    (*(MpyIter * (*)(PyMicArrayObject *, npy_uint32,\
                  NPY_ORDER, NPY_CASTING, PyArray_Descr*)) \
     PyMicArray_API[32])
#define MpyIter_Deallocate \
    (*(int (*)(MpyIter *)) \
     PyMicArray_API[33])
#define MpyIter_Reset \
    (*(int (*)(MpyIter *, char**)) \
     PyMicArray_API[34])
#define MpyIter_ResetBasePointers \
    (*(int (*)(MpyIter *, char **, char **)) \
     PyMicArray_API[35])
#define MpyIter_GetIterIndex \
    (*(npy_intp (*)(MpyIter *)) \
     PyMicArray_API[36])
#define MpyIter_GetIterIndexRange \
    (*(void (*)(MpyIter *, npy_intp *, npy_intp *)) \
     PyMicArray_API[37])
#define MpyIter_GetIterSize \
    (*(npy_intp (*)(MpyIter *)) \
     PyMicArray_API[38])
#define MpyIter_GetNDim \
    (*(int (*)(MpyIter *)) \
     PyMicArray_API[39])
#define MpyIter_GetNOp \
    (*(int (*)(MpyIter *)) \
     PyMicArray_API[40])
#define MpyIter_GetShape \
    (*(int (*)(MpyIter *, npy_intp *)) \
     PyMicArray_API[41])
#define MpyIter_GetDescrArray \
    (*(PyArray_Descr ** (*)(MpyIter *)) \
     PyMicArray_API[42])
#define MpyIter_GetOperandArray \
    (*(PyMicArrayObject ** (*)(MpyIter *)) \
     PyMicArray_API[43])
#define MpyIter_GetDataPtrArray \
    (*(char ** (*)(MpyIter *)) \
     PyMicArray_API[44])
#define MpyIter_GetInnerLoopSizePtr \
    (*(npy_intp * (*)(MpyIter *)) \
     PyMicArray_API[45])
#define MpyIter_GetInnerStrideArray \
    (*(npy_intp * (*)(MpyIter *)) \
     PyMicArray_API[46])
#define MpyIter_GetIterNext \
    (*(MpyIter_IterNextFunc * (*)(MpyIter *, char **)) \
     PyMicArray_API[47])
#define MpyIter_GetDevice \
    (*(int (*)(MpyIter *)) \
     PyMicArray_API[48])
#define MpyIter_IterationNeedsAPI \
    (*(npy_bool (*)(MpyIter *)) \
     PyMicArray_API[49])
#define MpyIter_GetOffDataPtrArray \
    (*(npy_intp * (*)(MpyIter *)) \
     PyMicArray_API[50])
#define MpyIter_GetOffInnerLoopSizePtr \
    (*(npy_intp * (*)(MpyIter *)) \
     PyMicArray_API[51])
#define MpyIter_GetOffInnerStrideArray \
    (*(npy_intp * (*)(MpyIter *)) \
     PyMicArray_API[52])
#endif
