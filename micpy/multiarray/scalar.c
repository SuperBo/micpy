#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#define _MICARRAYMODULE
#include "arrayobject.h"
#include "scalar.h"
#include "common.h"
#include "creators.h"
#include "mpyndarraytypes.h"
#include "arraytypes.h"
#include "convert_datatype.h"

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr)
{
    int type_num;
    int align;
    npy_intp memloc;
    if (descr == NULL) {
        descr = PyArray_DescrFromScalar(scalar);
        type_num = descr->type_num;
        Py_DECREF(descr);
    }
    else {
        type_num = descr->type_num;
    }
    switch (type_num) {
#define CASE(ut,lt) case NPY_##ut: return &(((Py##lt##ScalarObject *)scalar)->obval)
        CASE(BOOL, Bool);
        CASE(BYTE, Byte);
        CASE(UBYTE, UByte);
        CASE(SHORT, Short);
        CASE(USHORT, UShort);
        CASE(INT, Int);
        CASE(UINT, UInt);
        CASE(LONG, Long);
        CASE(ULONG, ULong);
        CASE(LONGLONG, LongLong);
        CASE(ULONGLONG, ULongLong);
        CASE(HALF, Half);
        CASE(FLOAT, Float);
        CASE(DOUBLE, Double);
        CASE(LONGDOUBLE, LongDouble);
        CASE(CFLOAT, CFloat);
        CASE(CDOUBLE, CDouble);
        CASE(CLONGDOUBLE, CLongDouble);
        CASE(OBJECT, Object);
        CASE(DATETIME, Datetime);
        CASE(TIMEDELTA, Timedelta);
        CASE(VOID, Void);
#undef CASE
        case NPY_STRING:
            return (void *)PyString_AS_STRING(scalar);
        case NPY_UNICODE:
            return (void *)PyUnicode_AS_DATA(scalar);
    }

    /*
     * Must be a user-defined type --- check to see which
     * scalar it inherits from.
     */

#define _CHK(cls) (PyObject_IsInstance(scalar, \
            (PyObject *)&Py##cls##ArrType_Type))
#define _OBJ(lt) &(((Py##lt##ScalarObject *)scalar)->obval)
#define _IFCASE(cls) if _CHK(cls) return _OBJ(cls)

    if _CHK(Number) {
        if _CHK(Integer) {
            if _CHK(SignedInteger) {
                _IFCASE(Byte);
                _IFCASE(Short);
                _IFCASE(Int);
                _IFCASE(Long);
                _IFCASE(LongLong);
                _IFCASE(Timedelta);
            }
            else {
                /* Unsigned Integer */
                _IFCASE(UByte);
                _IFCASE(UShort);
                _IFCASE(UInt);
                _IFCASE(ULong);
                _IFCASE(ULongLong);
            }
        }
        else {
            /* Inexact */
            if _CHK(Floating) {
                _IFCASE(Half);
                _IFCASE(Float);
                _IFCASE(Double);
                _IFCASE(LongDouble);
            }
            else {
                /*ComplexFloating */
                _IFCASE(CFloat);
                _IFCASE(CDouble);
                _IFCASE(CLongDouble);
            }
        }
    }
    else if (_CHK(Bool)) {
        return _OBJ(Bool);
    }
    else if (_CHK(Datetime)) {
        return _OBJ(Datetime);
    }
    else if (_CHK(Flexible)) {
        if (_CHK(String)) {
            return (void *)PyString_AS_STRING(scalar);
        }
        if (_CHK(Unicode)) {
            return (void *)PyUnicode_AS_DATA(scalar);
        }
        if (_CHK(Void)) {
            return ((PyVoidScalarObject *)scalar)->obval;
        }
    }
    else {
        _IFCASE(Object);
    }


    /*
     * Use the alignment flag to figure out where the data begins
     * after a PyObject_HEAD
     */
    memloc = (npy_intp)scalar;
    memloc += sizeof(PyObject);
    /* now round-up to the nearest alignment value */
    align = descr->alignment;
    if (align > 1) {
        memloc = ((memloc + align - 1)/align)*align;
    }
    return (void *)memloc;
#undef _IFCASE
#undef _OBJ
#undef _CHK
}

/* Does nothing with descr (cannot be NULL) */
/*NUMPY_API
  Get scalar-equivalent to a region of memory described by a descriptor.
*/
NPY_NO_EXPORT PyObject *
PyMicArray_ToScalar(void *data, PyMicArrayObject *obj)
{
    PyArray_Descr *descr = PyMicArray_DESCR(obj);
    int elsize = descr->elsize;

    /* Allocate host data for transfer */
    char host_data[elsize];

    /* Transfer scalar from device to host */
    if (omp_target_memcpy(host_data, data, elsize, 0, 0,
                CPU_DEVICE, PyMicArray_DEVICE(obj)) != 0) {
        return NULL;
    }

    return PyArray_Scalar(host_data, descr, NULL);
}

 /*
 * Return either an array or the appropriate Python object if the array
 * is 0d and matches a Python type.
 * steals reference to mp
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Return(PyMicArrayObject *mp)
{

    if (mp == NULL) {
        return NULL;
    }
    if (PyErr_Occurred()) {
        Py_XDECREF(mp);
        return NULL;
    }
    if (!PyMicArray_Check(mp)) {
        return (PyObject *)mp;
    }
    if (PyMicArray_NDIM(mp) == 0) {
        PyObject *ret;
        ret = PyMicArray_ToScalar(PyMicArray_DATA(mp), mp);
        Py_DECREF(mp);
        return ret;
    }
    else {
        return (PyObject *)mp;
    }
}

/*NUMPY_API
 * Get 0-dim array from scalar
 *
 * 0-dim array from array-scalar object
 * always contains a copy of the data
 * unless outcode is NULL, it is of void type and the referrer does
 * not own it either.
 *
 * steals reference to outcode
 */
NPY_NO_EXPORT PyObject *
PyMicArray_FromScalar(PyObject *scalar, PyArray_Descr *outcode, int device)
{
    PyArray_Descr *typecode;
    PyMicArrayObject *r;
    char *memptr;
    PyObject *ret;

    /* convert to 0-dim array of scalar typecode */
    typecode = PyArray_DescrFromScalar(scalar);
    if (typecode == NULL) {
        return NULL;
    }
    if (!PyDataType_ISNUMBER(typecode)){
        Py_DECREF(typecode);
        return NULL;
    }

    /* Need to INCREF typecode because PyArray_NewFromDescr steals a
     * reference below and we still need to access typecode afterwards. */
    Py_INCREF(typecode);
    r = (PyMicArrayObject *)PyMicArray_NewFromDescr(device,
                                    &PyMicArray_Type,
                                    typecode,
                                    0, NULL,
                                    NULL, NULL, 0, NULL);
    if (r == NULL) {
        Py_DECREF(typecode); Py_XDECREF(outcode);
        return NULL;
    }
    if (PyDataType_FLAGCHK(typecode, NPY_USE_SETITEM)) {
        PyMicArray_SetItemFunc *setitem = PyMicArray_GetArrFuncs(typecode->type_num)->setitem;
        if (setitem == NULL || setitem(scalar, PyMicArray_DATA(r), r) < 0) {
            Py_DECREF(typecode); Py_XDECREF(outcode); Py_DECREF(r);
            return NULL;
        }
    }
    else {
        memptr = scalar_value(scalar, typecode);

        target_memcpy(PyMicArray_DATA(r), memptr, PyMicArray_ITEMSIZE(r),
                        PyMicArray_DEVICE(r), CPU_DEVICE);
    }

    if (outcode == NULL) {
        Py_DECREF(typecode);
        return (PyObject *)r;
    }
    if (PyArray_EquivTypes(outcode, typecode)) {
        if (!PyTypeNum_ISEXTENDED(typecode->type_num)
                || (outcode->elsize == typecode->elsize)) {
            Py_DECREF(typecode); Py_DECREF(outcode);
            return (PyObject *)r;
        }
    }

    /* cast if necessary to desired output typecode */
    ret = PyMicArray_CastToType((PyMicArrayObject *)r, outcode, 0);
    Py_DECREF(typecode); Py_DECREF(r);
    return ret;
}
