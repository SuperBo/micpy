#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>

#define _MICARRAYMODULE
#include "arrayobject.h"
#include "arraytypes.h"
#include "mpy_common.h"
#include "common.h"
#include "creators.h"

/* Converts a type number from unsigned to signed */
static int
type_num_unsigned_to_signed(int type_num)
{
    switch (type_num) {
        case NPY_UBYTE:
            return NPY_BYTE;
        case NPY_USHORT:
            return NPY_SHORT;
        case NPY_UINT:
            return NPY_INT;
        case NPY_ULONG:
            return NPY_LONG;
        case NPY_ULONGLONG:
            return NPY_LONGLONG;
        default:
            return type_num;
    }
}

/*
 * NOTE: While this is unlikely to be a performance problem, if
 *       it is it could be reverted to a simple positive/negative
 *       check as the previous system used.
 *
 * The is_small_unsigned output flag indicates whether it's an unsigned integer,
 * and would fit in a signed integer of the same bit size.
 */
static MPY_TARGET_MIC
int min_scalar_type_num(void *valueptr, int type_num,
                                            int *is_small_unsigned)
{
    switch (type_num) {
        case NPY_BOOL: {
            return NPY_BOOL;
        }
        case NPY_UBYTE: {
            npy_ubyte value = *(npy_ubyte *)valueptr;
            if (value <= NPY_MAX_BYTE) {
                *is_small_unsigned = 1;
            }
            return NPY_UBYTE;
        }
        case NPY_BYTE: {
            npy_byte value = *(npy_byte *)valueptr;
            if (value >= 0) {
                *is_small_unsigned = 1;
                return NPY_UBYTE;
            }
            break;
        }
        case NPY_USHORT: {
            npy_ushort value = *(npy_ushort *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }

            if (value <= NPY_MAX_SHORT) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_SHORT: {
            npy_short value = *(npy_short *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_USHORT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_ULONG:
#endif
        case NPY_UINT: {
            npy_uint value = *(npy_uint *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value < NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }

            if (value <= NPY_MAX_INT) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_LONG:
#endif
        case NPY_INT: {
            npy_int value = *(npy_int *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_UINT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            break;
        }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
        case NPY_ULONG: {
            npy_ulong value = *(npy_ulong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }

            if (value <= NPY_MAX_LONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_LONG: {
            npy_long value = *(npy_long *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
            break;
        }
#endif
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_ULONG:
#endif
        case NPY_ULONGLONG: {
            npy_ulonglong value = *(npy_ulonglong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value <= NPY_MAX_ULONG) {
                if (value <= NPY_MAX_LONG) {
                    *is_small_unsigned = 1;
                }
                return NPY_ULONG;
            }
#endif

            if (value <= NPY_MAX_LONGLONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_LONG:
#endif
        case NPY_LONGLONG: {
            npy_longlong value = *(npy_longlong *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONGLONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value >= NPY_MIN_LONG) {
                return NPY_LONG;
            }
#endif
            break;
        }
        /*
         * Float types aren't allowed to be demoted to integer types,
         * but precision loss is allowed.
         */
        case NPY_HALF: {
            return NPY_HALF;
        }
        case NPY_FLOAT: {
            float value = *(float *)valueptr;
            if ((value > -65000 && value < 65000) || !isfinite(value)) {
                return NPY_HALF;
            }
            break;
        }
        case NPY_DOUBLE: {
            double value = *(double *)valueptr;
            if ((value > -65000 && value < 65000) || !isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            break;
        }
        case NPY_LONGDOUBLE: {
            npy_longdouble value = *(npy_longdouble *)valueptr;
            if ((value > -65000 && value < 65000) || !isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            else if (value > -1.7e308 && value < 1.7e308) {
                return NPY_DOUBLE;
            }
            break;
        }
        /*
         * The code to demote complex to float is disabled for now,
         * as forcing complex by adding 0j is probably desirable.
         */
        case NPY_CFLOAT: {
            /*
            npy_cfloat value = *(npy_cfloat *)valueptr;
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_FLOAT, is_small_unsigned);
            }
            */
            break;
        }
        case NPY_CDOUBLE: {
            npy_cdouble value = *(npy_cdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_DOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            break;
        }
        case NPY_CLONGDOUBLE: {
            npy_clongdouble value = *(npy_clongdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_LONGDOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            else if (value.real > -1.7e308 && value.real < 1.7e308 &&
                     value.imag > -1.7e308 && value.imag < 1.7e308) {
                return NPY_CDOUBLE;
            }
            break;
        }
    }

    return type_num;
}

NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data, int scal_device,
                    PyArray_Descr *to, NPY_CASTING casting)
{
    int swap;
    int is_small_unsigned = 0, type_num;
    npy_bool ret;
    PyArray_Descr *dtype;

    /* An aligned memory buffer large enough to hold any type */
    npy_longlong value[4];
    npy_longlong temp_value[4];

    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     */
    if (scal_type == to || casting == NPY_UNSAFE_CASTING ) {
        return 1;
    }

    /*
     * If the scalar isn't a number, or the rule is stricter than
     * NPY_SAFE_CASTING, use the straight type-based rules
     */
    if (!PyTypeNum_ISNUMBER(scal_type->type_num) ||
                            casting < NPY_SAFE_CASTING) {
        return PyArray_CanCastTypeTo(scal_type, to, casting);
    }

    swap = !PyArray_ISNBO(scal_type->byteorder);

    if (scal_device != CPU_DEVICE) {
        target_memcpy(&temp_value, scal_data, scal_type->elsize,
                            CPU_DEVICE, scal_device);
        scal_data = (char *) &temp_value;
    }
    scal_type->f->copyswap(&value, scal_data, swap, NULL);
    type_num = min_scalar_type_num(&value, scal_type->type_num,
                                &is_small_unsigned);
    /*
     * If we've got a small unsigned scalar, and the 'to' type
     * is not unsigned, then make it signed to allow the value
     * to be cast more appropriately.
     */
    if (is_small_unsigned && !(PyTypeNum_ISUNSIGNED(to->type_num))) {
        type_num = type_num_unsigned_to_signed(type_num);
    }

    dtype = PyArray_DescrFromType(type_num);
    if (dtype == NULL) {
        return 0;
    }
#if 0
    printf("min scalar cast ");
    PyObject_Print(dtype, stdout, 0);
    printf(" to ");
    PyObject_Print(to, stdout, 0);
    printf("\n");
#endif
    ret = PyArray_CanCastTypeTo(dtype, to, casting);
    Py_DECREF(dtype);
    return ret;
}

/*
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
PyMicArray_CanCastArrayTo(PyMicArrayObject *arr, PyArray_Descr *to,
                        NPY_CASTING casting)
{
    PyArray_Descr *from = PyMicArray_DESCR(arr);

    /* If it's a scalar, check the value */
    if (PyMicArray_NDIM(arr) == 0 && !PyMicArray_HASFIELDS(arr)) {
        return can_cast_scalar_to(from, PyMicArray_DATA(arr),
                                    PyMicArray_DEVICE(arr), to, casting);
    }

    /* Otherwise, use the standard rules */
    return PyArray_CanCastTypeTo(from, to, casting);
}

/*
 * This function calls Py_DECREF on flex_dtype, and replaces it with
 * a new dtype that has been adapted based on the values in data_dtype
 * and data_obj. If the flex_dtype is not flexible, it leaves it as is.
 *
 * Usually, if data_obj is not an array, dtype should be the result
 * given by the PyArray_GetArrayParamsFromObject function.
 *
 * The data_obj may be NULL if just a dtype is is known for the source.
 *
 * If *flex_dtype is NULL, returns immediately, without setting an
 * exception. This basically assumes an error was already set previously.
 *
 * The current flexible dtypes include NPY_STRING, NPY_UNICODE, NPY_VOID,
 * and NPY_DATETIME with generic units.
 */
NPY_NO_EXPORT void
PyMicArray_AdaptFlexibleDType(PyObject *data_obj, PyArray_Descr *data_dtype,
                            PyArray_Descr **flex_dtype)
{
    /* TODO: do nothing right now
     * might enable in the future
     */
    return;
}

/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to dtype --- cannot be NULL
 *
 * This function always makes a copy of arr, even if the dtype
 * doesn't change.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_CastToType(PyMicArrayObject *arr, PyArray_Descr *dtype, int is_f_order)
{
    PyMicArrayObject *out;

    /* If the requested dtype is flexible, adapt it */
    PyMicArray_AdaptFlexibleDType((PyObject *)arr, PyMicArray_DESCR(arr), &dtype);
    if (dtype == NULL) {
        return NULL;
    }

    out = (PyMicArrayObject *)PyMicArray_NewFromDescr(PyMicArray_DEVICE(arr),
                               Py_TYPE(arr), dtype,
                               PyMicArray_NDIM(arr),
                               PyMicArray_DIMS(arr),
                               NULL, NULL,
                               is_f_order,
                               (PyObject *)arr);

    if (out == NULL) {
        return NULL;
    }

    if (PyMicArray_CopyInto(out, arr) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;
}

NPY_NO_EXPORT int
PyMicArray_ObjectType(PyObject *op, int minimum_type)
{
    PyArray_Descr *min_dtype = NULL;
    PyArray_Descr *dtype, *res_dtype;
    int ret;

    if (!PyMicArray_Check(op) && !PyArray_Check(op)) {
        return PyArray_ObjectType(op, minimum_type);
    }

    if (minimum_type != NPY_NOTYPE && minimum_type >= 0) {
        min_dtype = PyArray_DescrFromType(minimum_type);
        if (min_dtype == NULL) {
            return NPY_NOTYPE;
        }
    }

    dtype = PyArray_DESCR((PyArrayObject *)op);

    /* Do type promotion with 'dtype' */
    res_dtype = PyArray_PromoteTypes(dtype, min_dtype);

    if (res_dtype == NULL) {
        ret = NPY_DEFAULT_TYPE;
    }
    else {
        ret = res_dtype->type_num;
    }

    Py_DECREF(min_dtype);
    Py_XDECREF(res_dtype);

    return ret;
}
