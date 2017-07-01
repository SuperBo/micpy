#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

#include "npy_config.h"

#define _MICARRAYMODULE
#include "arrayobject.h"
#include "creators.h"
#include "common.h"
#include "number.h"
#include "calculation.h"
#include "array_assign.h"

static double
power_of_ten(int n)
{
    static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    double ret;
    if (n < 9) {
        ret = p10[n];
    }
    else {
        ret = 1e9;
        while (n-- > 9) {
            ret *= 10.;
        }
    }
    return ret;
}

/*NUMPY_API
 * ArgMax
 */
NPY_NO_EXPORT PyObject *
PyMicArray_ArgMax(PyMicArrayObject *op, int axis, PyMicArrayObject *out)
{
    return NULL;
}



/*NUMPY_API
 * ArgMin
 */
NPY_NO_EXPORT PyObject *
PyMicArray_ArgMin(PyMicArrayObject *op, int axis, PyMicArrayObject *out)
{
    return NULL;
}

/*NUMPY_API
 * Max
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Max(PyMicArrayObject *ap, int axis, PyMicArrayObject *out)
{
    PyMicArrayObject *arr;
    PyObject *ret;

    arr = (PyMicArrayObject *)PyMicArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }

    ret = PyMicArray_GenericReduceFunction(arr, n_ops.maximum, axis,
                                        PyMicArray_DESCR(arr)->type_num, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Min
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Min(PyMicArrayObject *ap, int axis, PyMicArrayObject *out)
{
    PyMicArrayObject *arr;
    PyObject *ret;

    arr=(PyMicArrayObject *)PyMicArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }

    ret = PyMicArray_GenericReduceFunction(arr, n_ops.minimum, axis,
                                        PyMicArray_DESCR(arr)->type_num, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Ptp
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Ptp(PyMicArrayObject *ap, int axis, PyMicArrayObject *out)
{
    return NULL;
}



/*NUMPY_API
 * Set variance to 1 to by-pass square-root calculation and return variance
 * Std
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Std(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out,
            int variance)
{
    return __New_PyMicArray_Std(self, axis, rtype, out, variance, 0);
}

NPY_NO_EXPORT PyObject *
__New_PyMicArray_Std(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out,
                  int variance, int num)
{
    return NULL;
}


/*NUMPY_API
 *Sum
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Sum(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out)
{
    return NULL;
}

/*NUMPY_API
 * Prod
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Prod(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 *CumSum
 */
NPY_NO_EXPORT PyObject *
PyMicArray_CumSum(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * CumProd
 */
NPY_NO_EXPORT PyObject *
PyMicArray_CumProd(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * Round
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Round(PyMicArrayObject *a, int decimals, PyMicArrayObject *out)
{
    //TODO
    return NULL;
}


/*NUMPY_API
 * Mean
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Mean(PyMicArrayObject *self, int axis, int rtype, PyMicArrayObject *out)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * Any
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Any(PyMicArrayObject *self, int axis, PyMicArrayObject *out)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * All
 */
NPY_NO_EXPORT PyObject *
PyMicArray_All(PyMicArrayObject *self, int axis, PyMicArrayObject *out)
{
    return NULL;
}


static PyObject *
_GenericBinaryOutFunction(PyMicArrayObject *m1, PyObject *m2, PyMicArrayObject *out,
                          PyObject *op)
{
    return NULL;
}

static PyObject *
_slow_array_clip(PyMicArrayObject *self, PyObject *min, PyObject *max, PyMicArrayObject *out)
{
    return NULL;
}


/*NUMPY_API
 * Clip
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Clip(PyMicArrayObject *self, PyObject *min, PyObject *max, PyMicArrayObject *out)
{
    return NULL;
}


/*NUMPY_API
 * Conjugate
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Conjugate(PyMicArrayObject *self, PyMicArrayObject *out)
{
    return NULL;
}

/*NUMPY_API
 * Trace
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Trace(PyMicArrayObject *self, int offset, int axis1, int axis2,
              int rtype, PyMicArrayObject *out)
{
    return NULL;
}
