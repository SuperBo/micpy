#define PY_SSIZE_T_CLEAN
#include <stdarg.h>
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/npy_3kcompat.h>

#define _MICARRAYMODULE
#include "common.h"
#include "shape.h"
#include "scalar.h"
#include "calculation.h"
#include "creators.h"
#include "convert.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "item_selection.h"
#include "methods.h"
#include "calculation.h"
#include "multiarraymodule.h"


/* NpyArg_ParseKeywords
 *
 * Utility function that provides the keyword parsing functionality of
 * PyArg_ParseTupleAndKeywords without having to have an args argument.
 *
 */
static int
NpyArg_ParseKeywords(PyObject *keys, const char *format, char **kwlist, ...)
{
    PyObject *args = PyTuple_New(0);
    int ret;
    va_list va;

    if (args == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed to allocate new tuple");
        return 0;
    }
    va_start(va, kwlist);
    ret = PyArg_VaParseTupleAndKeywords(args, keys, format, kwlist, va);
    va_end(va);
    Py_DECREF(args);
    return ret;
}

static PyObject *
get_forwarding_ndarray_method(const char *name) {
    //TODO: implement
    return NULL;
}

/*
 * Forwards an ndarray method to a the Python function
 * numpy.core._methods.<name>(...)
 */
static PyObject *
forward_ndarray_method(PyArrayObject *self, PyObject *args, PyObject *kwds,
                            PyObject *forwarding_callable)
{
    //TODO: implement
    return NULL;
}

/*
 * Forwards an ndarray method to the function numpy.core._methods.<name>(...),
 * caching the callable in a local static variable. Note that the
 * initialization is not thread-safe, but relies on the CPython GIL to
 * be correct.
 */
#define NPY_FORWARD_NDARRAY_METHOD(name) \
        static PyObject *callable = NULL; \
        if (callable == NULL) { \
            callable = get_forwarding_ndarray_method(name); \
            if (callable == NULL) { \
                return NULL; \
            } \
        } \
        return forward_ndarray_method(self, args, kwds, callable)


static PyObject *
array_take(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    int dimension = NPY_MAXDIMS;
    PyObject *indices;
    PyMicArrayObject *out = NULL;
    NPY_CLIPMODE mode = NPY_RAISE;
    static char *kwlist[] = {"indices", "axis", "out", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O&", kwlist,
                                     &indices,
                                     PyArray_AxisConverter, &dimension,
                                     PyMicArray_OutputConverter, &out,
                                     PyArray_ClipmodeConverter, &mode))
        return NULL;

    /*
    return PyMicArray_Return((PyMicArrayObject *)
                PyMicArray_TakeFrom(self, indices, dimension, out, mode));
                */
    return NULL;
}

static PyObject *
array_fill(PyMicArrayObject *self, PyObject *args)
{
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (PyMicArray_FillWithScalar(self, obj) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
array_put(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices, *values;
    NPY_CLIPMODE mode = NPY_RAISE;
    static char *kwlist[] = {"indices", "values", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&", kwlist,
                                     &indices,
                                     &values,
                                     PyArray_ClipmodeConverter, &mode))
        return NULL;
    return PyMicArray_PutTo(self, values, indices, mode);
}

static PyObject *
array_reshape(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"order", NULL};
    PyArray_Dims newshape;
    PyObject *ret;
    NPY_ORDER order = NPY_CORDER;
    Py_ssize_t n = PyTuple_Size(args);

    if (!NpyArg_ParseKeywords(kwds, "|O&", keywords,
                PyArray_OrderConverter, &order)) {
        return NULL;
    }

    if (n <= 1) {
        if (PyTuple_GET_ITEM(args, 0) == Py_None) {
            return PyMicArray_View(self, NULL, NULL);
        }
        if (!PyArg_ParseTuple(args, "O&", PyArray_IntpConverter,
                              &newshape)) {
            return NULL;
        }
    }
    else {
        if (!PyArray_IntpConverter(args, &newshape)) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError,
                                "invalid shape");
            }
            goto fail;
        }
    }
    ret = PyMicArray_Newshape(self, &newshape, order);
    PyDimMem_FREE(newshape.ptr);
    return ret;

 fail:
    PyDimMem_FREE(newshape.ptr);
    return NULL;
}

static PyObject *
array_squeeze(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *axis_in = NULL;
    npy_bool axis_flags[NPY_MAXDIMS];

    static char *kwlist[] = {"axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                     &axis_in)) {
        return NULL;
    }

    if (axis_in == NULL || axis_in == Py_None) {
        return PyMicArray_Squeeze(self);
    }
    else {
        if (PyMicArray_ConvertMultiAxis(axis_in, PyMicArray_NDIM(self),
                                            axis_flags) != NPY_SUCCEED) {
            return NULL;
        }

        return PyMicArray_SqueezeSelected(self, axis_flags);
    }
}

static PyObject *
array_view(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *out_dtype = NULL;
    PyObject *out_type = NULL;
    PyArray_Descr *dtype = NULL;

    static char *kwlist[] = {"dtype", "type", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
                                     &out_dtype,
                                     &out_type)) {
        return NULL;
    }

    /* If user specified a positional argument, guess whether it
       represents a type or a dtype for backward compatibility. */
    if (out_dtype) {
        /* type specified? */
        if (PyType_Check(out_dtype) &&
            PyType_IsSubtype((PyTypeObject *)out_dtype,
                             &PyArray_Type)) {
            if (out_type) {
                PyErr_SetString(PyExc_ValueError,
                                "Cannot specify output type twice.");
                return NULL;
            }
            out_type = out_dtype;
            out_dtype = NULL;
        }
    }

    if ((out_type) && (!PyType_Check(out_type) ||
                       !PyType_IsSubtype((PyTypeObject *)out_type,
                                         &PyArray_Type))) {
        PyErr_SetString(PyExc_ValueError,
                        "Type must be a sub-type of ndarray type");
        return NULL;
    }

    if ((out_dtype) &&
        (PyArray_DescrConverter(out_dtype, &dtype) == NPY_FAIL)) {
        return NULL;
    }

    return PyMicArray_View(self, dtype, (PyTypeObject*)out_type);
}

static PyObject *
array_argmax(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyMicArrayObject *out = NULL;
    static char *kwlist[] = {"axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out))
        return NULL;

    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_ArgMax(self, axis, out));
}

static PyObject *
array_argmin(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyMicArrayObject *out = NULL;
    static char *kwlist[] = {"axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyMicArray_OutputConverter, &out))
        return NULL;

    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_ArgMin(self, axis, out));
}

static PyObject *
array_max(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_min(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_ptp(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyMicArrayObject *out = NULL;
    static char *kwlist[] = {"axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyMicArray_OutputConverter, &out))
        return NULL;

    return PyMicArray_Ptp(self, axis, out);
}


static PyObject *
array_swapaxes(PyMicArrayObject *self, PyObject *args)
{
    int axis1, axis2;

    if (!PyArg_ParseTuple(args, "ii", &axis1, &axis2)) {
        return NULL;
    }
    return PyMicArray_SwapAxes(self, axis1, axis2);
}


/*NUMPY_API
  Get a subset of bytes from each element of the array
  steals reference to typed, must not be NULL
*/
NPY_NO_EXPORT PyObject *
PyMicArray_GetField(PyMicArrayObject *self, PyArray_Descr *typed, int offset)
{
    PyObject *ret = NULL;
    PyObject *safe;
    static PyObject *checkfunc = NULL;

    /* check that we are not reinterpreting memory containing Objects. */
    if (_may_have_objects(PyMicArray_DESCR(self)) || _may_have_objects(typed)) {

        return NULL;
    }

    ret = PyMicArray_NewFromDescr_int(PyMicArray_DEVICE(self),
                                   Py_TYPE(self),
                                   typed,
                                   PyMicArray_NDIM(self), PyMicArray_DIMS(self),
                                   PyMicArray_STRIDES(self),
                                   PyMicArray_BYTES(self) + offset,
                                   PyMicArray_FLAGS(self) & (~NPY_ARRAY_F_CONTIGUOUS),
                                   (PyObject *)self, 0, 1);
    if (ret == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyMicArray_SetBaseObject(((PyMicArrayObject *)ret), (PyObject *)self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    PyMicArray_UpdateFlags((PyMicArrayObject *)ret, NPY_ARRAY_UPDATE_ALL);
    return ret;
}

static PyObject *
array_getfield(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{

    PyArray_Descr *dtype = NULL;
    int offset = 0;
    static char *kwlist[] = {"dtype", "offset", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|i", kwlist,
                                     PyArray_DescrConverter, &dtype,
                                     &offset)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    return PyMicArray_GetField(self, dtype, offset);
}


/*NUMPY_API
  Set a subset of bytes from each element of the array
  steals reference to dtype, must not be NULL
*/
NPY_NO_EXPORT int
PyMicArray_SetField(PyMicArrayObject *self, PyArray_Descr *dtype,
                 int offset, PyObject *val)
{
    PyObject *ret = NULL;
    int retval = 0;

    if (PyMicArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return -1;
    }

    /* getfield returns a view we can write to */
    ret = PyMicArray_GetField(self, dtype, offset);
    if (ret == NULL) {
        return -1;
    }
    //TODO
    //retval = PyMicArray_CopyObject((PyMicArrayObject *)ret, val);
    Py_DECREF(ret);
    return retval;
}

static PyObject *
array_setfield(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyArray_Descr *dtype = NULL;
    int offset = 0;
    PyObject *value;
    static char *kwlist[] = {"value", "dtype", "offset", 0};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|i", kwlist,
                                     &value,
                                     PyArray_DescrConverter, &dtype,
                                     &offset)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    if (PyMicArray_SetField(self, dtype, offset, value) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* This doesn't change the descriptor just the actual data...
 */

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyMicArray_Byteswap(PyMicArrayObject *self, npy_bool inplace)
{
    return NULL;
}


static PyObject *
array_byteswap(PyMicArrayObject *self, PyObject *args)
{
    npy_bool inplace = NPY_FALSE;

    if (!PyArg_ParseTuple(args, "|O&",
                            PyArray_BoolConverter, &inplace)) {
        return NULL;
    }
    return PyMicArray_Byteswap(self, inplace);
}


static PyObject *
array_tobytes(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO: keep or delete
    return NULL;
}


/* This should grow an order= keyword to be consistent
 */

static PyObject *
array_tofile(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO: implement
    return NULL;
}

static PyObject *
array_toscalar(PyMicArrayObject *self, PyObject *args)
{
    npy_intp multi_index[NPY_MAXDIMS];
    int n = PyTuple_GET_SIZE(args);
    int idim, ndim = PyMicArray_NDIM(self);

    /* If there is a tuple as a single argument, treat it as the argument */
    if (n == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        args = PyTuple_GET_ITEM(args, 0);
        n = PyTuple_GET_SIZE(args);
    }

    if (n == 0) {
        if (PyMicArray_SIZE(self) == 1) {
            for (idim = 0; idim < ndim; ++idim) {
                multi_index[idim] = 0;
            }
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "can only convert an array of size 1 to a Python scalar");
            return NULL;
        }
    }
    /* Special case of C-order flat indexing... :| */
    else if (n == 1 && ndim != 1) {
        npy_intp *shape = PyMicArray_SHAPE(self);
        npy_intp value, size = PyMicArray_SIZE(self);

        value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, 0));
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }

        if (check_and_adjust_index(&value, size, -1, NULL) < 0) {
            return NULL;
        }

        /* Convert the flat index into a multi-index */
        for (idim = ndim-1; idim >= 0; --idim) {
            multi_index[idim] = value % shape[idim];
            value /= shape[idim];
        }
    }
    /* A multi-index tuple */
    else if (n == ndim) {
        npy_intp value;

        for (idim = 0; idim < ndim; ++idim) {
            value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, idim));
            if (value == -1 && PyErr_Occurred()) {
                return NULL;
            }
            multi_index[idim] = value;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "incorrect number of indices for array");
        return NULL;
    }

    return PyMicArray_MultiIndexGetItem(self, multi_index);
}

static PyObject *
array_setscalar(PyMicArrayObject *self, PyObject *args)
{
    npy_intp multi_index[NPY_MAXDIMS];
    int n = PyTuple_GET_SIZE(args) - 1;
    int idim, ndim = PyMicArray_NDIM(self);
    PyObject *obj;

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError,
                "itemset must have at least one argument");
        return NULL;
    }
    if (PyMicArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return NULL;
    }

    obj = PyTuple_GET_ITEM(args, n);

    /* If there is a tuple as a single argument, treat it as the argument */
    if (n == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        args = PyTuple_GET_ITEM(args, 0);
        n = PyTuple_GET_SIZE(args);
    }

    if (n == 0) {
        if (PyMicArray_SIZE(self) == 1) {
            for (idim = 0; idim < ndim; ++idim) {
                multi_index[idim] = 0;
            }
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "can only convert an array of size 1 to a Python scalar");
        }
    }
    /* Special case of C-order flat indexing... :| */
    else if (n == 1 && ndim != 1) {
        npy_intp *shape = PyMicArray_SHAPE(self);
        npy_intp value, size = PyMicArray_SIZE(self);

        value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, 0));
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }

        if (check_and_adjust_index(&value, size, -1, NULL) < 0) {
            return NULL;
        }

        /* Convert the flat index into a multi-index */
        for (idim = ndim-1; idim >= 0; --idim) {
            multi_index[idim] = value % shape[idim];
            value /= shape[idim];
        }
    }
    /* A multi-index tuple */
    else if (n == ndim) {
        npy_intp value;

        for (idim = 0; idim < ndim; ++idim) {
            value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, idim));
            if (value == -1 && PyErr_Occurred()) {
                return NULL;
            }
            multi_index[idim] = value;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "incorrect number of indices for array");
        return NULL;
    }

    if (PyMicArray_MultiIndexSetItem(self, multi_index, obj) < 0) {
        return NULL;
    }
    else {
        Py_RETURN_NONE;
    }
}


static PyObject *
array_astype(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"dtype", "order", "casting",
                             "subok", "copy", NULL};
    PyArray_Descr *dtype = NULL;
    /*
     * TODO: UNSAFE default for compatibility, I think
     *       switching to SAME_KIND by default would be good.
     */
    NPY_CASTING casting = NPY_UNSAFE_CASTING;
    NPY_ORDER order = NPY_KEEPORDER;
    int forcecopy = 1, subok = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&ii", kwlist,
                            PyArray_DescrConverter, &dtype,
                            PyArray_OrderConverter, &order,
                            PyArray_CastingConverter, &casting,
                            &subok,
                            &forcecopy)) {
        Py_XDECREF(dtype);
        return NULL;
    }
      int idim, ndim = PyMicArray_NDIM(self);

    /*
     * If the memory layout matches and, data types are equivalent,
     * and it's not a subtype if subok is False, then we
     * can skip the copy.
     */
    if (!forcecopy && (order == NPY_KEEPORDER ||
                       (order == NPY_ANYORDER &&
                            (PyMicArray_IS_C_CONTIGUOUS(self) ||
                            PyMicArray_IS_F_CONTIGUOUS(self))) ||
                       (order == NPY_CORDER &&
                            PyMicArray_IS_C_CONTIGUOUS(self)) ||
                       (order == NPY_FORTRANORDER &&
                            PyMicArray_IS_F_CONTIGUOUS(self))) &&
                    (subok || PyMicArray_CheckExact(self)) &&
                    PyArray_EquivTypes(dtype, PyMicArray_DESCR(self))) {
        Py_DECREF(dtype);
        Py_INCREF(self);
        return (PyObject *)self;
    }
    else if (PyMicArray_CanCastArrayTo(self, dtype, casting)) {
        PyMicArrayObject *ret;

        //TODO: flexible dtype ????
        /* If the requested dtype is flexible, ignore it */
        /*PyArray_AdaptFlexibleDType((PyObject *)self, PyMicArray_DESCR(self),
                                                                    &dtype);
        if (dtype == NULL) {
            return NULL;
        }
        */

        /* This steals the reference to dtype, so no DECREF of dtype */
        ret = (PyMicArrayObject *)PyMicArray_NewLikeArray(
                                    PyMicArray_DEVICE(self),
                                    (PyArrayObject *)self, order, dtype, subok);
        if (ret == NULL) {
            return NULL;
        }

        if (PyMicArray_CopyInto(ret, self) < 0) {
            Py_DECREF(ret);
            return NULL;
        }

        return (PyObject *)ret;
    }
    else {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast array from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyMicArray_DESCR(self)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)dtype));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        Py_DECREF(dtype);
        return NULL;
    }
}

static PyObject *
array_copy(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                                     PyArray_OrderConverter, &order)) {
        return NULL;
    }

    return PyMicArray_NewCopy(self, order);
}

/* Separate from array_copy to make __copy__ preserve Fortran contiguity. */
static PyObject *
array_copy_keeporder(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyMicArray_NewCopy(self, NPY_KEEPORDER);
}

#include <stdio.h>
static PyObject *
array_resize(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"refcheck", NULL};
    Py_ssize_t size = PyTuple_Size(args);
    int refcheck = 1;
    PyArray_Dims newshape;
    PyObject *ret, *obj;


    if (!NpyArg_ParseKeywords(kwds, "|i", kwlist,  &refcheck)) {
        return NULL;
    }

    if (size == 0) {
        Py_RETURN_NONE;
    }
    else if (size == 1) {
        obj = PyTuple_GET_ITEM(args, 0);
        if (obj == Py_None) {
            Py_RETURN_NONE;
        }
        args = obj;
    }
    if (!PyArray_IntpConverter(args, &newshape)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "invalid shape");
        }
        return NULL;
    }

    ret = PyMicArray_Resize(self, &newshape, refcheck, NPY_CORDER);
    PyDimMem_FREE(newshape.ptr);
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);
    Py_RETURN_NONE;
}

static PyObject *
array_repeat(PyMicArrayObject *self, PyObject *args, PyObject *kwds) {
    PyObject *repeats;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"repeats", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist,
                                     &repeats,
                                     PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_Repeat(self, repeats, axis));
}

static PyObject *
array_choose(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"out", "mode", NULL};
    PyObject *choices;
    PyMicArrayObject *out = NULL;
    NPY_CLIPMODE clipmode = NPY_RAISE;
    Py_ssize_t n = PyTuple_Size(args);

    if (n <= 1) {
        if (!PyArg_ParseTuple(args, "O", &choices)) {
            return NULL;
        }
    }
    else {
        choices = args;
    }

    if (!NpyArg_ParseKeywords(kwds, "|O&O&", keywords,
                PyMicArray_OutputConverter, &out,
                PyArray_ClipmodeConverter, &clipmode)) {
        return NULL;
    }

    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_Choose(self, choices, out, clipmode));
}

static PyObject *
array_sort(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO: implement
    return NULL;
}

static PyObject *
array_partition(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO: implement
    return NULL;
}

static PyObject *
array_argsort(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}


static PyObject *
array_argpartition(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_searchsorted(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static void
_deepcopy_call(char *iptr, char *optr, PyArray_Descr *dtype,
               PyObject *deepcopy, PyObject *visit)
{
    //TODO
    return;
}


static PyObject *
array_deepcopy(PyMicArrayObject *self, PyObject *args)
{
    //TODO
    return NULL;
}

/* Convert Array to flat list (using getitem) */


static PyObject *
array_setstate(PyMicArrayObject *self, PyObject *args)
{
    //TODO
    return NULL;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyMicArray_Dump(PyObject *self, PyObject *file, int protocol)
{
    //TODO: implement
    return -1;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyMicArray_Dumps(PyObject *self, int protocol)
{
    //TODO: implement
    return NULL;
}


static PyObject *
array_dump(PyMicArrayObject *self, PyObject *args)
{
    PyObject *file = NULL;
    int ret;

    if (!PyArg_ParseTuple(args, "O", &file)) {
        return NULL;
    }
    ret = PyMicArray_Dump((PyObject *)self, file, 2);
    if (ret < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
array_dumps(PyMicArrayObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyMicArray_Dumps((PyObject *)self, 2);
}


static PyObject *
array_sizeof(PyMicArrayObject *self)
{
    /* object + dimension and strides */
    Py_ssize_t nbytes = sizeof(PyMicArrayObject) +
        PyMicArray_NDIM(self) * sizeof(npy_intp) * 2;
    if (PyMicArray_CHKFLAGS(self, NPY_ARRAY_OWNDATA)) {
        nbytes += PyMicArray_NBYTES(self);
    }
    return PyLong_FromSsize_t(nbytes);
}


static PyObject *
array_transpose(PyMicArrayObject *self, PyObject *args)
{
    PyObject *shape = Py_None;
    Py_ssize_t n = PyTuple_Size(args);
    PyArray_Dims permute;
    PyObject *ret;

    if (n > 1) {
        shape = args;
    }
    else if (n == 1) {
        shape = PyTuple_GET_ITEM(args, 0);
    }

    if (shape == Py_None) {
        ret = PyMicArray_Transpose(self, NULL);
    }
    else {
        if (!PyArray_IntpConverter(shape, &permute)) {
            return NULL;
        }
        ret = PyMicArray_Transpose(self, &permute);
        PyDimMem_FREE(permute.ptr);
    }

    return ret;
}

#define _CHKTYPENUM(typ) ((typ) ? (typ)->type_num : NPY_NOTYPE)

static PyObject *
array_mean(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_sum(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}


static PyObject *
array_cumsum(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArray_Descr *dtype = NULL;
    PyMicArrayObject *out = NULL;
    int rtype;
    static char *kwlist[] = {"axis", "dtype", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_DescrConverter2, &dtype,
                                     PyMicArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);
    return PyMicArray_CumSum(self, axis, rtype, out);
}

static PyObject *
array_prod(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_cumprod(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyArray_Descr *dtype = NULL;
    PyMicArrayObject *out = NULL;
    int rtype;
    static char *kwlist[] = {"axis", "dtype", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_DescrConverter2, &dtype,
                                     PyMicArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);
    return PyMicArray_CumProd(self, axis, rtype, out);
}


static PyObject *
array_dot(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *a = (PyObject *)self, *b, *o = NULL;
    PyMicArrayObject *ret;
    char* kwlist[] = {"b", "out", NULL };


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:dot", kwlist, &b, &o)) {
        return NULL;
    }

    if (o != NULL) {
        if (o == Py_None) {
            o = NULL;
        }
        else if (!PyMicArray_Check(o)) {
            PyErr_SetString(PyExc_TypeError,
                            "'out' must be an mic array");
            return NULL;
        }
    }
    ret = (PyMicArrayObject *)PyMicArray_MatrixProduct2(a, b, (PyMicArrayObject *)o);
    return PyMicArray_Return(ret);
}


static PyObject *
array_any(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}


static PyObject *
array_all(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_stddev(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_variance(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    //TODO
    return NULL;
}

static PyObject *
array_compress(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis = NPY_MAXDIMS;
    PyObject *condition;
    PyMicArrayObject *out = NULL;
    static char *kwlist[] = {"condition", "axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&", kwlist,
                                     &condition,
                                     PyArray_AxisConverter, &axis,
                                     PyArray_OutputConverter, &out)) {
        return NULL;
    }
    return PyMicArray_Return(
                (PyMicArrayObject *)PyMicArray_Compress(self, condition, axis, out));
}


static PyObject *
array_nonzero(PyMicArrayObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyMicArray_Nonzero(self);
}


static PyObject *
array_trace(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis1 = 0, axis2 = 1, offset = 0;
    PyArray_Descr *dtype = NULL;
    PyMicArrayObject *out = NULL;
    int rtype;
    static char *kwlist[] = {"offset", "axis1", "axis2", "dtype", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiO&O&", kwlist,
                                     &offset,
                                     &axis1,
                                     &axis2,
                                     PyArray_DescrConverter2, &dtype,
                                     PyMicArray_OutputConverter, &out)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    rtype = _CHKTYPENUM(dtype);
    Py_XDECREF(dtype);
    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_Trace(self, offset, axis1, axis2, rtype, out));
}

#undef _CHKTYPENUM


static PyObject *
array_clip(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *min = NULL, *max = NULL;
    PyMicArrayObject *out = NULL;
    static char *kwlist[] = {"min", "max", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO&", kwlist,
                                     &min,
                                     &max,
                                     PyArray_OutputConverter, &out)) {
        return NULL;
    }
    if (max == NULL && min == NULL) {
        PyErr_SetString(PyExc_ValueError, "One of max or min must be given.");
        return NULL;
    }
    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_Clip(self, min, max, out));
}


static PyObject *
array_conjugate(PyMicArrayObject *self, PyObject *args)
{

    PyMicArrayObject *out = NULL;
    if (!PyArg_ParseTuple(args, "|O&",
                          PyMicArray_OutputConverter,
                          &out)) {
        return NULL;
    }
    return PyMicArray_Conjugate(self, out);
}


static PyObject *
array_diagonal(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int axis1 = 0, axis2 = 1, offset = 0;
    static char *kwlist[] = {"offset", "axis1", "axis2", NULL};
    PyMicArrayObject *ret;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
                                     &offset,
                                     &axis1,
                                     &axis2)) {
        return NULL;
    }

    ret = (PyMicArrayObject *)PyMicArray_Diagonal(self, offset, axis1, axis2);
    return PyMicArray_Return(ret);
}


static PyObject *
array_flatten(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                            PyArray_OrderConverter, &order)) {
        return NULL;
    }
    return PyMicArray_Flatten(self, order);
}


static PyObject *
array_ravel(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    NPY_ORDER order = NPY_CORDER;
    static char *kwlist[] = {"order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
                            PyArray_OrderConverter, &order)) {
        return NULL;
    }
    return PyMicArray_Ravel(self, order);
}


static PyObject *
array_round(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    int decimals = 0;
    PyMicArrayObject *out = NULL;
    static char *kwlist[] = {"decimals", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&", kwlist,
                                     &decimals,
                                     PyMicArray_OutputConverter, &out)) {
        return NULL;
    }
    return PyMicArray_Return((PyMicArrayObject *)PyMicArray_Round(self, decimals, out));
}



static PyObject *
array_setflags(PyMicArrayObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"write", "align", "uic", NULL};
    PyObject *write_flag = Py_None;
    PyObject *align_flag = Py_None;
    PyObject *uic = Py_None;
    int flagback = PyMicArray_FLAGS(self);

    PyMicArrayObject *fa = (PyMicArrayObject *)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", kwlist,
                                     &write_flag,
                                     &align_flag,
                                     &uic))
        return NULL;

    if (align_flag != Py_None) {
        if (PyObject_Not(align_flag)) {
            PyMicArray_CLEARFLAGS(self, NPY_ARRAY_ALIGNED);
        }
        else if (_IsAligned(self)) {
            PyMicArray_ENABLEFLAGS(self, NPY_ARRAY_ALIGNED);
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                            "cannot set aligned flag of mis-"\
                            "aligned array to True");
            return NULL;
        }
    }

    if (uic != Py_None) {
        if (PyObject_IsTrue(uic)) {
            fa->flags = flagback;
            PyErr_SetString(PyExc_ValueError,
                            "cannot set UPDATEIFCOPY "       \
                            "flag to True");
            return NULL;
        }
        else {
            PyMicArray_CLEARFLAGS(self, NPY_ARRAY_UPDATEIFCOPY);
            Py_XDECREF(fa->base);
            fa->base = NULL;
        }
    }

    if (write_flag != Py_None) {
        if (PyObject_IsTrue(write_flag)) {
            if (_IsWriteable(self)) {
                PyMicArray_ENABLEFLAGS(self, NPY_ARRAY_WRITEABLE);
            }
            else {
                fa->flags = flagback;
                PyErr_SetString(PyExc_ValueError,
                                "cannot set WRITEABLE "
                                "flag to True of this "
                                "array");
                return NULL;
            }
        }
        else {
            PyMicArray_CLEARFLAGS(self, NPY_ARRAY_WRITEABLE);
        }
    }

    Py_RETURN_NONE;
}


static PyObject *
array_newbyteorder(PyMicArrayObject *self, PyObject *args)
{
    char endian = NPY_SWAP;
    PyArray_Descr *new;

    if (!PyArg_ParseTuple(args, "|O&", PyArray_ByteorderConverter,
                          &endian)) {
        return NULL;
    }
    new = PyArray_DescrNewByteorder(PyMicArray_DESCR(self), endian);
    if (!new) {
        return NULL;
    }
    return PyMicArray_View(self, new, NULL);

}


static PyObject *
array_tohost(PyMicArrayObject *self, PyObject *args)
{
    PyArrayObject *ret = (PyArrayObject *)PyArray_NewLikeArray((PyArrayObject *) self,
                                NPY_KEEPORDER, NULL, 0);

    if (PyMicArray_CopyIntoHost(ret, self) < 0){
        Py_XDECREF(ret);
        return NULL;
    }

    return (PyObject *) ret;
}


static PyObject *
array_complex(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    //TODO
    return NULL;
}

NPY_NO_EXPORT PyMethodDef array_methods[] = {

    /* for the sys module */
    {"__sizeof__",
        (PyCFunction) array_sizeof,
        METH_NOARGS, NULL},

    /* for the copy module */
    {"__copy__",
        (PyCFunction)array_copy_keeporder,
        METH_VARARGS, NULL},
    {"__deepcopy__",
        (PyCFunction)array_deepcopy,
        METH_VARARGS, NULL},

    /* for Pickling */
    /*{"__reduce__",
        (PyCFunction) array_reduce,
        METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction) array_setstate,
        METH_VARARGS, NULL},
    {"dumps",
        (PyCFunction) array_dumps,
        METH_VARARGS, NULL},
    {"dump",
        (PyCFunction) array_dump,
        METH_VARARGS, NULL},

    {"__complex__",
        (PyCFunction) array_complex,
        METH_VARARGS, NULL},*/

    /* Original and Extended methods added 2005 */
    /*{"all",
        (PyCFunction)array_all,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"any",
        (PyCFunction)array_any,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmax",
        (PyCFunction)array_argmax,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmin",
        (PyCFunction)array_argmin,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argpartition",
        (PyCFunction)array_argpartition,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"argsort",
        (PyCFunction)array_argsort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"astype",
        (PyCFunction)array_astype,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"byteswap",
        (PyCFunction)array_byteswap,
        METH_VARARGS, NULL},
    {"choose",
        (PyCFunction)array_choose,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip",
        (PyCFunction)array_clip,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"compress",
        (PyCFunction)array_compress,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"conj",
        (PyCFunction)array_conjugate,
        METH_VARARGS, NULL},
    {"conjugate",
        (PyCFunction)array_conjugate,
        METH_VARARGS, NULL},
    {"copy",
        (PyCFunction)array_copy,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"cumprod",
        (PyCFunction)array_cumprod,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"cumsum",
        (PyCFunction)array_cumsum,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"diagonal",
        (PyCFunction)array_diagonal,
        METH_VARARGS | METH_KEYWORDS, NULL},*/
    {"dot",
        (PyCFunction)array_dot,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fill",
        (PyCFunction)array_fill,
        METH_VARARGS, NULL},
    /*{"flatten",
        (PyCFunction)array_flatten,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"getfield",
        (PyCFunction)array_getfield,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"item",
        (PyCFunction)array_toscalar,
        METH_VARARGS, NULL},
    {"itemset",
        (PyCFunction) array_setscalar,
        METH_VARARGS, NULL},
    {"max",
        (PyCFunction)array_max,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"mean",
        (PyCFunction)array_mean,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"min",
        (PyCFunction)array_min,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"newbyteorder",
        (PyCFunction)array_newbyteorder,
        METH_VARARGS, NULL},
    {"nonzero",
        (PyCFunction)array_nonzero,
        METH_VARARGS, NULL},
    {"partition",
        (PyCFunction)array_partition,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"prod",
        (PyCFunction)array_prod,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ptp",
        (PyCFunction)array_ptp,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"put",
        (PyCFunction)array_put,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ravel",
        (PyCFunction)array_ravel,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"repeat",
        (PyCFunction)array_repeat,
        METH_VARARGS | METH_KEYWORDS, NULL},*/
    {"reshape",
        (PyCFunction)array_reshape,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"resize",
        (PyCFunction)array_resize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /*{"round",
        (PyCFunction)array_round,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"searchsorted",
        (PyCFunction)array_searchsorted,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"setfield",
        (PyCFunction)array_setfield,
        METH_VARARGS | METH_KEYWORDS, NULL},*/
    {"setflags",
        (PyCFunction)array_setflags,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /*{"sort",
        (PyCFunction)array_sort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"squeeze",
        (PyCFunction)array_squeeze,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"std",
        (PyCFunction)array_stddev,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"sum",
        (PyCFunction)array_sum,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"swapaxes",
        (PyCFunction)array_swapaxes,
        METH_VARARGS, NULL},
    {"take",
        (PyCFunction)array_take,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"tobytes",
        (PyCFunction)array_tobytes,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"tofile",
        (PyCFunction)array_tofile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"tostring",
        (PyCFunction)array_tobytes,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"trace",
        (PyCFunction)array_trace,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"transpose",
        (PyCFunction)array_transpose,
        METH_VARARGS, NULL},
    {"var",
        (PyCFunction)array_variance,
        METH_VARARGS | METH_KEYWORDS, NULL},*/
    {"view",
        (PyCFunction)array_view,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"to_cpu",
        (PyCFunction)array_tohost,
        METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};
