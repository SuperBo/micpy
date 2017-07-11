/*
  Provide Offload multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young University

  Modified for Offload device by SuperBo

  maintainer email: supernbo@gmail.com

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/npy_3kcompat.h"

#define _MICARRAYMODULE
#include "arrayobject.h"
#include "creators.h"
#include "methods.h"
#include "getset.h"
#include "alloc.h"
#include "number.h"
#include "mpy_binop_override.h"

/* NUMPY_API
 * Compute the size of an array (in number of items)
 */
NPY_NO_EXPORT npy_intp
PyMicArray_Size(PyObject *op)
{
    if (PyMicArray_Check(op)) {
        return PyArray_SIZE((PyArrayObject *)op);
    }
    else {
        return 0;
    }
}

/* NUMPY_API
 *
 * This function does nothing if obj is writeable, and raises an exception
 * (and returns -1) if obj is not writeable. It may also do other
 * house-keeping, such as issuing warnings on arrays which are transitioning
 * to become views. Always call this function at some point before writing to
 * an array.
 *
 * 'name' is a name for the array, used to give better error
 * messages. Something like "assignment destination", "output array", or even
 * just "array".
 */
NPY_NO_EXPORT int
PyMicArray_FailUnlessWriteable(PyMicArrayObject *obj, const char *name)
{
    if (!PyMicArray_ISWRITEABLE(obj)) {
        PyErr_Format(PyExc_ValueError, "%s is read-only", name);
        return -1;
    }

    return 0;
}

/*NUMPY_API
 *
 * Precondition: 'arr' is a copy of 'base' (though possibly with different
 * strides, ordering, etc.). This function sets the UPDATEIFCOPY flag and the
 * ->base pointer on 'arr', so that when 'arr' is destructed, it will copy any
 * changes back to 'base'.
 *
 * Steals a reference to 'base'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_SetUpdateIfCopyBase(PyMicArrayObject *arr, PyMicArrayObject *base)
{
    if (base == NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot UPDATEIFCOPY to NULL array");
        return -1;
    }
    if (PyMicArray_BASE(arr) != NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot set array with existing base to UPDATEIFCOPY");
        goto fail;
    }
    if (PyMicArray_FailUnlessWriteable(base, "UPDATEIFCOPY base") < 0) {
        goto fail;
    }

    /*
     * Any writes to 'arr' will magically turn into writes to 'base', so we
     * should warn if necessary.
     */
    if (PyMicArray_FLAGS(base) & NPY_ARRAY_WARN_ON_WRITE) {
        PyMicArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
    }

    /*
     * Unlike PyArray_SetBaseObject, we do not compress the chain of base
     * references.
     */
    ((PyMicArrayObject *)arr)->base = (PyObject *)base;
    PyMicArray_ENABLEFLAGS(arr, NPY_ARRAY_UPDATEIFCOPY);
    PyMicArray_CLEARFLAGS(base, NPY_ARRAY_WRITEABLE);

    return 0;

  fail:
    Py_DECREF(base);
    return -1;
}

/*NUMPY_API
 * Sets the 'base' attribute of the array. This steals a reference
 * to 'obj'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_SetBaseObject(PyMicArrayObject *arr, PyObject *obj)
{
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency to NULL after initialization");
        return -1;
    }
    /*
     * Allow the base to be set only once. Once the object which
     * owns the data is set, it doesn't make sense to change it.
     */
    if (PyMicArray_BASE(arr) != NULL) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency more than once");
        return -1;
    }

    /*
     * Don't allow infinite chains of views, always set the base
     * to the first owner of the data.
     * That is, either the first object which isn't an array,
     * or the first object which owns its own data.
     */

    while (PyMicArray_Check(obj) && (PyObject *)arr != obj) {
        PyMicArrayObject *obj_arr = (PyMicArrayObject *)obj;
        PyObject *tmp;

        /* Propagate WARN_ON_WRITE through views. */
        if (PyMicArray_FLAGS(obj_arr) & NPY_ARRAY_WARN_ON_WRITE) {
            PyArray_ENABLEFLAGS((PyArrayObject *)arr, NPY_ARRAY_WARN_ON_WRITE);
        }

        /* If this array owns its own data, stop collapsing */
        if (PyMicArray_CHKFLAGS(obj_arr, NPY_ARRAY_OWNDATA)) {
            break;
        }

        tmp = PyMicArray_BASE(obj_arr);
        /* If there's no base, stop collapsing */
        if (tmp == NULL) {
            break;
        }
        /* Stop the collapse new base when the would not be of the same
         * type (i.e. different subclass).
         */
        if (Py_TYPE(tmp) != Py_TYPE(arr)) {
            break;
        }


        Py_INCREF(tmp);
        Py_DECREF(obj);
        obj = tmp;
    }

    /* Disallow circular references */
    if ((PyObject *)arr == obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot create a circular NumPy array 'base' dependency");
        return -1;
    }

    ((PyMicArrayObject *)arr)->base = obj;

    return 0;
}


/*NUMPY_API*/
NPY_NO_EXPORT int
PyMicArray_CopyObject(PyMicArrayObject *dest, PyMicArrayObject *src_object)
{
    //TODO
    return -1;
}


/* returns an Array-Scalar Object of the type of arr
   from the given pointer to memory -- main Scalar creation function
   default new method calls this.
*/

/* Ideally, here the descriptor would contain all the information needed.
   So, that we simply need the data and the descriptor, and perhaps
   a flag
*/


/*
  Given a string return the type-number for
  the data-type with that string as the type-object name.
  Returns NPY_NOTYPE without setting an error if no type can be
  found.  Only works for user-defined data-types.
*/


/*********************** end C-API functions **********************/

/* array object functions */

static void
array_dealloc(PyMicArrayObject *self)
{
    PyMicArrayObject *fa = self;

    /* TODO: delete this line cause we don't implement buffer protocol
    _array_dealloc_buffer_info(self);
    */

    if (fa->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    if (fa->base) {
        /*
         * UPDATEIFCOPY means that base points to an
         * array that should be updated with the contents
         * of this array upon destruction.
         * fa->base->flags must have been WRITEABLE
         * (checked previously) and it was locked here
         * thus, unlock it.
         */
        if (fa->flags & NPY_ARRAY_UPDATEIFCOPY) {
            //TODO(superbo): implement this
            PyMicArray_ENABLEFLAGS(((PyMicArrayObject *)fa->base),
                                                    NPY_ARRAY_WRITEABLE);
            Py_INCREF(self); /* hold on to self in next call */
            if (PyMicArray_CopyAnyInto((PyMicArrayObject *)fa->base, self) < 0) {
                PyErr_Print();
                PyErr_Clear();
            }
            /*
             * Don't need to DECREF -- because we are deleting
             *self already...
             */
        }
        /*
         * In any case base is pointing to something that we need
         * to DECREF -- either a view or a buffer object
         */
        Py_DECREF(fa->base);
    }

    if ((fa->flags & NPY_ARRAY_OWNDATA) && fa->data) {
        /* Free internal references if an Object array */
        if (PyDataType_FLAGCHK(fa->descr, NPY_ITEM_REFCOUNT)) {
            Py_INCREF(self); /*hold on to self */
            PyArray_XDECREF((PyArrayObject *)self);
            /*
             * Don't need to DECREF -- because we are deleting
             * self already...
             */
        }
        mpy_free_cache(fa->data, PyMicArray_NBYTES(self), fa->device);
    }

    /* must match allocation in PyArray_NewFromDescr */
    mpy_free_cache_dim(fa->dimensions, 2 * fa->nd);
    Py_DECREF(fa->descr);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*NUMPY_API
 * Prints the raw data of the ndarray in a form useful for debugging
 * low-level C issues.
 */
NPY_NO_EXPORT void
PyMicArray_DebugPrint(PyMicArrayObject *obj)
{
    int i;
    PyMicArrayObject *fobj = (PyMicArrayObject *)obj;

    printf("-------------------------------------------------------\n");
    printf(" Dump of MicPy ndarray at address %p\n", obj);
    if (obj == NULL) {
        printf(" It's NULL!\n");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
        return;
    }
    printf(" ndim   : %d\n", fobj->nd);
    printf(" shape  :");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %d", (int)fobj->dimensions[i]);
    }
    printf("\n");

    printf(" dtype  : ");
    PyObject_Print((PyObject *)fobj->descr, stdout, 0);
    printf("\n");
    printf(" data   : %p\n", fobj->data);
    printf(" device : %d\n", fobj->device);
    printf(" strides:");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %d", (int)fobj->strides[i]);
    }
    printf("\n");

    printf(" base   : %p\n", fobj->base);

    printf(" flags :");
    if (fobj->flags & NPY_ARRAY_C_CONTIGUOUS)
        printf(" NPY_C_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_F_CONTIGUOUS)
        printf(" NPY_F_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_OWNDATA)
        printf(" NPY_OWNDATA");
    if (fobj->flags & NPY_ARRAY_ALIGNED)
        printf(" NPY_ALIGNED");
    if (fobj->flags & NPY_ARRAY_WRITEABLE)
        printf(" NPY_WRITEABLE");
    if (fobj->flags & NPY_ARRAY_UPDATEIFCOPY)
        printf(" NPY_UPDATEIFCOPY");
    printf("\n");

    if (fobj->base != NULL && PyMicArray_Check(fobj->base)) {
        printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
        printf("Dump of array's BASE:\n");
        PyMicArray_DebugPrint((PyMicArrayObject *)fobj->base);
        printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    }
    printf("-------------------------------------------------------\n");
    fflush(stdout);
}

static PyObject *
array_repr(PyMicArrayObject *self)
{
    PyObject* ret;
    ret = PyUString_FromFormat("micpy.array dtype='%c' on device(%d) at address(%p)",
                                PyMicArray_DESCR(self)->type,
                                self->device,
                                self->data);
    return ret;
}

static PyObject *
array_str(PyMicArrayObject *self)
{
    PyObject* ret;
    ret = PyUString_FromFormat("micpy.array dtype='%c' on device(%d)",
                                PyMicArray_DESCR(self)->type,
                                self->device);
    return ret;
}


/*
 * VOID-type arrays can only be compared equal and not-equal
 * in which case the fields are all compared by extracting the fields
 * and testing one at a time...
 * equality testing is performed using logical_ands on all the fields.
 * in-equality testing is performed using logical_ors on all the fields.
 *
 * VOID-type arrays without fields are compared for equality by comparing their
 * memory at each location directly (using string-code).
 */
static PyObject *
_void_compare(PyArrayObject *self, PyArrayObject *other, int cmp_op)
{
    //TODO
    return NULL;
}

NPY_NO_EXPORT PyObject *
array_richcompare(PyMicArrayObject *self, PyObject *other, int cmp_op)
{
    PyMicArrayObject *array_other;
    PyObject *obj_self = (PyObject *)self;
    PyObject *result = NULL;

    /* Special case for string arrays (which don't and currently can't have
     * ufunc loops defined, so there's no point in trying).
     */
    if (!PyMicArray_ISNUMBER(self)) {
        PyErr_SetString(PyExc_ValueError, "Only support number types");
        return NULL;
    }

    switch (cmp_op) {
    case Py_LT:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other, n_ops.less);
        break;
    case Py_LE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other, n_ops.less_equal);
        break;
    case Py_EQ:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self,
                (PyObject *)other,
                n_ops.equal);
        /*
         * If the comparison results in NULL, then the
         * two array objects can not be compared together;
         * indicate that
         */
        if (result == NULL) {
            /*
             * Comparisons should raise errors when element-wise comparison
             * is not possible.
             */
            /* 2015-05-14, 1.10 */
            PyErr_Clear();
            if (DEPRECATE("elementwise == comparison failed; "
                          "this will raise an error in the future.") < 0) {
                return NULL;
            }

            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        break;
    case Py_NE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, (PyObject *)other,
                n_ops.not_equal);
        if (result == NULL) {
            /*
             * Comparisons should raise errors when element-wise comparison
             * is not possible.
             */
            /* 2015-05-14, 1.10 */
            PyErr_Clear();
            if (DEPRECATE("elementwise != comparison failed; "
                          "this will raise an error in the future.") < 0) {
                return NULL;
            }

            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        break;
    case Py_GT:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.greater);
        break;
    case Py_GE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.greater_equal);
        break;
    default:
        result = Py_NotImplemented;
        Py_INCREF(result);
    }
    return result;
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyMicArray_ElementStrides(PyObject *obj)
{
    PyArrayObject *arr;
    int itemsize;
    int i, ndim;
    npy_intp *strides;

    if (!PyArray_Check(obj) && !PyMicArray_Check(obj)) {
        return 0;
    }

    arr = (PyArrayObject *)obj;

    itemsize = PyArray_ITEMSIZE(arr);
    ndim = PyArray_NDIM(arr);
    strides = PyArray_STRIDES(arr);

    for (i = 0; i < ndim; i++) {
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;
}

static PyObject *
array_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"device", "shape", "dtype", "buffer", "offset",
                             "strides", "order", NULL};
    npy_int device;
    PyArray_Descr *descr = NULL;
    int itemsize;
    PyArray_Dims dims = {NULL, 0};
    PyArray_Dims strides = {NULL, 0};
    PyArray_Chunk buffer;
    npy_longlong offset = 0;
    NPY_ORDER order = NPY_CORDER;
    int is_f_order = 0;
    PyMicArrayObject *ret;

    buffer.ptr = NULL;
    /*
     * Usually called with shape and type but can also be called with buffer,
     * strides, and swapped info For now, let's just use this to create an
     * empty, contiguous array of a specific type and shape.
     */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO&|O&O&LO&O&",
                                     kwlist,
                                     &device,
                                     PyArray_IntpConverter,
                                     &dims,
                                     PyArray_DescrConverter,
                                     &descr,
                                     PyArray_BufferConverter,
                                     &buffer,
                                     &offset,
                                     &PyArray_IntpConverter,
                                     &strides,
                                     &PyArray_OrderConverter,
                                     &order)) {
        goto fail;
    }
    if (order == NPY_FORTRANORDER) {
        is_f_order = 1;
    }
    if (descr == NULL) {
        descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    itemsize = descr->elsize;

    if (strides.ptr != NULL) {
        npy_intp nb, off;
        if (strides.len != dims.len) {
            PyErr_SetString(PyExc_ValueError,
                            "strides, if given, must be "   \
                            "the same length as shape");
            goto fail;
        }

        if (buffer.ptr == NULL) {
            nb = 0;
            off = 0;
        }
        else {
            nb = buffer.len;
            off = (npy_intp) offset;
        }


        if (!PyArray_CheckStrides(itemsize, dims.len,
                                  nb, off,
                                  dims.ptr, strides.ptr)) {
            PyErr_SetString(PyExc_ValueError,
                            "strides is incompatible "      \
                            "with shape of requested "      \
                            "array and size of buffer");
            goto fail;
        }
    }

    if (buffer.ptr == NULL) {
        ret = (PyMicArrayObject *)
            PyMicArray_NewFromDescr_int(device, subtype, descr,
                                     (int)dims.len,
                                     dims.ptr,
                                     strides.ptr, NULL, is_f_order, NULL,
                                     0, 1);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        if (PyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)) {
            /* place Py_None in object positions */
            /* TODO: support object or not */
            //PyMicArray_FillObjectArray(ret, Py_None);
            PyErr_SetString(PyExc_TypeError, "micpy does not support object array");
            descr = NULL;
            goto fail;

            /*
            if (PyErr_Occurred()) {
                descr = NULL;
                goto fail;
            }
            */
        }
    }
    else {
        /* buffer given -- use it */
        if (dims.len == 1 && dims.ptr[0] == -1) {
            dims.ptr[0] = (buffer.len-(npy_intp)offset) / itemsize;
        }
        else if ((strides.ptr == NULL) &&
                 (buffer.len < (offset + (((npy_intp)itemsize)*
                                          PyArray_MultiplyList(dims.ptr,
                                                               dims.len))))) {
            PyErr_SetString(PyExc_TypeError,
                            "buffer is too small for "      \
                            "requested array");
            goto fail;
        }
        /* get writeable and aligned */
        if (is_f_order) {
            buffer.flags |= NPY_ARRAY_F_CONTIGUOUS;
        }
        ret = (PyMicArrayObject *)
            PyMicArray_NewFromDescr_int(device, subtype, descr,
                                     dims.len, dims.ptr,
                                     strides.ptr,
                                     offset + (char *)buffer.ptr,
                                     buffer.flags, NULL, 0, 1);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        PyMicArray_UpdateFlags(ret, NPY_ARRAY_UPDATE_ALL);
        Py_INCREF(buffer.base);
        if (PyMicArray_SetBaseObject(ret, buffer.base) < 0) {
            Py_DECREF(ret);
            ret = NULL;
            goto fail;
        }
    }

    PyDimMem_FREE(dims.ptr);
    PyDimMem_FREE(strides.ptr);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(descr);
    PyDimMem_FREE(dims.ptr);
    PyDimMem_FREE(strides.ptr);
    return NULL;
}

/*
static PyObject *
array_iter(PyArrayObject *arr)
{
    if (PyArray_NDIM(arr) == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "iteration over a 0-d array");
        return NULL;
    }
    return PySeqIter_New((PyObject *)arr);
}
*/

static PyObject *
array_alloc(PyTypeObject *type, Py_ssize_t NPY_UNUSED(nitems))
{
    /* nitems will always be 0 */
    PyObject *obj = PyObject_Malloc(type->tp_basicsize);
    PyObject_Init(obj, type);
    return obj;
}

static void
array_free(PyObject * v)
{
    /* avoid same deallocator as PyBaseObject, see gentype_free */
    PyObject_Free(v);
}


NPY_NO_EXPORT PyTypeObject PyMicArray_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "micpy.ndarray",                            /* tp_name */
    sizeof(PyMicArrayObject),                   /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)array_dealloc,                  /* tp_dealloc */
    (printfunc)NULL,                            /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)array_repr,                       /* tp_repr */
    &array_as_number,                            /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    /*
     * The tp_hash slot will be set PyObject_HashNotImplemented when the
     * module is loaded.
     */
    (hashfunc)0,                                /* tp_hash */
    (ternaryfunc)0,                             /* tp_call */
    (reprfunc)array_str,                        /* tp_str */
    (getattrofunc)0,                            /* tp_getattro */
    (setattrofunc)0,                            /* tp_setattro */
    0,                                          /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
#if !defined(NPY_PY3K)
     | Py_TPFLAGS_CHECKTYPES
#endif
     | Py_TPFLAGS_BASETYPE),                    /* tp_flags */
    0,                                          /* tp_doc */
    (traverseproc)0,                            /* tp_traverse */
    (inquiry)0,                                 /* tp_clear */
    (richcmpfunc)array_richcompare,             /* tp_richcompare */
    offsetof(PyMicArrayObject, weakreflist),    /* tp_weaklistoffset */
    0, /*(getiterfunc)array_iter,*/             /* tp_iter */
    0, /*(iternextfunc)0,*/                     /* tp_iternext */
    array_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    array_getsetlist,                           /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)0,                                /* tp_init */
    (allocfunc)array_alloc,                     /* tp_alloc */
    (newfunc)array_new,                         /* tp_new */
    (freefunc)array_free,                       /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};
