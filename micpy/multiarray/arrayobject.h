#ifndef _MPY_ARRAYOBJECT_H
#define _MPY_ARRAYOBJECT_H
typedef struct _OffloadArrayObject {
    PyObject_HEAD
   /* Pointer to the raw data buffer */
    char *data;
    /* The number of dimensions, also called 'ndim' */
    int nd;
    /* The size in each dimension, also called 'shape' */
    npy_intp *dimensions;
    /*
     * Number of bytes to jump to get to the
     * next element in each dimension
     */
    npy_intp *strides;
    /*
     * This object is decref'd upon
     * deletion of array. Except in the
     * case of UPDATEIFCOPY which has
     * special handling.
     *
     * For views it points to the original
     * array, collapsed so no chains of
     * views occur.
     *
     * For creation from buffer object it
     * points to an object that should be
     * decref'd on deletion
     *
     * For UPDATEIFCOPY flag this is an
     * array to-be-updated upon deletion
     * of this one
     */
    PyObject *base;
    /* Pointer to type structure */
    PyArray_Descr *descr;
    /* Flags describing array -- see below */
    int flags;
    /* For weak references */
    PyObject *weakreflist;
    /* The device on which the array data reside */
    int device;
} PyMicArrayObject;

/*API*/
NPY_NO_EXPORT int
PyMicArray_SetBaseObject(PyMicArrayObject *arr, PyObject *obj);

NPY_NO_EXPORT int
PyMicArray_FailUnlessWriteable(PyMicArrayObject *obj, const char *name);

/*
 * This flag is used to mark arrays which we would like to, in the future,
 * turn into views. It causes a warning to be issued on the first attempt to
 * write to the array (but the write is allowed to succeed).
 *
 * This flag is for internal use only, and may be removed in a future release,
 * which is why the #define is not exposed to user code. Currently it is set
 * on arrays returned by ndarray.diagonal.
 */
static const int NPY_ARRAY_WARN_ON_WRITE = (1 << 31);

/*
 * Some useful macro for PyMicArrayObject
 * Some of them use PyArray implementation
 */

extern PyTypeObject PyMicArray_Type;
#define PyMicArray_Check(op) PyObject_TypeCheck(op, &PyMicArray_Type)

#define PyMicArray_ISONESEGMENT(m) (PyMicArray_NDIM(m) == 0 || \
                PyMicArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS) || \
                PyMicArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS))

#define PyMicArray_ISFORTRAN(m) (PyMicArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS) && \
                (!PyMicArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS)))

#define PyMicArray_FORTRAN_IF(m) ((PyMicArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS) ? \
                NPY_ARRAY_F_CONTIGUOUS : 0))

#define PyMicArray_ISCONTIGUOUS(m) PyMicArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS)
#define PyMicArray_ISWRITEABLE(m) PyMicArray_CHKFLAGS(m, NPY_ARRAY_WRITEABLE)
#define PyMicArray_ISALIGNED(m) PyMicArray_CHKFLAGS(m, NPY_ARRAY_ALIGNED)

#define PyMicArray_IS_C_CONTIGUOUS(m) PyMicArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS)
#define PyMicArray_IS_F_CONTIGUOUS(m) PyMicArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS)

#define PyMicArray_DEVICE(obj) (((PyMicArrayObject *)(obj))->device)
#define PyMicArray_DTYPE(obj) (((PyMicArrayObject *)(obj))->descr)
#define PyMicArray_SHAPE(obj) (((PyMicArrayObject *)(obj))->dimensions)
#define PyMicArray_NDIM(obj) (((PyMicArrayObject *)(obj))->nd)
#define PyMicArray_BYTES(obj) (((PyMicArrayObject *)(obj))->data)
#define PyMicArray_DATA(obj) ((void *)((PyMicArrayObject *)(obj))->data)
#define PyMicArray_DIMS(obj) (((PyMicArrayObject *)(obj))->dimensions)
#define PyMicArray_STRIDES(obj) (((PyMicArrayObject *)(obj))->strides)
#define PyMicArray_DIM(obj,n) (PyMicArray_DIMS(obj)[n])
#define PyMicArray_STRIDE(obj,n) (PyMicArray_STRIDES(obj)[n])
#define PyMicArray_BASE(obj) (((PyMicArrayObject *)(obj))->base)
#define PyMicArray_DESCR(obj) (((PyMicArrayObject *)(obj))->descr)
#define PyMicArray_FLAGS(obj) (((PyMicArrayObject *)(obj))->flags)
#define PyMicArray_CHKFLAGS(m, FLAGS) \
        ((((PyMicArrayObject *)(m))->flags & (FLAGS)) == (FLAGS))
#define PyMicArray_CLEARFLAGS(m, FLAGS) \
        ((PyMicArrayObject *)m)->flags &= ~FLAGS
#define PyMicArray_ENABLEFLAGS(m, FLAGS) \
        ((PyMicArrayObject *)(m))->flags |= FLAGS

#define PyMicArray_ITEMSIZE(obj) \
                    (((PyMicArrayObject *)(obj))->descr->elsize)
#define PyMicArray_TYPE(obj) \
                    (((PyMicArrayObject *)(obj))->descr->type_num)

#define PyMicArray_SIZE(m) PyArray_MultiplyList(PyMicArray_DIMS(m), PyMicArray_NDIM(m))
#define PyMicArray_NBYTES(m) (PyMicArray_ITEMSIZE(m) * PyMicArray_SIZE(m))

#define PyMicArray_UpdateFlags(o, FLAGS)\
            PyArray_UpdateFlags((PyArrayObject *)o, FLAGS)

/* Type macros*/
#define PyMicArray_ISBOOL(obj) PyTypeNum_ISBOOL(PyMicArray_TYPE(obj))
#define PyMicArray_ISUNSIGNED(obj) PyTypeNum_ISUNSIGNED(PyMicArray_TYPE(obj))
#define PyMicArray_ISSIGNED(obj) PyTypeNum_ISSIGNED(PyMicArray_TYPE(obj))
#define PyMicArray_ISINTEGER(obj) PyTypeNum_ISINTEGER(PyMicArray_TYPE(obj))
#define PyMicArray_ISFLOAT(obj) PyTypeNum_ISFLOAT(PyMicArray_TYPE(obj))
#define PyMicArray_ISNUMBER(obj) PyTypeNum_ISNUMBER(PyMicArray_TYPE(obj))
#define PyMicArray_ISSTRING(obj) PyTypeNum_ISSTRING(PyMicArray_TYPE(obj))
#define PyMicArray_ISCOMPLEX(obj) PyTypeNum_ISCOMPLEX(PyMicArray_TYPE(obj))
#define PyMicArray_ISPYTHON(obj) PyTypeNum_ISPYTHON(PyMicArray_TYPE(obj))
#define PyMicArray_ISFLEXIBLE(obj) PyTypeNum_ISFLEXIBLE(PyMicArray_TYPE(obj))
#define PyMicArray_ISDATETIME(obj) PyTypeNum_ISDATETIME(PyMicArray_TYPE(obj))
#define PyMicArray_ISUSERDEF(obj) PyTypeNum_ISUSERDEF(PyMicArray_TYPE(obj))
#define PyMicArray_ISEXTENDED(obj) PyTypeNum_ISEXTENDED(PyMicArray_TYPE(obj))
#define PyMicArray_ISOBJECT(obj) PyTypeNum_ISOBJECT(PyMicArray_TYPE(obj))
#define PyMicArray_HASFIELDS(obj) PyDataType_HASFIELDS(PyMicArray_DESCR(obj))

typedef int (PyMicArray_FinalizeFunc)(PyMicArrayObject *, PyObject *);

#endif
