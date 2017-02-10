#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <numpy/npy_math.h>
#include <numpy/npy_3kcompat.h>

#include "npy_config.h"

#include "alloc.h"
#include "arrayobject.h"
#include "creators.h"
#include "shape.h"
#include "convert.h"

#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "common.h" /* for convert_shape_to_string */

static int
_fix_unknown_dimension(PyArray_Dims *newshape, PyMicArrayObject *arr);

static int
_attempt_nocopy_reshape(PyMicArrayObject *self, int newnd, npy_intp* newdims,
                        npy_intp *newstrides, int is_f_order);

static void
_putzero(int device, char *optr, PyObject *zero, PyArray_Descr *dtype);

/*NUMPY_API
 * Resize (reallocate data).  Only works if nothing else is referencing this
 * array and it is contiguous.  If refcheck is 0, then the reference count is
 * not checked and assumed to be 1.  You still must own this data and have no
 * weak-references and no base object.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Resize(PyMicArrayObject *self, PyArray_Dims *newshape, int refcheck,
               NPY_ORDER order)
{
    npy_intp oldsize, newsize;
    int new_nd=newshape->len, k, n, elsize;
    int refcnt;
    npy_intp* new_dimensions=newshape->ptr;
    npy_intp new_strides[NPY_MAXDIMS];
    size_t sd;
    npy_intp *dimptr;
    char *new_data;
    npy_intp largest;

    if (!PyMicArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError,
                "resize only works on single-segment arrays");
        return NULL;
    }

    if (PyMicArray_DESCR(self)->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                "Bad data-type size.");
        return NULL;
    }
    newsize = 1;
    largest = NPY_MAX_INTP / PyMicArray_DESCR(self)->elsize;
    for(k = 0; k < new_nd; k++) {
        if (new_dimensions[k] == 0) {
            break;
        }
        if (new_dimensions[k] < 0) {
            PyErr_SetString(PyExc_ValueError,
                    "negative dimensions not allowed");
            return NULL;
        }
        newsize *= new_dimensions[k];
        if (newsize <= 0 || newsize > largest) {
            return PyErr_NoMemory();
        }
    }
    oldsize = PyMicArray_SIZE(self);

    if (oldsize != newsize) {
        if (!(PyMicArray_FLAGS(self) & NPY_ARRAY_OWNDATA)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize this array: it does not own its data");
            return NULL;
        }

        if (refcheck) {
#ifdef PYPY_VERSION
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array with refcheck=True on PyPy.\n"
                    "Use the resize function or refcheck=False");

            return NULL;
#else
            refcnt = PyArray_REFCOUNT(self);
#endif /* PYPY_VERSION */
        }
        else {
            refcnt = 1;
        }
        if ((refcnt > 2)
                || (PyMicArray_BASE(self) != NULL)
                || (((PyMicArrayObject *)self)->weakreflist != NULL)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array that "
                    "references or is referenced\n"
                    "by another array in this way.  Use the resize function");
            return NULL;
        }

        if (newsize == 0) {
            sd = PyMicArray_DESCR(self)->elsize;
        }
        else {
            sd = newsize*PyMicArray_DESCR(self)->elsize;
        }
        /* Reallocate space if needed */
        new_data = PyDataMemMic_RENEW(PyMicArray_DATA(self), sd,
                                        PyMicArray_DEVICE(self));
        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        ((PyMicArrayObject *)self)->data = new_data;
    }

    if ((newsize > oldsize) && PyMicArray_ISWRITEABLE(self)) {
        /* Fill new memory with zeros */
        elsize = PyMicArray_DESCR(self)->elsize;
        if (PyDataType_FLAGCHK(PyMicArray_DESCR(self), NPY_ITEM_REFCOUNT)) {
            PyObject *zero = PyInt_FromLong(0);
            char *optr;
            optr = PyMicArray_BYTES(self) + oldsize*elsize;
            n = newsize - oldsize;
            for (k = 0; k < n; k++) {
                _putzero(PyMicArray_DEVICE(self), (char *)optr,
                            zero, PyMicArray_DESCR(self));
                optr += elsize;
            }
            Py_DECREF(zero);
        }
        else{
            void *addr = (void *) PyMicArray_BYTES(self) + oldsize * elsize;
            npy_intp fill_size = (newsize - oldsize) * elsize;
            #pragma omp target device(self->device) map(to:addr,fill_size)
            memset(addr, 0, fill_size);
        }
    }

    if (PyMicArray_NDIM(self) != new_nd) {
        /* Different number of dimensions. */
        ((PyMicArrayObject *)self)->nd = new_nd;
        /* Need new dimensions and strides arrays */
        dimptr = PyDimMem_RENEW(PyMicArray_DIMS(self), 3*new_nd);
        if (dimptr == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        ((PyMicArrayObject *)self)->dimensions = dimptr;
        ((PyMicArrayObject *)self)->strides = dimptr + new_nd;
    }

    /* make new_strides variable */
    _array_fill_strides(
        new_strides, new_dimensions, new_nd, PyMicArray_DESCR(self)->elsize,
        PyMicArray_FLAGS(self), &(((PyMicArrayObject *)self)->flags));
    memmove(PyMicArray_DIMS(self), new_dimensions, new_nd*sizeof(npy_intp));
    memmove(PyMicArray_STRIDES(self), new_strides, new_nd*sizeof(npy_intp));
    Py_RETURN_NONE;
}

/*
 * Returns a new array
 * with the new shape from the data
 * in the old array --- order-perspective depends on order argument.
 * copy-only-if-necessary
 */

/*NUMPY_API
 * New shape for an array
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Newshape(PyMicArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER order)
{
    npy_intp i;
    npy_intp *dimensions = newdims->ptr;
    PyMicArrayObject *ret;
    int ndim = newdims->len;
    npy_bool same, incref = NPY_TRUE;
    npy_intp *strides = NULL;
    npy_intp newstrides[NPY_MAXDIMS];
    int flags;

    if (order == NPY_ANYORDER) {
        order = PyMicArray_ISFORTRAN(self);
    }
    else if (order == NPY_KEEPORDER) {
        PyErr_SetString(PyExc_ValueError,
                "order 'K' is not permitted for reshaping");
        return NULL;
    }
    /*  Quick check to make sure anything actually needs to be done */
    if (ndim == PyMicArray_NDIM(self)) {
        same = NPY_TRUE;
        i = 0;
        while (same && i < ndim) {
            if (PyMicArray_DIM(self,i) != dimensions[i]) {
                same = NPY_FALSE;
            }
            i++;
        }
        if (same) {
            return PyMicArray_View(self, NULL, NULL);
        }
    }

    /*
     * fix any -1 dimensions and check new-dimensions against old size
     */
    if (_fix_unknown_dimension(newdims, self) < 0) {
        return NULL;
    }
    /*
     * sometimes we have to create a new copy of the array
     * in order to get the right orientation and
     * because we can't just re-use the buffer with the
     * data in the order it is in.
     * NPY_RELAXED_STRIDES_CHECKING: size check is unnecessary when set.
     */
    if ((PyMicArray_SIZE(self) > 1) &&
        ((order == NPY_CORDER && !PyMicArray_IS_C_CONTIGUOUS(self)) ||
         (order == NPY_FORTRANORDER && !PyMicArray_IS_F_CONTIGUOUS(self)))) {
        int success = 0;
        success = _attempt_nocopy_reshape(self, ndim, dimensions,
                                          newstrides, order);
        if (success) {
            /* no need to copy the array after all */
            strides = newstrides;
        }
        else {
            PyObject *newcopy;
            newcopy = PyMicArray_NewCopy(self, order);
            if (newcopy == NULL) {
                return NULL;
            }
            incref = NPY_FALSE;
            self = (PyMicArrayObject *)newcopy;
        }
    }
    /* We always have to interpret the contiguous buffer correctly */

    /* Make sure the flags argument is set. */
    flags = PyMicArray_FLAGS(self);
    if (ndim > 1) {
        if (order == NPY_FORTRANORDER) {
            flags &= ~NPY_ARRAY_C_CONTIGUOUS;
            flags |= NPY_ARRAY_F_CONTIGUOUS;
        }
        else {
            flags &= ~NPY_ARRAY_F_CONTIGUOUS;
            flags |= NPY_ARRAY_C_CONTIGUOUS;
        }
    }

    Py_INCREF(PyMicArray_DESCR(self));
    ret = (PyMicArrayObject *)PyMicArray_NewFromDescr_int(
                                       PyMicArray_DEVICE(self),
                                       Py_TYPE(self),
                                       PyMicArray_DESCR(self),
                                       ndim, dimensions,
                                       strides,
                                       PyMicArray_DATA(self),
                                       flags, (PyObject *)self, 0, 1);

    if (ret == NULL) {
        goto fail;
    }

    if (incref) {
        Py_INCREF(self);
    }
    if (PyMicArray_SetBaseObject(ret, (PyObject *)self)) {
        Py_DECREF(ret);
        return NULL;
    }

    PyMicArray_UpdateFlags(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return (PyObject *)ret;

 fail:
    if (!incref) {
        Py_DECREF(self);
    }
    return NULL;
}



/* For back-ward compatability -- Not recommended */

/*NUMPY_API
 * Reshape
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Reshape(PyMicArrayObject *self, PyObject *shape)
{
    PyObject *ret;
    PyArray_Dims newdims;

    if (!PyArray_IntpConverter(shape, &newdims)) {
        return NULL;
    }
    ret = PyMicArray_Newshape(self, &newdims, NPY_CORDER);
    PyDimMem_FREE(newdims.ptr);
    return ret;
}


static void
_putzero(int device, char *optr, PyObject *zero, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        npy_intp num = dtype->elsize;
        #pragma omp target device(device) map(to:optr,num)
        memset(optr, 0, num);
    }
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _putzero(device, optr + offset, zero, new);
        }
    }
    else {
        //TODO: putzero object support or delte???
        npy_intp i;
        for (i = 0; i < dtype->elsize / sizeof(zero); i++) {
            Py_INCREF(zero);
            NPY_COPY_PYOBJECT_PTR(optr, &zero);
            optr += sizeof(zero);
        }
    }
    return;
}


/*
 * attempt to reshape an array without copying data
 *
 * The requested newdims are not checked, but must be compatible with
 * the size of self, which must be non-zero. Other than that this
 * function should correctly handle all reshapes, including axes of
 * length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills newstrides
 *     with appropriate strides
 *
 * The "is_f_order" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in PyArray_STRIDES(self)).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
static int
_attempt_nocopy_reshape(PyMicArrayObject *self, int newnd, npy_intp* newdims,
                        npy_intp *newstrides, int is_f_order)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS];
    npy_intp last_stride;
    int oi, oj, ok, ni, nj, nk;

    oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oi = 0; oi < PyMicArray_NDIM(self); oi++) {
        if (PyMicArray_DIMS(self)[oi]!= 1) {
            olddims[oldnd] = PyMicArray_DIMS(self)[oi];
            oldstrides[oldnd] = PyMicArray_STRIDES(self)[oi];
            oldnd++;
        }
    }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while (ni < newnd && oi < oldnd) {
        npy_intp np = newdims[ni];
        npy_intp op = olddims[oi];

        while (np != op) {
            if (np < op) {
                /* Misses trailing 1s, these are handled later */
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        /* Check whether the original axes can be combined */
        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        /* Calculate new strides for all axes currently worked with */
        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (ni >= 1) {
        last_stride = newstrides[ni - 1];
    }
    else {
        last_stride = PyMicArray_ITEMSIZE(self);
    }
    if (is_f_order) {
        last_stride *= newdims[ni - 1];
    }
    for (nk = ni; nk < newnd; nk++) {
        newstrides[nk] = last_stride;
    }

    return 1;
}

static void
raise_reshape_size_mismatch(PyArray_Dims *newshape, PyMicArrayObject *arr)
{
    PyObject *msg = PyUString_FromFormat("cannot reshape array of size %zd "
                                         "into shape ", PyMicArray_SIZE(arr));
    PyObject *tmp = convert_shape_to_string(newshape->len, newshape->ptr, "");

    PyUString_ConcatAndDel(&msg, tmp);
    if (msg != NULL) {
        PyErr_SetObject(PyExc_ValueError, msg);
        Py_DECREF(msg);
    }
}

static int
_fix_unknown_dimension(PyArray_Dims *newshape, PyMicArrayObject *arr)
{
    npy_intp *dimensions;
    npy_intp s_original = PyMicArray_SIZE(arr);
    npy_intp i_unknown, s_known;
    int i, n;

    dimensions = newshape->ptr;
    n = newshape->len;
    s_known = 1;
    i_unknown = -1;

    for (i = 0; i < n; i++) {
        if (dimensions[i] < 0) {
            if (i_unknown == -1) {
                i_unknown = i;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                                "can only specify one unknown dimension");
                return -1;
            }
        }
        else if (npy_mul_with_overflow_intp(&s_known, s_known,
                                            dimensions[i])) {
            raise_reshape_size_mismatch(newshape, arr);
            return -1;
        }
    }

    if (i_unknown >= 0) {
        if (s_known == 0 || s_original % s_known != 0) {
            raise_reshape_size_mismatch(newshape, arr);
            return -1;
        }
        dimensions[i_unknown] = s_original / s_known;
    }
    else {
        if (s_original != s_known) {
            raise_reshape_size_mismatch(newshape, arr);
            return -1;
        }
    }
    return 0;
}

/*NUMPY_API
 *
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Squeeze(PyMicArrayObject *self)
{
    PyMicArrayObject *ret;
    npy_bool unit_dims[NPY_MAXDIMS];
    int idim, ndim, any_ones;
    npy_intp *shape;

    ndim = PyMicArray_NDIM(self);
    shape = PyMicArray_SHAPE(self);

    any_ones = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (shape[idim] == 1) {
            unit_dims[idim] = 1;
            any_ones = 1;
        }
        else {
            unit_dims[idim] = 0;
        }
    }

    /* If there were no ones to squeeze out, return the same array */
    if (!any_ones) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    ret = (PyMicArrayObject *)PyMicArray_View(self, NULL, &PyMicArray_Type);
    if (ret == NULL) {
        return NULL;
    }

    PyMicArray_RemoveAxesInPlace(ret, unit_dims);

    /*
     * If self isn't not a base class ndarray, call its
     * __array_wrap__ method
     */
    if (Py_TYPE(self) != &PyMicArray_Type) {
        PyMicArrayObject *tmp = PyMicArray_SubclassWrap(
                                    (PyMicArrayObject *) self,
                                    (PyMicArrayObject *) ret);
        Py_DECREF(ret);
        ret = tmp;
    }

    return (PyObject *)ret;
}

/*
 * Just like PyArray_Squeeze, but allows the caller to select
 * a subset of the size-one dimensions to squeeze out.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_SqueezeSelected(PyMicArrayObject *self, npy_bool *axis_flags)
{
    PyMicArrayObject *ret;
    int idim, ndim, any_ones;
    npy_intp *shape;

    ndim = PyMicArray_NDIM(self);
    shape = PyMicArray_SHAPE(self);

    /* Verify that the axes requested are all of size one */
    any_ones = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim] != 0) {
            if (shape[idim] == 1) {
                any_ones = 1;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                        "cannot select an axis to squeeze out "
                        "which has size not equal to one");
                return NULL;
            }
        }
    }

    /* If there were no axes to squeeze out, return the same array */
    if (!any_ones) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    ret = (PyMicArrayObject *)PyMicArray_View(self, NULL, &PyMicArray_Type);
    if (ret == NULL) {
        return NULL;
    }

    PyMicArray_RemoveAxesInPlace(ret, axis_flags);

    /*
     * If self isn't not a base class ndarray, call its
     * __array_wrap__ method
     */
    if (Py_TYPE(self) != &PyMicArray_Type) {
        PyMicArrayObject *tmp = PyMicArray_SubclassWrap(self, ret);
        Py_DECREF(ret);
        ret = tmp;
    }

    return (PyObject *)ret;
}

/*NUMPY_API
 * SwapAxes
 */
NPY_NO_EXPORT PyObject *
PyMicArray_SwapAxes(PyMicArrayObject *ap, int a1, int a2)
{
    PyArray_Dims new_axes;
    npy_intp dims[NPY_MAXDIMS];
    int n = PyMicArray_NDIM(ap);
    int i;

    if (a1 < 0) {
        a1 += n;
    }
    if (a2 < 0) {
        a2 += n;
    }
    if ((a1 < 0) || (a1 >= n)) {
        PyErr_SetString(PyExc_ValueError,
                        "bad axis1 argument to swapaxes");
        return NULL;
    }
    if ((a2 < 0) || (a2 >= n)) {
        PyErr_SetString(PyExc_ValueError,
                        "bad axis2 argument to swapaxes");
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        dims[i] = i;
    }
    dims[a1] = a2;
    dims[a2] = a1;

    new_axes.ptr = dims;
    new_axes.len = n;

    return PyMicArray_Transpose(ap, &new_axes);
}


/*NUMPY_API
 * Return Transpose.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Transpose(PyMicArrayObject *ap, PyArray_Dims *permute)
{
    npy_intp *axes, axis;
    npy_intp i, n;
    npy_intp permutation[NPY_MAXDIMS], reverse_permutation[NPY_MAXDIMS];
    PyMicArrayObject *ret = NULL;
    int flags;

    if (permute == NULL) {
        n = PyMicArray_NDIM(ap);
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != PyMicArray_NDIM(ap)) {
            PyErr_SetString(PyExc_ValueError,
                            "axes don't match array");
            return NULL;
        }
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        for (i = 0; i < n; i++) {
            axis = axes[i];
            if (axis < 0) {
                axis = PyMicArray_NDIM(ap) + axis;
            }
            if (axis < 0 || axis >= PyMicArray_NDIM(ap)) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid axis for this array");
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                PyErr_SetString(PyExc_ValueError,
                                "repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
    }

    flags = PyMicArray_FLAGS(ap);

    /*
     * this allocates memory for dimensions and strides (but fills them
     * incorrectly), sets up descr, and points data at PyArray_DATA(ap).
     */
    Py_INCREF(PyMicArray_DESCR(ap));
    ret = (PyMicArrayObject *)
        PyMicArray_NewFromDescr(PyMicArray_DEVICE(ap),
                             Py_TYPE(ap),
                             PyMicArray_DESCR(ap),
                             n, PyMicArray_DIMS(ap),
                             NULL, PyMicArray_DATA(ap),
                             flags,
                             (PyObject *)ap);
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    Py_INCREF(ap);
    if (PyMicArray_SetBaseObject(ret, (PyObject *)ap) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        PyMicArray_DIMS(ret)[i] = PyMicArray_DIMS(ap)[permutation[i]];
        PyMicArray_STRIDES(ret)[i] = PyMicArray_STRIDES(ap)[permutation[i]];
    }
    PyMicArray_UpdateFlags(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS |
                        NPY_ARRAY_ALIGNED);
    return (PyObject *)ret;
}

/*
 * Sorts items so stride is descending, because C-order
 * is the default in the face of ambiguity.
 */
static int _npy_stride_sort_item_comparator(const void *a, const void *b)
{
    npy_intp astride = ((const npy_stride_sort_item *)a)->stride,
            bstride = ((const npy_stride_sort_item *)b)->stride;

    /* Sort the absolute value of the strides */
    if (astride < 0) {
        astride = -astride;
    }
    if (bstride < 0) {
        bstride = -bstride;
    }

    if (astride == bstride) {
        /*
         * Make the qsort stable by next comparing the perm order.
         * (Note that two perm entries will never be equal)
         */
        npy_intp aperm = ((const npy_stride_sort_item *)a)->perm,
                bperm = ((const npy_stride_sort_item *)b)->perm;
        return (aperm < bperm) ? -1 : 1;
    }
    if (astride > bstride) {
        return -1;
    }
    return 1;
}


/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Ravel(PyMicArrayObject *arr, NPY_ORDER order)
{
    PyArray_Dims newdim = {NULL,1};
    npy_intp val[1] = {-1};

    newdim.ptr = val;

    if (order == NPY_KEEPORDER) {
        /* This handles some corner cases, such as 0-d arrays as well */
        if (PyMicArray_IS_C_CONTIGUOUS(arr)) {
            order = NPY_CORDER;
        }
        else if (PyMicArray_IS_F_CONTIGUOUS(arr)) {
            order = NPY_FORTRANORDER;
        }
    }
    else if (order == NPY_ANYORDER) {
        order = PyMicArray_ISFORTRAN(arr) ? NPY_FORTRANORDER : NPY_CORDER;
    }

    if (order == NPY_CORDER && PyMicArray_IS_C_CONTIGUOUS(arr)) {
        return PyMicArray_Newshape(arr, &newdim, NPY_CORDER);
    }
    else if (order == NPY_FORTRANORDER && PyMicArray_IS_F_CONTIGUOUS(arr)) {
        return PyMicArray_Newshape(arr, &newdim, NPY_FORTRANORDER);
    }
    /* For KEEPORDER, check if we can make a flattened view */
    else if (order == NPY_KEEPORDER) {
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        npy_intp stride;
        int i, ndim = PyMicArray_NDIM(arr);

        PyArray_CreateSortedStridePerm(PyMicArray_NDIM(arr),
                                PyMicArray_STRIDES(arr), strideperm);

        /* The output array must be contiguous, so the first stride is fixed */
        stride = PyMicArray_ITEMSIZE(arr);

        for (i = ndim-1; i >= 0; --i) {
            if (PyMicArray_DIM(arr, strideperm[i].perm) == 1) {
                /* A size one dimension does not matter */
                continue;
            }
            if (strideperm[i].stride != stride) {
                break;
            }
            stride *= PyMicArray_DIM(arr, strideperm[i].perm);
        }

        /* If all the strides matched a contiguous layout, return a view */
        if (i < 0) {
            PyMicArrayObject *ret;

            stride = PyMicArray_ITEMSIZE(arr);
            val[0] = PyMicArray_SIZE(arr);

            Py_INCREF(PyMicArray_DESCR(arr));
            ret = (PyMicArrayObject *)PyMicArray_NewFromDescr(
                               PyMicArray_DEVICE(arr),
                               Py_TYPE(arr),
                               PyMicArray_DESCR(arr),
                               1, val,
                               &stride,
                               PyMicArray_BYTES(arr),
                               PyMicArray_FLAGS(arr),
                               (PyObject *)arr);
            if (ret == NULL) {
                return NULL;
            }

            PyMicArray_UpdateFlags(ret,
                        NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
            Py_INCREF(arr);
            if (PyMicArray_SetBaseObject(ret, (PyObject *)arr) < 0) {
                Py_DECREF(ret);
                return NULL;
            }

            return (PyObject *)ret;
        }
    }

    return PyMicArray_Flatten(arr, order);
}

/*NUMPY_API
 * Flatten
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Flatten(PyMicArrayObject *a, NPY_ORDER order)
{
    PyMicArrayObject *ret;
    npy_intp size;

    if (order == NPY_ANYORDER) {
        order = PyMicArray_ISFORTRAN(a) ? NPY_FORTRANORDER : NPY_CORDER;
    }

    size = PyMicArray_SIZE(a);
    Py_INCREF(PyMicArray_DESCR(a));
    ret = (PyMicArrayObject *)PyMicArray_NewFromDescr(PyMicArray_DEVICE(a),
                               Py_TYPE(a),
                               PyMicArray_DESCR(a),
                               1, &size,
                               NULL,
                               NULL,
                               0, (PyObject *)a);
    if (ret == NULL) {
        return NULL;
    }

    if (PyMicArray_CopyAsFlat(ret, a, order) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

/* See shape.h for parameters documentation */
NPY_NO_EXPORT PyObject *
build_shape_string(npy_intp n, npy_intp *vals)
{
    npy_intp i;
    PyObject *ret, *tmp;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    i = 0;
    while (i < n && vals[i] < 0) {
        ++i;
    }

    if (i == n) {
        return PyUString_FromFormat("()");
    }
    else {
        ret = PyUString_FromFormat("(%" NPY_INTP_FMT, vals[i++]);
        if (ret == NULL) {
            return NULL;
        }
    }

    for (; i < n; ++i) {
        if (vals[i] < 0) {
            tmp = PyUString_FromString(",newaxis");
        }
        else {
            tmp = PyUString_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        PyUString_ConcatAndDel(&ret, tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    tmp = PyUString_FromFormat(")");
    PyUString_ConcatAndDel(&ret, tmp);
    return ret;
}

NPY_NO_EXPORT void
PyMicArray_RemoveAxesInPlace(PyMicArrayObject *arr, npy_bool *flags)
{
    PyArray_RemoveAxesInPlace((PyArrayObject *) arr, flags);
}
