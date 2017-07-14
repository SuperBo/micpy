#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/npy_cpu.h>

#include "npy_config.h"

#include "common.h"
#include "arrayobject.h"
#include "creators.h"
//#include "lowlevel_strided_loops.h"

#include "item_selection.h"
//#include "npy_sort.h"
//#include "npy_partition.h"
//#include "npy_binsearch.h"

/*NUMPY_API
 * Take
 */
NPY_NO_EXPORT PyObject *
PyMicArray_TakeFrom(PyMicArrayObject *self, PyObject *indices, int axis,
                 PyArrayObject *out, NPY_CLIPMODE clipmode)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * Put values into an array
 */
NPY_NO_EXPORT PyObject *
PyMicArray_PutTo(PyMicArrayObject *self, PyObject* values, PyObject *indices,
              NPY_CLIPMODE clipmode)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * Put values into an array according to a mask.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_PutMask(PyMicArrayObject *self, PyObject* values, PyObject* mask)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 * Repeat the array.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Repeat(PyMicArrayObject *aop, PyObject *op, int axis)
{
    //TODO
    return NULL;
}

/*NUMPY_API
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Choose(PyMicArrayObject *ip, PyObject *op, PyMicArrayObject *out,
               NPY_CLIPMODE clipmode)
{
    //TODO
    return NULL;
}


/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyMicArray_Sort(PyMicArrayObject *op, int axis, NPY_SORTKIND which)
{
    return -1;
}



/*NUMPY_API
 * Partition an array in-place
 */
NPY_NO_EXPORT int
PyMicArray_Partition(PyMicArrayObject *op, PyMicArrayObject * ktharray, int axis,
                  NPY_SELECTKIND which)
{
    return -1;
}


/*NUMPY_API
 * ArgSort an array
 */
NPY_NO_EXPORT PyObject *
PyMicArray_ArgSort(PyMicArrayObject *op, int axis, NPY_SORTKIND which)
{
    return NULL;
}


/*NUMPY_API
 * ArgPartition an array
 */
NPY_NO_EXPORT PyObject *
PyMicArray_ArgPartition(PyMicArrayObject *op, PyMicArrayObject *ktharray, int axis,
                     NPY_SELECTKIND which)
{
    return NULL;
}


/*NUMPY_API
 *LexSort an array providing indices that will sort a collection of arrays
 *lexicographically.  The first key is sorted on first, followed by the second key
 *-- requires that arg"merge"sort is available for each sort_key
 *
 *Returns an index array that shows the indexes for the lexicographic sort along
 *the given axis.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_LexSort(PyObject *sort_keys, int axis)
{
    return NULL;
}


/*NUMPY_API
 *
 * Search the sorted array op1 for the location of the items in op2. The
 * result is an array of indexes, one for each element in op2, such that if
 * the item were to be inserted in op1 just before that index the array
 * would still be in sorted order.
 *
 * Parameters
 * ----------
 * op1 : PyArrayObject *
 *     Array to be searched, must be 1-D.
 * op2 : PyObject *
 *     Array of items whose insertion indexes in op1 are wanted
 * side : {NPY_SEARCHLEFT, NPY_SEARCHRIGHT}
 *     If NPY_SEARCHLEFT, return first valid insertion indexes
 *     If NPY_SEARCHRIGHT, return last valid insertion indexes
 * perm : PyObject *
 *     Permutation array that sorts op1 (optional)
 *
 * Returns
 * -------
 * ret : PyObject *
 *   New reference to npy_intp array containing indexes where items in op2
 *   could be validly inserted into op1. NULL on error.
 *
 * Notes
 * -----
 * Binary search is used to find the indexes.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_SearchSorted(PyMicArrayObject *op1, PyObject *op2,
                     NPY_SEARCHSIDE side, PyObject *perm)
{
    return NULL;
}

/*NUMPY_API
 * Diagonal
 *
 * In NumPy versions prior to 1.7,  this function always returned a copy of
 * the diagonal array. In 1.7, the code has been updated to compute a view
 * onto 'self', but it still copies this array before returning, as well as
 * setting the internal WARN_ON_WRITE flag. In a future version, it will
 * simply return a view onto self.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Diagonal(PyMicArrayObject *self, int offset, int axis1, int axis2)
{
    return NULL;
}

/*NUMPY_API
 * Compress
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Compress(PyMicArrayObject *self, PyObject *condition, int axis,
                 PyMicArrayObject *out)
{
    return NULL;
}

/*
 * count number of nonzero bytes in 48 byte block
 * w must be aligned to 8 bytes
 *
 * even though it uses 64 bit types its faster than the bytewise sum on 32 bit
 * but a 32 bit type version would make it even faster on these platforms
 */
static NPY_INLINE npy_intp
count_nonzero_bytes_384(const npy_uint64 * w)
{
    const npy_uint64 w1 = w[0];
    const npy_uint64 w2 = w[1];
    const npy_uint64 w3 = w[2];
    const npy_uint64 w4 = w[3];
    const npy_uint64 w5 = w[4];
    const npy_uint64 w6 = w[5];
    npy_intp r;

    /*
     * last part of sideways add popcount, first three bisections can be
     * skipped as we are dealing with bytes.
     * multiplication equivalent to (x + (x>>8) + (x>>16) + (x>>24)) & 0xFF
     * multiplication overflow well defined for unsigned types.
     * w1 + w2 guaranteed to not overflow as we only have 0 and 1 data.
     */
    r = ((w1 + w2 + w3 + w4 + w5 + w6) * 0x0101010101010101ULL) >> 56ULL;

    /*
     * bytes not exclusively 0 or 1, sum them individually.
     * should only happen if one does weird stuff with views or external
     * buffers.
     * Doing this after the optimistic computation allows saving registers and
     * better pipelining
     */
    if (NPY_UNLIKELY(
             ((w1 | w2 | w3 | w4 | w5 | w6) & 0xFEFEFEFEFEFEFEFEULL) != 0)) {
        /* reload from pointer to avoid a unnecessary stack spill with gcc */
        const char * c = (const char *)w;
        npy_uintp i, count = 0;
        for (i = 0; i < 48; i++) {
            count += (c[i] != 0);
        }
        return count;
    }

    return r;
}

/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char *data, npy_intp *ashape, npy_intp *astrides)
{
    return -1;
}

/*NUMPY_API
 * Counts the number of non-zero elements in the array.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
PyMicArray_CountNonzero(PyMicArrayObject *self)
{
    return -1;
}

/*NUMPY_API
 * Nonzero
 *
 * TODO: In NumPy 2.0, should make the iteration order a parameter.
 */
NPY_NO_EXPORT PyObject *
PyMicArray_Nonzero(PyMicArrayObject *self)
{
    return NULL;
}

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
NPY_NO_EXPORT PyObject *
PyMicArray_MultiIndexGetItem(PyMicArrayObject *self, npy_intp *multi_index)
{
    return NULL;
}

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyMicArray_MultiIndexSetItem(PyMicArrayObject *self, npy_intp *multi_index,
                                                PyObject *obj)
{
    return -1;
}
