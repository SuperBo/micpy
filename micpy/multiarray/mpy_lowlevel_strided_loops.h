#ifndef __MPY_LOWLEVEL_STRIDED_LOOPS_H
#define __MPY_LOWLEVEL_STRIDED_LOOPS_H
#include <npy_config.h>

/*
 * This function pointer is for unary operations that input an
 * arbitrarily strided one-dimensional array segment and output
 * an arbitrarily strided array segment of the same size.
 * It may be a fully general function, or a specialized function
 * when the strides or item size have particular known values.
 *
 * Examples of unary operations are a straight copy, a byte-swap,
 * and a casting operation,
 *
 * The 'transferdata' parameter is slightly special, following a
 * generic auxiliary data pattern defined in ndarraytypes.h
 * Use NPY_AUXDATA_CLONE and NPY_AUXDATA_FREE to deal with this data.
 *
 */
typedef void (PyMicArray_StridedUnaryOp)(void *dst, npy_intp dst_stride,
                                    void *src, npy_intp src_stride,
                                    npy_intp N, npy_intp src_itemsize,
                                    NpyAuxData *transferdata, int device);

/*
 * This is for pointers to functions which behave exactly as
 * for PyArray_StridedUnaryOp, but with an additional mask controlling
 * which values are transformed.
 *
 * In particular, the 'i'-th element is operated on if and only if
 * mask[i*mask_stride] is true.
 */
typedef void (PyMicArray_MaskedStridedUnaryOp)(void *dst, npy_intp dst_stride,
                                    void *src, npy_intp src_stride,
                                    npy_bool *mask, npy_intp mask_stride,
                                    npy_intp N, npy_intp src_itemsize,
                                    NpyAuxData *transferdata, int device);

/*
 * Gives back a function pointer to a specialized function for copying
 * strided memory.  Returns NULL if there is a problem with the inputs.
 *
 * aligned:
 *      Should be 1 if the src and dst pointers are always aligned,
 *      0 otherwise.
 * src_stride:
 *      Should be the src stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * dst_stride:
 *      Should be the dst stride if it will always be the same,
 *      NPY_MAX_INTP otherwise.
 * itemsize:
 *      Should be the item size if it will always be the same, 0 otherwise.
 *
 */
NPY_NO_EXPORT PyMicArray_StridedUnaryOp *
PyMicArray_GetStridedCopyFn(int aligned,
                        npy_intp src_stride, npy_intp dst_stride,
                        npy_intp itemsize);

/*
 * Gives back a function pointer to a specialized function for copying
 * and swapping strided memory.  This assumes each element is a single
 * value to be swapped.
 *
 * For information on the 'aligned', 'src_stride' and 'dst_stride' parameters
 * see above.
 *
 * Parameters are as for PyArray_GetStridedCopyFn.
 */
NPY_NO_EXPORT PyMicArray_StridedUnaryOp *
PyMicArray_GetStridedCopySwapFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp itemsize);

/*
 * Gives back a function pointer to a specialized function for copying
 * and swapping strided memory.  This assumes each element is a pair
 * of values, each of which needs to be swapped.
 *
 * For information on the 'aligned', 'src_stride' and 'dst_stride' parameters
 * see above.
 *
 * Parameters are as for PyArray_GetStridedCopyFn.
 */
NPY_NO_EXPORT PyMicArray_StridedUnaryOp *
PyMicArray_GetStridedCopySwapPairFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp itemsize);

/*
 * Gives back a transfer function and transfer data pair which copies
 * the data from source to dest, truncating it if the data doesn't
 * fit, and padding with zero bytes if there's too much space.
 *
 * For information on the 'aligned', 'src_stride' and 'dst_stride' parameters
 * see above.
 *
 * Returns NPY_SUCCEED or NPY_FAIL
 */
/*NPY_NO_EXPORT int
PyMicArray_GetStridedZeroPadCopyFn(int aligned, int unicode_swap,
                            npy_intp src_stride, npy_intp dst_stride,
                            npy_intp src_itemsize, npy_intp dst_itemsize,
                            PyMicArray_StridedUnaryOp **outstransfer,
                            NpyAuxData **outtransferdata);*/

/*
 * For casts between built-in numeric types,
 * this produces a function pointer for casting from src_type_num
 * to dst_type_num.  If a conversion is unsupported, returns NULL
 * without setting a Python exception.
 */
NPY_NO_EXPORT PyMicArray_StridedUnaryOp *
PyMicArray_GetStridedNumericCastFn(int aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            int src_type_num, int dst_type_num);


#pragma omp declare target
/*
 * Return number of elements that must be peeled from
 * the start of 'addr' with 'nvals' elements of size 'esize'
 * in order to reach 'alignment'.
 * alignment must be a power of two.
 * see npy_blocked_end for an example
 */
static NPY_INLINE npy_uintp
mpy_aligned_block_offset(const void * addr, const npy_uintp esize,
                         const npy_uintp alignment, const npy_uintp nvals)
{
    const npy_uintp offset = (npy_uintp)addr & (alignment - 1);
    npy_uintp peel = offset ? (alignment - offset) / esize : 0;
    peel = nvals < peel ? nvals : peel;
    return peel;
}

/*
 * Return upper loop bound for an array of 'nvals' elements
 * of size 'esize' peeled by 'offset' elements and blocking to
 * a vector size of 'vsz' in bytes
 *
 * example usage:
 * npy_intp i;
 * double v[101];
 * npy_intp esize = sizeof(v[0]);
 * npy_intp peel = npy_aligned_block_offset(v, esize, 16, n);
 * // peel to alignment 16
 * for (i = 0; i < peel; i++)
 *   <scalar-op>
 * // simd vectorized operation
 * for (; i < npy_blocked_end(peel, esize, 16, n); i += 16 / esize)
 *   <blocked-op>
 * // handle scalar rest
 * for(; i < n; i++)
 *   <scalar-op>
 */
static NPY_INLINE npy_uintp
mpy_blocked_end(const npy_uintp offset, const npy_uintp esize,
                const npy_uintp vsz, const npy_uintp nvals)
{
    return nvals - offset - (nvals - offset) % (vsz / esize);
}


/* byte swapping functions */
static NPY_INLINE npy_uint16
mpy_bswap2(npy_uint16 x)
{
    return ((x & 0xffu) << 8) | (x >> 8);
}

/*
 * treat as int16 and byteswap unaligned memory,
 * some cpus don't support unaligned access
 */
static NPY_INLINE void
mpy_bswap2_unaligned(char * x)
{
    char a = x[0];
    x[0] = x[1];
    x[1] = a;
}

static NPY_INLINE npy_uint32
mpy_bswap4(npy_uint32 x)
{
    return __builtin_bswap32(x);
}

static NPY_INLINE void
mpy_bswap4_unaligned(char * x)
{
    char a = x[0];
    x[0] = x[3];
    x[3] = a;
    a = x[1];
    x[1] = x[2];
    x[2] = a;
}

static NPY_INLINE npy_uint64
mpy_bswap8(npy_uint64 x)
{
    return __builtin_bswap64(x);
}

static NPY_INLINE void
mpy_bswap8_unaligned(char * x)
{
    char a = x[0]; x[0] = x[7]; x[7] = a;
    a = x[1]; x[1] = x[6]; x[6] = a;
    a = x[2]; x[2] = x[5]; x[5] = a;
    a = x[3]; x[3] = x[4]; x[4] = a;
}

#pragma omp end declare target

/* Start raw iteration */
#define NPY_RAW_ITER_START(idim, ndim, coord, shape) \
        memset((coord), 0, (ndim) * sizeof(coord[0])); \
        do {

/* Increment to the next n-dimensional coordinate for one raw array */
#define NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (data) -= ((shape)[idim] - 1) * (strides)[idim]; \
                } \
                else { \
                    (data) += (strides)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for two raw arrays */
#define NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, dataB, stridesB) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for three raw arrays */
#define NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for four raw arrays */
#define NPY_RAW_ITER_FOUR_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC, \
                              dataD, stridesD) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                    (dataD) -= ((shape)[idim] - 1) * (stridesD)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    (dataD) += (stridesD)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))


#endif