#ifndef __MPY_LOWLEVEL_STRIDED_LOOPS_H
#define __MPY_LOWLEVEL_STRIDED_LOOPS_H
#include <npy_config.h>


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

#endif