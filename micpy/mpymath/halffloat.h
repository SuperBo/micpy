#ifndef __MPY_HALFFLOAT_H__
#define __MPY_HALFFLOAT_H__

#include <numpy/npy_math.h>
#include <mpymath/mpy_math.h>

#pragma omp declare target

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Half-precision routines
 */

/* Conversions */
float mpy_half_to_float(npy_half h);
double mpy_half_to_double(npy_half h);
npy_half mpy_float_to_half(float f);
npy_half mpy_double_to_half(double d);
/* Comparisons */
int mpy_half_eq(npy_half h1, npy_half h2);
int mpy_half_ne(npy_half h1, npy_half h2);
int mpy_half_le(npy_half h1, npy_half h2);
int mpy_half_lt(npy_half h1, npy_half h2);
int mpy_half_ge(npy_half h1, npy_half h2);
int mpy_half_gt(npy_half h1, npy_half h2);
/* faster *_nonan variants for when you know h1 and h2 are not NaN */
int mpy_half_eq_nonan(npy_half h1, npy_half h2);
int mpy_half_lt_nonan(npy_half h1, npy_half h2);
int mpy_half_le_nonan(npy_half h1, npy_half h2);
/* Miscellaneous functions */
int mpy_half_iszero(npy_half h);
int mpy_half_isnan(npy_half h);
int mpy_half_isinf(npy_half h);
int mpy_half_isfinite(npy_half h);
int mpy_half_signbit(npy_half h);
npy_half mpy_half_copysign(npy_half x, npy_half y);
npy_half mpy_half_spacing(npy_half h);
npy_half mpy_half_nextafter(npy_half x, npy_half y);
npy_half mpy_half_divmod(npy_half x, npy_half y, npy_half *modulus);

/*
 * Half-precision constants
 */

#define NPY_HALF_ZERO   (0x0000u)
#define NPY_HALF_PZERO  (0x0000u)
#define NPY_HALF_NZERO  (0x8000u)
#define NPY_HALF_ONE    (0x3c00u)
#define NPY_HALF_NEGONE (0xbc00u)
#define NPY_HALF_PINF   (0x7c00u)
#define NPY_HALF_NINF   (0xfc00u)
#define NPY_HALF_NAN    (0x7e00u)

#define NPY_MAX_HALF    (0x7bffu)

/*
 * Bit-level conversions
 */

npy_uint16 mpy_floatbits_to_halfbits(npy_uint32 f);
npy_uint16 mpy_doublebits_to_halfbits(npy_uint64 d);
npy_uint32 mpy_halfbits_to_floatbits(npy_uint16 h);
npy_uint64 mpy_halfbits_to_doublebits(npy_uint16 h);

#ifdef __cplusplus
}
#endif

#pragma omp end declare target

#endif
