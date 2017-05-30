#ifndef _MPY_MATH_HELPER_
#define _MPY_MATH_HELPER_

#include "halffloat.h"

#ifndef INFINITY
static const union { npy_uint32 __i; float __f;} __binff = {0x7f800000UL};
#define INFINITY (__binff.__f)
#endif

#ifndef NAN
static const union { npy_uint32 __i; float __f;} __bnanf = {0x7fc00000UL};
#define NAN (__bnanf.__f)
#endif

#ifndef PZERO
static const union { npy_uint32 __i; float __f;} __bpzerof = {0x00000000UL};
#define PZERO (__bpzerof.__f)
#endif

#ifndef NZERO
static const union { npy_uint32 __i; float __f;} __bnzerof = {0x80000000UL};
#define NZERO (__bnzerof.__f)
#endif

#define MPY_INFINITYF INFINITY
#define MPY_NANF NAN
#define MPY_PZEROF PZERO
#define MPY_NZEROF NZERO

#define MPY_INFINITY ((npy_double)MPY_INFINITYF)
#define MPY_NAN ((npy_double)MPY_NANF)
#define MPY_PZERO ((npy_double)MPY_PZEROF)
#define MPY_NZERO ((npy_double)MPY_NZEROF)

#define MPY_INFINITYL ((npy_longdouble)MPY_INFINITYF)
#define MPY_NANL ((npy_longdouble)MPY_NANF)
#define MPY_PZEROL ((npy_longdouble)MPY_PZEROF)
#define MPY_NZEROL ((npy_longdouble)MPY_NZEROF)

/*
 * C99 double math funcs
 */
#define mpy_sin sin
#define mpy_cos cos
#define mpy_tan tan
#define mpy_sinh sinh
#define mpy_cosh cosh
#define mpy_tanh tanh

#define mpy_asin asin
#define mpy_acos acos
#define mpy_atan atan

#define mpy_log log
#define mpy_log10 log10
#define mpy_exp exp
#define mpy_sqrt sqrt
#define mpy_cbrt cbrt

#define mpy_fabs fabs
#define mpy_ceil ceil
#define mpy_fmod fmod
#define mpy_floor floor

#define mpy_expm1 expm1
#define mpy_log1p log1p
#define mpy_hypot hypot
#define mpy_acosh acosh
#define mpy_asinh asinh
#define mpy_atanh atanh
#define mpy_rint rint
#define mpy_trunc trunc
#define mpy_exp2 exp2
#define mpy_log2 log2

#define mpy_atan2 atan2
#define mpy_pow pow
#define mpy_modf modf
#define mpy_frexp frexp
#define mpy_ldexp ldexp

#define mpy_copysign copysign
#define mpy_nextafter nextafter

/*
 * float C99 math functions
 */
#define mpy_sinf sinf
#define mpy_cosf cosf
#define mpy_tanf tanf
#define mpy_sinhf sinhf
#define mpy_coshf coshf
#define mpy_tanhf tanhf
#define mpy_fabsf fabsf
#define mpy_floorf floorf
#define mpy_ceilf ceilf
#define mpy_rintf rintf
#define mpy_truncf truncf
#define mpy_sqrtf sqrtf
#define mpy_cbrtf cbrtf
#define mpy_log10f log10f
#define mpy_logf logf
#define mpy_expf expf
#define mpy_expm1f expm1f
#define mpy_asinf asinf
#define mpy_acosf acosf
#define mpy_atanf atanf
#define mpy_asinhf asinhf
#define mpy_acoshf acoshf
#define mpy_atanhf atanhf
#define mpy_log1pf log1pf
#define mpy_exp2f exp2f
#define mpy_log2f log2f

#define mpy_atan2f atan2f
#define mpy_hypotf hypotf
#define mpy_powf powf
#define mpy_fmodf fmodf

#define mpy_modff modff
#define mpy_frexpf frexpf
#define mpy_ldexpf ldexpf

#define mpy_copysignf copysignf
#define mpy_nextafterf nextafterf

/*
 * long double C99 math functions
 */
#define mpy_sinl sinl
#define mpy_cosl cosl
#define mpy_tanl tanl
#define mpy_sinhl sinhl
#define mpy_coshl coshl
#define mpy_tanhl tanhl
#define mpy_fabsl fabsl
#define mpy_floorl floorl
#define mpy_ceill ceill
#define mpy_rintl rintl
#define mpy_truncl truncl
#define mpy_sqrtl sqrtl
#define mpy_cbrtl cbrtl
#define mpy_log10l log10l
#define mpy_logl logl
#define mpy_expl expl
#define mpy_expm1l expm1l
#define mpy_asinl asinl
#define mpy_acosl acosl
#define mpy_atanl atanl
#define mpy_asinhl asinhl
#define mpy_acoshl acoshl
#define mpy_atanhl atanhl
#define mpy_log1pl log1pl
#define mpy_exp2l exp2l
#define mpy_log2l log2l

#define mpy_atan2l atan2l
#define mpy_hypotl hypotl
#define mpy_powl powl
#define mpy_fmodl fmodl

#define mpy_modfl modfl
#define mpy_frexpl frexpl
#define mpy_ldexpl ldexpl

#define mpy_copysignl copysignl
#define mpy_nextafterl nextafterl

#pragma omp declare target
/*
 * Complex declarations
 */

/*
 * C99 specifies that complex numbers have the same representation as
 * an array of two elements, where the first element is the real part
 * and the second element is the imaginary part.
 */
#define __NPY_CPACK_IMP(x, y, type, ctype)   \
    union {                                  \
        ctype z;                             \
        type a[2];                           \
    } z1;;                                   \
                                             \
    z1.a[0] = (x);                           \
    z1.a[1] = (y);                           \
                                             \
    return z1.z;

static NPY_INLINE npy_cdouble mpy_cpack(double x, double y)
{
    __NPY_CPACK_IMP(x, y, double, npy_cdouble);
}

static NPY_INLINE npy_cfloat mpy_cpackf(float x, float y)
{
    __NPY_CPACK_IMP(x, y, float, npy_cfloat);
}

static NPY_INLINE npy_clongdouble mpy_cpackl(npy_longdouble x, npy_longdouble y)
{
    __NPY_CPACK_IMP(x, y, npy_longdouble, npy_clongdouble);
}
#undef __NPY_CPACK_IMP

/*
 * Same remark as above, but in the other direction: extract first/second
 * member of complex number, assuming a C99-compatible representation
 *
 * Those are defineds as static inline, and such as a reasonable compiler would
 * most likely compile this to one or two instructions (on CISC at least)
 */
#define __NPY_CEXTRACT_IMP(z, index, type, ctype)   \
    union {                                         \
        ctype z;                                    \
        type a[2];                                  \
    } __z_repr;                                     \
    __z_repr.z = z;                                 \
                                                    \
    return __z_repr.a[index];

static NPY_INLINE double mpy_creal(npy_cdouble z)
{
    __NPY_CEXTRACT_IMP(z, 0, double, npy_cdouble);
}

static NPY_INLINE double mpy_cimag(npy_cdouble z)
{
    __NPY_CEXTRACT_IMP(z, 1, double, npy_cdouble);
}

static NPY_INLINE float mpy_crealf(npy_cfloat z)
{
    __NPY_CEXTRACT_IMP(z, 0, float, npy_cfloat);
}

static NPY_INLINE float mpy_cimagf(npy_cfloat z)
{
    __NPY_CEXTRACT_IMP(z, 1, float, npy_cfloat);
}

static NPY_INLINE npy_longdouble mpy_creall(npy_clongdouble z)
{
    __NPY_CEXTRACT_IMP(z, 0, npy_longdouble, npy_clongdouble);
}

static NPY_INLINE npy_longdouble mpy_cimagl(npy_clongdouble z)
{
    __NPY_CEXTRACT_IMP(z, 1, npy_longdouble, npy_clongdouble);
}
#undef __NPY_CEXTRACT_IMP

/*
 * Double precision complex functions
 */
double mpy_cabs(npy_cdouble z);
double mpy_carg(npy_cdouble z);

npy_cdouble mpy_cexp(npy_cdouble z);
npy_cdouble mpy_clog(npy_cdouble z);
npy_cdouble mpy_cpow(npy_cdouble x, npy_cdouble y);

npy_cdouble mpy_csqrt(npy_cdouble z);

npy_cdouble mpy_ccos(npy_cdouble z);
npy_cdouble mpy_csin(npy_cdouble z);
npy_cdouble mpy_ctan(npy_cdouble z);

npy_cdouble mpy_ccosh(npy_cdouble z);
npy_cdouble mpy_csinh(npy_cdouble z);
npy_cdouble mpy_ctanh(npy_cdouble z);

npy_cdouble mpy_cacos(npy_cdouble z);
npy_cdouble mpy_casin(npy_cdouble z);
npy_cdouble mpy_catan(npy_cdouble z);

npy_cdouble mpy_cacosh(npy_cdouble z);
npy_cdouble mpy_casinh(npy_cdouble z);
npy_cdouble mpy_catanh(npy_cdouble z);

/*
 * Single precision complex functions
 */
float mpy_cabsf(npy_cfloat z);
float mpy_cargf(npy_cfloat z);

npy_cfloat mpy_cexpf(npy_cfloat z);
npy_cfloat mpy_clogf(npy_cfloat z);
npy_cfloat mpy_cpowf(npy_cfloat x, npy_cfloat y);

npy_cfloat mpy_csqrtf(npy_cfloat z);

npy_cfloat mpy_ccosf(npy_cfloat z);
npy_cfloat mpy_csinf(npy_cfloat z);
npy_cfloat mpy_ctanf(npy_cfloat z);

npy_cfloat mpy_ccoshf(npy_cfloat z);
npy_cfloat mpy_csinhf(npy_cfloat z);
npy_cfloat mpy_ctanhf(npy_cfloat z);

npy_cfloat mpy_cacosf(npy_cfloat z);
npy_cfloat mpy_casinf(npy_cfloat z);
npy_cfloat mpy_catanf(npy_cfloat z);

npy_cfloat mpy_cacoshf(npy_cfloat z);
npy_cfloat mpy_casinhf(npy_cfloat z);
npy_cfloat mpy_catanhf(npy_cfloat z);

/*
 * Extended precision complex functions
 */
npy_longdouble mpy_cabsl(npy_clongdouble z);
npy_longdouble mpy_cargl(npy_clongdouble z);

npy_clongdouble mpy_cexpl(npy_clongdouble z);
npy_clongdouble mpy_clogl(npy_clongdouble z);
npy_clongdouble mpy_cpowl(npy_clongdouble x, npy_clongdouble y);

npy_clongdouble mpy_csqrtl(npy_clongdouble z);

npy_clongdouble mpy_ccosl(npy_clongdouble z);
npy_clongdouble mpy_csinl(npy_clongdouble z);
npy_clongdouble mpy_ctanl(npy_clongdouble z);

npy_clongdouble mpy_ccoshl(npy_clongdouble z);
npy_clongdouble mpy_csinhl(npy_clongdouble z);
npy_clongdouble mpy_ctanhl(npy_clongdouble z);

npy_clongdouble mpy_cacosl(npy_clongdouble z);
npy_clongdouble mpy_casinl(npy_clongdouble z);
npy_clongdouble mpy_catanl(npy_clongdouble z);

npy_clongdouble mpy_cacoshl(npy_clongdouble z);
npy_clongdouble mpy_casinhl(npy_clongdouble z);
npy_clongdouble mpy_catanhl(npy_clongdouble z);


/*
 * platform-dependent code translates floating point
 * status to an integer sum of these values
 */

int mpy_get_floatstatus(void);
int mpy_clear_floatstatus(void);
void mpy_set_floatstatus_divbyzero(void);
void mpy_set_floatstatus_overflow(void);
void mpy_set_floatstatus_underflow(void);
void mpy_set_floatstatus_invalid(void);

npy_float mpy_spacingf(npy_float x);
npy_double mpy_spacing(npy_double x);
npy_longdouble mpy_spacingl(npy_longdouble x);

#pragma omp end declare target

#endif
