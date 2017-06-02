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

#pragma omp declare target
/*
 * C99 double math funcs
 */
inline double mpy_sin(double x){
    return sin(x);
}
inline double mpy_cos(double x){
    return cos(x);
}
inline double mpy_tan(double x){
    return tan(x);
}
inline double mpy_sinh(double x){
    return sinh(x);
}
inline double mpy_cosh(double x){
    return cosh(x);
}
inline double mpy_tanh(double x){
    return tanh(x);
}

inline double mpy_asin(double x){
    return asin(x);
}
inline double mpy_acos(double x){
    return acos(x);
}
inline double mpy_atan(double x){
    return atan(x);
}

inline double mpy_log(double x){
    return log(x);
}
inline double mpy_log10(double x){
    return log10(x);
}
inline double mpy_exp(double x){
    return exp(x);
}
inline double mpy_sqrt(double x){
    return sqrt(x);
}
inline double mpy_cbrt(double x){
    return cbrt(x);
}

inline double mpy_fabs(double x){
    return fabs(x);
}
inline double mpy_ceil(double x){
    return ceil(x);
}
inline double mpy_fmod(double x, double y){
    return fmod(x, y);
}
inline double mpy_floor(double x){
    return floor(x);
}

inline double mpy_expm1(double x){
    return expm1(x);
}
inline double mpy_log1p(double x){
    return log1p(x);
}
inline double mpy_hypot(double x, double y){
    return hypot(x, y);
}
inline double mpy_acosh(double x){
    return acosh(x);
}
inline double mpy_asinh(double x){
    return asinh(x);
}
inline double mpy_atanh(double x){
    return atanh(x);
}
inline double mpy_rint(double x){
    return rint(x);
}
inline double mpy_trunc(double x){
    return trunc(x);
}
inline double mpy_exp2(double x){
    return exp2(x);
}
inline double mpy_log2(double x){
    return log2(x);
}

inline double mpy_atan2(double x, double y){
    return atan2(x, y);
}
inline double mpy_pow(double x, double y){
    return pow(x, y);
}
inline double mpy_modf(double x, double* y){
    return modf(x, y);
}
inline double mpy_frexp(double x, int* y){
    return frexp(x, y);
}
inline double mpy_ldexp(double n, int y){
    return ldexp(n, y);
}

inline double mpy_copysign(double x, double y){
    return copysign(x, y);
}
inline double mpy_nextafter(double x, double y){
    return nextafter(x, y);
}

/*
 * float C99 math functions
 */
inline float mpy_sinf(float x){
    return sinf(x);
}
inline float mpy_cosf(float x){
    return cosf(x);
}
inline float mpy_tanf(float x){
    return tanf(x);
}
inline float mpy_sinhf(float x){
    return sinhf(x);
}
inline float mpy_coshf(float x){
    return coshf(x);
}
inline float mpy_tanhf(float x){
    return tanhf(x);
}
inline float mpy_fabsf(float x){
    return fabsf(x);
}
inline float mpy_floorf(float x){
    return floorf(x);
}
inline float mpy_ceilf(float x){
    return ceilf(x);
}
inline float mpy_rintf(float x){
    return rintf(x);
}
inline float mpy_truncf(float x){
    return truncf(x);
}
inline float mpy_sqrtf(float x){
    return sqrtf(x);
}
inline float mpy_cbrtf(float x){
    return cbrtf(x);
}
inline float mpy_log10f(float x){
    return log10f(x);
}
inline float mpy_logf(float x){
    return logf(x);
}
inline float mpy_expf(float x){
    return expf(x);
}
inline float mpy_expm1f(float x){
    return expm1f(x);
}
inline float mpy_asinf(float x){
    return asinf(x);
}
inline float mpy_acosf(float x){
    return acosf(x);
}
inline float mpy_atanf(float x){
    return atanf(x);
}
inline float mpy_asinhf(float x){
    return asinhf(x);
}
inline float mpy_acoshf(float x){
    return acoshf(x);
}
inline float mpy_atanhf(float x){
    return atanhf(x);
}
inline float mpy_log1pf(float x){
    return log1pf(x);
}
inline float mpy_exp2f(float x){
    return exp2f(x);
}
inline float mpy_log2f(float x){
    return log2f(x);
}

inline float mpy_atan2f(float x, float y){
    return atan2f(x, y);
}
inline float mpy_hypotf(float x, float y){
    return hypotf(x, y);
}
inline float mpy_powf(float x, float y){
    return powf(x, y);
}
inline float mpy_fmodf(float x, float y){
    return fmodf(x, y);
}

inline float mpy_modff(float x, float* y){
    return modff(x, y);
}
inline float mpy_frexpf(float x, int* y){
    return frexpf(x, y);
}
inline float mpy_ldexpf(float x, int y){
    return ldexpf(x, y);
}

inline float mpy_copysignf(float x, float y){
    return copysignf(x, y);
}
inline float mpy_nextafterf(float x, float y){
    return nextafterf(x, y);
}

/*
 * long double C99 math functions
 */
/*
 * Complex declarations
 */
inline npy_longdouble mpy_sinl(npy_longdouble x){
    return sinl(x);
}
inline npy_longdouble mpy_cosl(npy_longdouble x){
    return cosl(x);
}
inline npy_longdouble mpy_tanl(npy_longdouble x){
    return tanl(x);
}
inline npy_longdouble mpy_sinhl(npy_longdouble x){
    return sinhl(x);
}
inline npy_longdouble mpy_coshl(npy_longdouble x){
    return coshl(x);
}
inline npy_longdouble mpy_tanhl(npy_longdouble x){
    return tanhl(x);
}
inline npy_longdouble mpy_fabsl(npy_longdouble x){
    return fabsl(x);
}
inline npy_longdouble mpy_floorl(npy_longdouble x){
    return floorl(x);
}
inline npy_longdouble mpy_ceill(npy_longdouble x){
    return ceill(x);
}
inline npy_longdouble mpy_rintl(npy_longdouble x){
    return rintl(x);
}
inline npy_longdouble mpy_truncl(npy_longdouble x){
    return truncl(x);
}
inline npy_longdouble mpy_sqrtl(npy_longdouble x){
    return sqrtl(x);
}
inline npy_longdouble mpy_cbrtl(npy_longdouble x){
    return cbrtl(x);
}
inline npy_longdouble mpy_log10l(npy_longdouble x){
    return log10l(x);
}
inline npy_longdouble mpy_logl(npy_longdouble x){
    return logl(x);
}
inline npy_longdouble mpy_expl(npy_longdouble x){
    return expl(x);
}
inline npy_longdouble mpy_expm1l(npy_longdouble x){
    return expm1l(x);
}
inline npy_longdouble mpy_asinl(npy_longdouble x){
    return asinl(x);
}
inline npy_longdouble mpy_acosl(npy_longdouble x){
    return acosl(x);
}
inline npy_longdouble mpy_atanl(npy_longdouble x){
    return atanl(x);
}
inline npy_longdouble mpy_asinhl(npy_longdouble x){
    return asinhl(x);
}
inline npy_longdouble mpy_acoshl(npy_longdouble x){
    return acoshl(x);
}
inline npy_longdouble mpy_atanhl(npy_longdouble x){
    return atanhl(x);
}
inline npy_longdouble mpy_log1pl(npy_longdouble x){
    return log1pl(x);
}
inline npy_longdouble mpy_exp2l(npy_longdouble x){
    return exp2l(x);
}
inline npy_longdouble mpy_log2l(npy_longdouble x){
    return log2l(x);
}

inline npy_longdouble mpy_atan2l(npy_longdouble x, npy_longdouble y){
    return atan2l(x, y);
}
inline npy_longdouble mpy_hypotl(npy_longdouble x, npy_longdouble y){
    return hypotl(x, y);
}
inline npy_longdouble mpy_powl(npy_longdouble x, npy_longdouble y){
    return powl(x, y);
}
inline npy_longdouble mpy_fmodl(npy_longdouble x, npy_longdouble y){
    return fmodl(x, y);
}

inline npy_longdouble mpy_modfl(npy_longdouble x, npy_longdouble* y){
    return modfl(x, y);
}
inline npy_longdouble mpy_frexpl(npy_longdouble x, int* y){
    return frexpl(x, y);
}
inline npy_longdouble mpy_ldexpl(npy_longdouble x, int y){
    return ldexpl(x, y);
}

inline npy_longdouble mpy_copysignl(npy_longdouble x, npy_longdouble y){
    return copysignl(x, y);
}
inline npy_longdouble mpy_nextafterl(npy_longdouble x, npy_longdouble y){
    return nextafterl(x, y);
}

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
