#ifndef _MPY_MATH_HELPER_
#define _MPY_MATH_HELPER_

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

#pragma omp declare target

#endif
