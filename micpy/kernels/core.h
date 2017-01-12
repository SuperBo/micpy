#include <mathimf.h>
#include <pymic_kernel.h>
#include <mkl.h>

/* Data types, needs to match _data_type_map in _misc.py */
#define DTYPE_INT32     0
#define DTYPE_INT64     1
#define DTYPE_FLOAT32   2
#define DTYPE_FLOAT64   3
#define DTYPE_COMPLEX   4
#define DTYPE_UINT64    5

#define _max_(x, y) (x < y) ? (y) : (x)

#define _min_(x, y) (x < y) ? (x) : (y)

#define _max_fa_(a, n) a[cblas_isamax(n, a, 1)]

#define _max_da_(a, n) a[cblas_idamax(n, a, 1)]

float smax(const int64_t n, float *x, const int64_t incx);
int64_t ismax(const int64_t n, float *x, const int64_t incx);
double dmax(const int64_t n, double *x, const int64_t incx);
int64_t idmax(const int64_t n, double *x, const int64_t incx);

float smin(const int64_t n, float *x, const int64_t incx);
int64_t ismin(const int64_t n, float *x, const int64_t incx);
double dmin(const int64_t n, double *x, const int64_t incx);
int64_t idmin(const int64_t n, double *x, const int64_t incx);


void flat_softmax_forward_float32(const float* x, float* c, float* sum, float* y, const int64_t* n);
void flat_softmax_forward_float64(const double* x, double* c, double* sum, double* y, const int64_t* n);
void softmax_forward_float32(const float* x, float* y, const int64_t* m, const int64_t* n);
void softmax_forward_float64(const double* x, double* y, const int64_t* m, const int64_t* n);
