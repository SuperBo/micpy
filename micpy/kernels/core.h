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


void flat_softmax_forward_float32(const float* x, float* c, float* sum, float* y, const int64_t* n);
void flat_softmax_forward_float64(const double* x, double* c, double* sum, double* y, const int64_t* n);
void softmax_forward_float32(const float* x, float* y, const int64_t* m, const int64_t* n);
void softmax_forward_float64(const double* x, double* y, const int64_t* m, const int64_t* n);
