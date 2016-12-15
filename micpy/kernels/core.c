#include "core.h"

PYMIC_KERNEL
void vector_dot(const int64_t *dtype, const int64_t *n,
						const void *x_, const int64_t *incx,
						const void *y_, const int64_t *incy,
						void *r_, const int64_t *incr) {
	switch (*dtype) {
	case DTYPE_FLOAT32:
		{
		float* x = (float*) x_;
		float* y = (float*) y_;
		float* r = (float*) r_;
		*r = cblas_sdot(*n, x, *incx, y, *incy);
		}
		break;
	case DTYPE_FLOAT64:
		{
		double* x = (double*) x_;
		double* y = (double*) y_;
		double* r = (double*) r_;
		*r = cblas_ddot(*n, x, *incx, y, *incy);
		}
		break;
	}
}

PYMIC_KERNEL
void matrix_mul(const int64_t *dtype, const int64_t *m, const int64_t *n, const int64_t *k,
					const void *x_, const void *y_, void *r_) {
	switch (*dtype) {
	case DTYPE_FLOAT32:
		{
		float* x = (float*) x_;
		float* y = (float*) y_;
		float* r = (float*) r_;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						*m, *n, *k, 1.0f, x, *k, y, *n, 0.0f, r, *n);
		}
		break;
	case DTYPE_FLOAT64:
		{
		double* x = (double*) x_;
		double* y = (double*) y_;
		double* r = (double*) r_;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						*m, *n, *k, 1.0, x, *k, y, *n, 0.0, r, *n);
		}
		break;
	}
}
