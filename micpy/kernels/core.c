#include "core.h"

PYMIC_KERNEL
void normalize_coefficient(const int64_t* size, const int32_t* t, const int64_t* ignored_label, double* coeff) {
	int valid = 0;

	if (ignored_label) {
		#pragma omp simd aligned(t:64) reduction(+:valid)
		for (int64_t i = 0; i < *size; ++i) {
			valid += (t[i] != *ignored_label);
		}
	}
	else {
		valid = *size;
	}

	valid = _max_(valid, 1);

	*coeff = 1.0 / (double)valid;
}

PYMIC_KERNEL
void bin_predict_float32(const float* y, const double* threshold, int64_t* pred, const int64_t* size) {
	float thres = (float) *threshold;
	#pragma omp simd aligned(pred,y:64)
	for (int64_t i = 0; i < *size; ++i) {
		pred[i] = (y[i] > thres);
	}
}

PYMIC_KERNEL
void bin_predict_float64(const double* y, const double* threshold, int64_t* pred, const int64_t* size) {
	double thres = *threshold;
	#pragma omp simd aligned(pred,y:64)
	for (int64_t i = 0; i < *size; ++i) {
		pred[i] = (y[i] > thres);
	}
}

PYMIC_KERNEL
void accuracy(const int64_t* size, const int64_t* predict, const int32_t* t, const int64_t* ignored_label, double* accu) {
	int valid, match;
	valid = match = 0;

	if (ignored_label) {
		int tmp;
		#pragma omp simd aligned(predict,t:64) reduction(+:valid,match)
		for (int64_t i = 0; i < *size; ++i) {
			tmp = (t[i] != *ignored_label);
			valid += tmp;
			match += tmp & (t[i] == predict[i]);
		}
	}
	else {
		#pragma omp simd aligned(predict,t:64) reduction(+:match)
		for (int64_t i = 0; i < *size; ++i) {
			match += (t[i] == predict[i]);
		}
		valid = *size;
	}

	*accu = (valid > 0) ? (double) match / valid : 0.0;
}


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

PYMIC_KERNEL
void matrix_mul_transB(const int64_t *dtype, const int64_t *m, const int64_t *n, const int64_t *k,
					const void *x_, const void *y_, void *r_) {
	switch (*dtype) {
	case DTYPE_FLOAT32:
		{
		float* x = (float*) x_;
		float* y = (float*) y_;
		float* r = (float*) r_;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
						*m, *n, *k, 1.0f, x, *k, y, *k, 0.0f, r, *n);
		}
		break;
	case DTYPE_FLOAT64:
		{
		double* x = (double*) x_;
		double* y = (double*) y_;
		double* r = (double*) r_;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
						*m, *n, *k, 1.0, x, *k, y, *k, 0.0, r, *n);
		}
		break;
	}
}

PYMIC_KERNEL
void matrix_mul_transA(const int64_t *dtype, const int64_t *m, const int64_t *n, const int64_t *k,
					const void *x_, const void *y_, void *r_) {

	switch (*dtype) {
		case DTYPE_FLOAT32:
		{
			float* x = (float*) x_;
			float* y = (float*) y_;
			float* r = (float*) r_;
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
							*m, *n, *k, 1.0f, x, *m, y, *n, 0.0f, r, *n);
		}
		break;
		case DTYPE_FLOAT64:
		{
			double* x = (double*) x_;
			double* y = (double*) y_;
			double* r = (double*) r_;
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
							*m, *n, *k, 1.0, x, *m, y, *n, 0.0, r, *n);
		}
		break;
	}
}



PYMIC_KERNEL
void grad_decrease(const int64_t *dtype, const int64_t *n, const void *params, const int64_t *incp, const void *grad, const int64_t *incg, const double *lrate) {
	switch (*dtype) {
		case DTYPE_FLOAT32:
		{
			float alpha = -(*lrate);
			cblas_saxpy(*n, alpha, (float*)grad, *incg, (float*) params, *incp);
		}
		break;
		case DTYPE_FLOAT64:
		{
			double alpha = -(*lrate);
			cblas_daxpy(*n, alpha, (double*)grad, *incg, (double*)params, *incp);
		}
		break;
	}
}
