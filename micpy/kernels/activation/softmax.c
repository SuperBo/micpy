#include "core.h"

void flat_softmax_forward_float32(const float* x, float* c, float* sum, float* y, const int64_t* size) {
	float max = _max_fa_(x, *size);
	if (c) *c = max;
	float _sum = 0.0f;

	#pragma omp simd reduction(+:_sum)
	for (int64_t i = 0; i < *size; ++i) {
		float e_xm = expf(x[i] - max);
		y[i] = e_xm;
		_sum += e_xm;
	}

	if (sum) *sum = _sum;

	// Calculate output of softmax
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		y[i] /= _sum;
	}
}

void flat_softmax_forward_float64(const double* x, double* c, double* sum, double* y, const int64_t* size) {
	double max = _max_da_(x, *size);
	if (c) *c = max;
	double _sum = 0.0;

	#pragma omp simd reduction(+:_sum)
	for (int64_t i = 0; i < *size; ++i) {
		double e_xm = exp(x[i] - max);
		y[i] = e_xm;
		_sum += e_xm;
	}

	if (sum) *sum = _sum;

	// Calculate output of softmax
	#pragma omp simd
	for (int64_t i = 0; i < *size; ++i) {
		y[i] /= _sum;
	}
}

#define _call_softmax_(type, m, n, x, y)\
	for (int64_t i = 0; i < *m; ++i) {\
		size_t offset = (*n) * i;\
		flat_softmax_forward_ ## type(x + offset, NULL, NULL, y + offset, n);\
	}

PYMIC_KERNEL
void softmax_forward_float32(const float* x, float* y, const int64_t* m, const int64_t* n) {
	_call_softmax_(float32, m, n, x, y)
}

PYMIC_KERNEL
void softmax_forward_float64(const double* x, double* y, const int64_t* m, const int64_t* n) {
	_call_softmax_(float64, m, n, x, y)
}

void flat_softmax_backward_float32(const float* y, const float* gy, float* gx, const int64_t* size) {
	float sum = 0.0f;

	#pragma omp simd reduction(+:sum)
	for (int64_t i = 0; i < *size; ++i) {
		float t = y[i] * gy[i];
		gx[i] = t;
		sum +=  t;
	}

	cblas_saxpy(*size, -sum, y, 1, gx, 1);
}

void flat_softmax_backward_float64(const double* y, const double* gy, double* gx, const int64_t* size) {
	double sum = 0.0;

	#pragma omp simd reduction(+:sum)
	for (size_t i = 0; i < *size; ++i) {
		double t = y[i] * gy[i];
		gx[i] = y[i] * gy[i];
		sum +=  t;
	}

	cblas_daxpy(*size, -sum, y, 1, gx, 1);
}

#define _call_softmaxback_(type, m, n, y, gy, gx)\
	for (int64_t i = 0; i < *m; ++i) {\
		size_t offset = (*n) * i;\
		flat_softmax_backward_ ## type(y + offset, gy + offset, gx + offset, n);\
	}

PYMIC_KERNEL
void softmax_backward_float32(const float* y, const float* gy, float* gx, const int64_t* m, const int64_t* n) {
	_call_softmaxback_(float32, m, n, y, gy, gx)
}

PYMIC_KERNEL
void softmax_backward_float64(const double* y, const double* gy, double* gx, const int64_t* m, const int64_t* n) {
	_call_softmaxback_(float64, m, n, y, gy, gx)
}
