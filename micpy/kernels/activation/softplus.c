#include "core.h"

PYMIC_KERNEL
void softplus_forward_float32(float* x, double* alpha, double* beta, float* y, size_t *size) {
	float _alpha = (float) *alpha;
	float _beta = (float) *beta;
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		y[i] = _alpha * logf(1.0f + expf(_beta * x[i]));
	}
}

PYMIC_KERNEL
void softplus_forward_float64(double* x, double* alpha, double* beta, double* y, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		y[i] = *alpha * log(1.0 + exp(*beta * x[i]));
	}
}

PYMIC_KERNEL
void softplus_backward_float32(float* x, float* gy, double* alpha, double* beta, float* gx, size_t* size) {
	float a = (float) ((*alpha) * (*beta));
	float b = (float) -(*beta);

	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		gx[i] = gy[i] * (a / (1.0f + expf(b * x[i])));
	}
}

PYMIC_KERNEL
void softplus_backward_float64(double* x, double* gy, double* alpha, double* beta, double* gx, size_t* size){
	double a = (*alpha) * (*beta);
	double b = -(*beta);

	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		gx[i] = gy[i] * (a / (1.0 + exp(b * x[i])));
	}
}
