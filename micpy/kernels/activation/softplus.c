#include "core.h"

void softplus_forward_float32(float* x, float* y, size_t size, float alpha, float beta) {
	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		y[i] = alpha * logf(1.0f + expf(beta * x[i]));
	}
}

void softplus_forward_float64(double* x, double* y, size_t size, double alpha, double beta) {
	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		y[i] = alpha * log(1.0 + exp(beta * x[i]));
	}
}

void softplus_backward_float32(float* x, float* gy, float* gx, size_t size, float alpha, float beta) {
	float a = alpha * beta;
	float b = -beta;

	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		gx[i] = gy[i] * (a / (1.0f + expf(b * x[i])));
	}
}

void softplus_backward_float64(double* x, double* gy, double* gx, size_t size, double alpha, double beta){
	double a = alpha * beta;
	double b = -beta;

	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		gx[i] = gy[i] * (a / (1.0 + exp(b * x[i])));
	}
}
