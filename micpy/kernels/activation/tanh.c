#include "core.h"

PYMIC_KERNEL
void tanh_forward_float32(float* input, float* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		output[i] = tanhf(input[i]);
	}
}

PYMIC_KERNEL
void tanh_forward_float64(double* input, double* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		output[i] = tanh(input[i]);
	}
}

PYMIC_KERNEL
void tanh_backward_float32(float* pre, float* grad, float* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		output[i] = grad[i] * (1.0f - pre[i] * pre[i]);
	}
}

PYMIC_KERNEL
void tanh_backward_float64(double* pre, double* grad, double* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		output[i] = grad[i] * (1.0 - pre[i] * pre[i]);
	}
}
