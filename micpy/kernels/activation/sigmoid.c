#include "core.h"

PYMIC_KERNEL
void sigmoid_forward_float32(float* input, float* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; ++i) {
		output[i] = 1.0f / (1.0f + expf(-input[i]));
	}
}

PYMIC_KERNEL
void sigmoid_forward_float64(double* input, double* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; i++) {
		output[i] = 1.0 / (1.0 + exp(-input[i]));
	}
}

PYMIC_KERNEL
void sigmoid_backward_float32(float* x, float* grad, float* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; i++) {
		output[i] = grad[i] * x[i] * (1.0f - x[i]);
	}
}

PYMIC_KERNEL
void sigmoid_backward_float64(double* x, double* grad, double* output, size_t* size) {
	#pragma omp simd
	for (size_t i = 0; i < *size; i++) {
		output[i] = grad[i] * x[i] * (1.0 - x[i]);
	}
}
