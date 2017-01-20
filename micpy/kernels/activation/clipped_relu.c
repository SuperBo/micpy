#include "core.h"

PYMIC_KERNEL
void clipped_relu_forward_float32(float* input, float* z, float* output, size_t* size) {
	#pragma omp parallel for simd linear(input,output:1)
	for (size_t i = 0; i < *size; ++i) {
		float x = input[i];
		output[i] = _min_(_max_(x, 0.0f), *z);
	}
}

PYMIC_KERNEL
void clipped_relu_forward_float64(double* input, double* z, double* output, size_t* size) {
	#pragma omp parallel for simd linear(input,output:1)
	for (size_t i = 0; i < *size; ++i) {
		double x = input[i];
		output[i] = _min_(_max_(x, 0.0), *z);
	}
}

PYMIC_KERNEL
void clipped_relu_backward_float32(float* input, float* grad, float* z, float* output, size_t* size) {
	#pragma omp simd parallel for simd linear(input,output,grad:1)
	for (size_t i = 0; i < *size; ++i) {
		float x = input[i];
		output[i] = (x > 0.0f && x < *z) ? grad[i] : 0.0f;
	}
}

PYMIC_KERNEL
void clipped_relu_backward_float64(double* input, double* grad, double* z, double* output, size_t* size) {
	#pragma omp simd parallel for simd linear(input,output,grad:1)
	for (size_t i = 0; i < *size; ++i) {
		double x = input[i];
		output[i] = (x > 0.0 && x < *z) ? grad[i] : 0.0;
	}
}
