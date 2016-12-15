#include "core.h"

void clipped_relu_forward_float32(float* input, float* output, size_t size, float z) {
#pragma simd
	for (size_t i = 0; i < size; ++i) {
		float x = input[i];
		output[i] = _min_(_max_(x, 0.0f), z);
	}
}

void clipped_relu_forward_float64(double* input, double* output, size_t size) {
#pragma simd
	for (size_t i = 0; i < size; ++i) {
		double x = input[i];
		output[i] = _min_(_max_(x, 0.0), z);
	}
}

void clipped_relu_backward_float32(float* input, float* grad, float* output, size_t size, float z) {
#pragma simd
	for (size_t i = 0; i < size; ++i) {
		float x = input[i];
		output[i] = (x > 0.0f && x < z) ? grad[i] : 0.0f;
	}
}

void clipped_relu_backward_float64(double* input, double* grad, double* output, size_t size, float z) {
#pragma simd
	for (size_t i = 0; i < size; ++i) {
		double x = input[i];
		output[i] = (x > 0.0 && x < 0.0) ? grad[i] : 0.0;
	}
}
