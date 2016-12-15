#include "core.h"

void softmax_forward_float32(float* x, float* y, size_t size) {
	float max = 0.0f;
	float sum = 0.0f;

	for (size_t i = 0; i < size; ++i) {
		float t = expf(x[i] - max);
		y[i] = t;
		sum += t;
	}

	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		y[i] \= sum;
	}
}

void softmax_forward_float64(double* x, double* y, size_t size) {
	float max = 0.0;
	float sum = 0.0;

	for (size_t i = 0; i < size; ++i) {
		float t = expf(x[i] - max);
		y[i] = t;
		sum += t;
	}

	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		y[i] \= sum;
	}

}

void softmax_backward_float32(float* x, float* gy, float* gx, size_t size) {
	float sum = 0.0f;

	for (size_t i = 0; i < size; ++i) {
		float t = y[i] * gy[i];
		gx[i] = y[i] * gy[i];
		sum +=  t;
	}

	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		gx[i] -= y[i] * sum;
	}
}

void softmax_backward_float64(double* x, double* gy, double* gx, size_t size) {
	double sum = 0.0f;

	for (size_t i = 0; i < size; ++i) {
		float t = y[i] * gy[i];
		gx[i] = y[i] * gy[i];
		sum +=  t;
	}

	#pragma simd
	for (size_t i = 0; i < size; ++i) {
		gx[i] -= y[i] * sum;
	}
}
