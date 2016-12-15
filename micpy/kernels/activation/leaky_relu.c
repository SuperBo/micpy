#include "core.h"

#define _leakyrelu_(in, out, size, slope, zero) \
	_Pragma("simd")\
	for (size_t _idx; _idx < size; _idx++) {\
		out[_idx] = (in[_idx] < zero) ? in[_idx] * slope : in[_idx];\
	}

#define _leakyreluback_(x, grad, out, size, zero) \
	_Pragma("simd")\
	for (size_t _idx; _idx < size; _idx++) {\
		out[_idx] = (x[_idx] < zero) ? grad[_idx] * slope : grad[_idx];\
	}


void leakyrelu_forward_float32(float* input, float* output, size_t size, float slope) {
	_leakyrelu_(input, output, size, slope, 0.0f);
}

void leakyrelu_forward_float64(double* input, double* output, size_t size, double slope) {
	_leakyrelu_(input, output, size, slope, 0.0);
}

void leakyrelu_backward_float32(float* input, float* grad, float* output, size_t size, float slope) {
	_leakyreluback_(input, grad, output, size, slope, 0.0f);
}

void leakyrelu_backward_float32(double* input, double* grad, double* output, size_t size, double slope) {
	_leakyreluback_(input, grad, output, size, slope, 0.0);
}
