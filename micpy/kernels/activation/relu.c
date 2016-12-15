#include "core.h"

#define _relu_(in, out, size, zero) \
	_Pragma("simd")\
	for (size_t _idx; _idx < size; ++_idx) {\
		out[_idx] = (in[_idx] < zero) ? zero : in[_idx];\
	}

#define _reluback_(in, gra, out, size, zero) \
	_Pragma("simd")\
	for (size_t _idx; _idx < size; ++_idx) {\
		out[_idx] = (in[_idx] < zero) ? zero : gra[_idx];\
	}

void relu_forward_float32(float* input, float* output, size_t size) {
	_relu_(input, output, size, 0.0f);
}

void relu_forward_float64(double* input, double* output, size_t size) {
	_relu_(input, output, size, 0.0);
}

void relu_backward_float32(float* input, float* grad, float* output, size_t size){
	_reluback_(input, grad, output, size, 0.0f);

}
void relu_backward_float64(double* input, double* grad, double* output, size_t size){
	_reluback_(input, grad, output, size, 0.0);
}
