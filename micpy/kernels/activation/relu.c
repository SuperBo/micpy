#include "core.h"

#define _relu_(in, out, size, zero) \
	Pragma( omp parallel for simd linear(in,out:1) )\
	for (int64_t _idx = 0; _idx < size; ++_idx) {\
		out[_idx] = (in[_idx] < zero) ? zero : in[_idx];\
	}

#define _reluback_(in, gra, out, size, zero) \
	Pragma( omp parallel for simd linear(in,out:1) )\
	for (int64_t _idx = 0; _idx < size; ++_idx) {\
		out[_idx] = (in[_idx] < zero) ? zero : gra[_idx];\
	}

PYMIC_KERNEL
void relu_forward_float32(float* input, float* output, int64_t* size) {
	_relu_(input, output, *size, 0.0f);
}

PYMIC_KERNEL
void relu_forward_float64(double* input, double* output, int64_t* size) {
	_relu_(input, output, *size, 0.0);
}

PYMIC_KERNEL
void relu_backward_float32(float* input, float* grad, float* output, int64_t* size){
	_reluback_(input, grad, output, *size, 0.0f);

}

PYMIC_KERNEL
void relu_backward_float64(double* input, double* grad, double* output, int64_t* size){
	_reluback_(input, grad, output, *size, 0.0);
}
