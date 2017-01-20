#include "core.h"

#define _leakyrelu_(in, out, size, slope, zero) \
	Pragma( omp parallel for simd linear(in,out:1) )\
	for (size_t _idx = 0; _idx < size; _idx++) {\
		out[_idx] = (in[_idx] < zero) ? in[_idx] * slope : in[_idx];\
	}

#define _leakyreluback_(x, grad, out, size, slope, zero) \
	_Pragma( omp parallel for simd linear(in,out,grad:1) )\
	for (size_t _idx = 0; _idx < size; _idx++) {\
		out[_idx] = (x[_idx] < zero) ? grad[_idx] * slope : grad[_idx];\
	}


PYMIC_KERNEL
void leakyrelu_forward_float32(float* input, double* slope, float* output, size_t* size) {
	float _slope = (float) *slope;
	_leakyrelu_(input, output, (*size), _slope, 0.0f);
}

PYMIC_KERNEL
void leakyrelu_forward_float64(double* input, double* slope, double* output, size_t* size) {
	_leakyrelu_(input, output, (*size), (*slope), 0.0);
}

PYMIC_KERNEL
void leakyrelu_backward_float32(float* input, float* grad, double* slope, float* output, size_t* size) {
	float _slope = (float) *slope;
	_leakyreluback_(input, grad, output, (*size), _slope, 0.0f);
}

PYMIC_KERNEL
void leakyrelu_backward_float64(double* input, double* grad, double* slope, double* output, size_t* size) {
	_leakyreluback_(input, grad, output, (*size), (*slope), 0.0);
}
