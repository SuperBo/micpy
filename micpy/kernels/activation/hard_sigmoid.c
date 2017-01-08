#include "core.h"

#define _hsigmoid_(in, out, size, type, zero, one, a, b)\
	_Pragma("omp simd")\
	for (size_t _i = 0; _i < size; ++_i){\
		type v = in[_i] * a + b;\
		out[_i] = (v > zero) ? ((v < one) ? v : one) : zero ;\
	}

#define _hsigmoidback_(in, grad, out, size, zero, a, bound)\
	_Pragma("omp simd")\
	for (size_t _i = 0; _i < size; ++_i){\
		out[_i] = (in[_i] < -bound || in[_i] > bound) ? zero : grad[_i] * a;\
	}

PYMIC_KERNEL
void hardsigmoid_forward_float32(float* input, float* output, size_t* size){
	_hsigmoid_(input, output, *size, float, 0.0f, 1.0f, 0.2f, 0.5f);
}

PYMIC_KERNEL
void hardsigmoid_forward_float64(double* input, double* output, size_t* size){
	_hsigmoid_(input, output, *size, double, 0.0, 1.0, 0.2, 0.5);
}

PYMIC_KERNEL
void hardsigmoid_backward_float32(float* input, float* grad, float* output, size_t* size){
	_hsigmoidback_(input, grad, output, *size, 0.0f, 0.2f, 2.5f);
}

PYMIC_KERNEL
void hardsigmoid_backward_float64(double* input, double* grad, double* output, size_t* size){
	_hsigmoidback_(input, grad, output, *size, 0.0, 0.2, 2.5);
}
