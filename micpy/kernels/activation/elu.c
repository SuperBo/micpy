#include "core.h"

#define _elu_(in, out, size, a, type, zero, exp_func) \
	Pragma( omp parallel for simd linear(in,out:1) )\
	for (size_t _i = 0; _i < size; ++_i){\
		type x = in[_i];\
		out[_i] = (x < zero) ? (a * (exp_func(x) - 1.0)) : x;\
	}

#define _eluback_(in, gra, out, size, a, type, zero, exp_func) \
	Pragma( omp parallel for simd linear(in,out,gra:1) )\
	for (size_t _i = 0; _i < size; ++_i){\
		type x = in[_i];\
		out[_i] = (x < zero) ? (gra[_i] * a * exp_func(x)) : gra[_i];\
	}

PYMIC_KERNEL
void elu_forward_float32(float* input, double* alpha, float* output, size_t* size){
	float a = (float) *alpha;
	_elu_(input, output, *size, a, float, 0.0f, expf);
}

PYMIC_KERNEL
void elu_forward_float64(double* input, double* alpha, double* output, size_t* size){
	_elu_(input, output, *size, *alpha,  double, 0.0, exp);
}

PYMIC_KERNEL
void elu_backward_float32(float* input, float* grad, double* alpha, float* output, size_t* size){
	float a = (float) *alpha;
	_eluback_(input, grad, output, *size, a, float, 0.0f, expf);
}

PYMIC_KERNEL
void elu_backward_float64(double* input, double* grad, double* alpha, double* output, size_t* size){
	_eluback_(input, grad, output, *size, *alpha, double, 0.0, exp);
}
