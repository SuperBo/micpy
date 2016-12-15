#include "core.h"

#define _elu_(in, out, size, a, type, zero, exp_func) \
	_Pragma("simd")\
	for (size_t _idx, _idx < size; ++_idx){\
		type x = in[_idx];\
		out[_idx] = (x < zero) ? (a * (exp_func(x) - 1.0)) : x;\
	}

#define _eluback_(in, gra, out, size, a, type, zero, exp_func) \
	_Pragma("simd")\
	for (size_t _idx, _idx < size; ++_idx){\
		type x = in[_idx];\
		out[_idx] = (x < zero) ? (gra[_idx] * a * exp_func(x)) : gra[_idx];\
	}

void elu_forward_float32(float* input, float* output, size_t size, double alpha){
	_elu_(input, output, size, alpha, float, 0.0f, expf);
}

void elu_forward_float64(double* input, double* output, size_t size, double alpha){
	_elu_(input, output, size, alpha,  double, 0.0, exp);
}

void elu_backward_float32(float* input, float* grad, float* ouput, size_t size, double alpha){
	_eluback_(input, grad, output, size, alpha, float, 0.0f, expf);
}

void elu_backward_float64(double* input, double* grad, double* output, size_t size, double alpha){
	_eluback_(input, grad, output, size, alpha, double, 0.0, exp);
}
