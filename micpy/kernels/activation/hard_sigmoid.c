#include "core.h"

void hardsigmoid_forward_float32(float* input, float* output, size_t size){
	#pragma simd
	for (size_t i = 0; i < size; ++i){
		float val = input[i] * 0.2f + 0.5f;
		output[i] = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
	}
}

void hardsigmoid_forward_float64(double* input, double* output, size_t size){
	#pragma simd
	for (size_t i = 0; i < size; ++i){
		double val = input[i] * 0.2 + 0.5;
		output[i] = (val < 0.0) ? 0.0 : ((val > 1.0) ? 1.0 : val);
	}

}

void hardsigmoid_backward_float32(float* input, float* grad, float* output, size_t size){
	#pragma simd
	for (size_t i = 0; i < size; ++i){
		output[i] = (intput[i] < 2.5f && input[i] > 2.5f) ? 0.0f : grad[i] * 0.2f;
	}
}

void hardsigmoid_backward_float64(double* input, double* grad, double* output, size_t size){
	#pragma simd
	for (size_t i = 0; i < size; ++i){
		output[i] = (intput[i] < 2.5 && input[i] > 2.5) ? 0.0 : grad[i] * 0.2;
	}
}
