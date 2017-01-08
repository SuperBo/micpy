#include "core.h"

PYMIC_KERNEL
void crelu_forward_float32(float* x_in, float* y_out, size_t* size){
}

PYMIC_KERNEL
void crelu_forward_float64(double* x_in, double* y_out, size_t* size){
}

PYMIC_KERNEL
void crelu_backward_float32(float* input, float* grad, float* output, size_t* size){
}

PYMIC_KERNEL
void crelu_backward_float64(double* input, double* grad, double* output, size_t* size){
}
