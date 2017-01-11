#include "core.h"

PYMIC_KERNEL
void sum(const int64_t* dtype, const void* a, const int64_t* niter, const int64_t* inciter, const int64_t* n, const int64_t* incn, void* out) {
	int64_t i, j;
	int64_t step = *incn;
	switch (*dtype) {
		case DTYPE_FLOAT32:
		{
			float* x = (float*) a;
			float* sum = (float*) out;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				float s = 0.0f;
				#pragma omp simd linear(x:step) reduction(+:s)
				for (j = 0; j < *n; j += step) {
					s += x[j];
				}
				sum[i] = s;
			}
		}
		break;
		case DTYPE_FLOAT64:
		{
			double* x = (double*) a;
			double* sum = (double*) out;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				double s = 0.0;
				#pragma omp simd linear(x:step) reduction(+:s)
				for (j = 0; j < *n; j += step) {
					s += x[j];
				}
				sum[i] = s;
			}
		}
		break;
		case DTYPE_INT32:
		{
			int* x = (int*) a;
			int* sum = (int*) out;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				int s = 0;
				#pragma omp simd linear(x:step) reduction(+:s)
				for (j = 0; j < *n; j += step) {
					s += x[j];
				}
				sum[i] = s;
			}
		}
		break;
		case DTYPE_INT64:
		{
			long* x = (long*) a;
			long* sum = (long*) out;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				long s = 0l;
				#pragma omp simd linear(x:step) reduction(+:s)
				for (j = 0; j < *n; j += step) {
					s += x[j];
				}
				sum[i] = s;
			}
		}
		break;
	}
}

PYMIC_KERNEL
void argmax(const int64_t* dtype, const void* a, const int64_t* niter, const int64_t* inciter, const int64_t* n, const int64_t* inca, int64_t* out) {
	int64_t i;
	switch (*dtype) {
		case DTYPE_FLOAT32:
		{
			float* x = (float*) a;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = cblas_isamax(*n, x, *inca);
			}
		}
		break;
		case DTYPE_FLOAT64:
		{
			double* x = (double*) a;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = cblas_idamax(*n, x, *inca);
			}
		}
		break;
	}
}

PYMIC_KERNEL
void argmin(const int64_t* dtype, const int64_t* niter, const int64_t* n, const void* a, const int64_t* inciter, const int64_t* inca, int64_t* out) {
	int64_t i;
	switch (*dtype) {
		case DTYPE_FLOAT32:
		{
			float* x = (float*) a;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = cblas_isamin(*n, x, *inca);
			}
		}
		break;
		case DTYPE_FLOAT64:
		{
			double* x = (double*) a;
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = cblas_idamin(*n, x, *inca);
			}
		}
		break;
	}
}
