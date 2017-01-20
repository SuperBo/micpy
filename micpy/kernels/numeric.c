#include "core.h"

float smax(const int64_t n, float* x, const int64_t incx) {
	float ma = x[0];

	#pragma omp parallel for simd reduction(max:ma)
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] > ma)
			ma = x[i];
	}

	return ma;
}

double dmax(const int64_t n, double* x, const int64_t incx) {
	double ma = x[0];

	#pragma omp parallel for simd reduction(max:ma)
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] > ma)
			ma = x[i];
	}

	return ma;
}

float smin(const int64_t n, float* x, const int64_t incx) {
	float ma = x[0];

	#pragma omp parallel for simd reduction(min:ma)
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] < ma)
			ma = x[i];
	}

	return ma;
}

double dmin(const int64_t n, double* x, const int64_t incx) {
	double ma = x[0];

	#pragma omp parallel for simd reduction(min:ma)
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] < ma)
			ma = x[i];
	}

	return ma;
}

int64_t ismax(const int64_t n, float* x, const int64_t incx) {
	int64_t idx = 0;
	float mx_val = x[0];

	#pragma omp parallel for
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] > mx_val) {
			mx_val = x[i];
			idx = i;
		}
	}

	return idx;
}

int64_t idmax(const int64_t n, double* x, const int64_t incx) {
	int64_t idx = 0;
	double mx_val = x[0];

	#pragma omp parallel for
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] > mx_val) {
			mx_val = x[i];
			idx = i;
		}
	}

	return idx;
}

int64_t ismin(const int64_t n, float* x, const int64_t incx) {
	int64_t idx = 0;
	float mi_val = x[0];

	#pragma omp parallel for
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] < mi_val) {
			mi_val = x[i];
			idx = i;
		}
	}

	return idx;
}

int64_t idmin(const int64_t n, double* x, const int64_t incx) {
	int64_t idx = 0;
	double mi_val = x[0];

	#pragma omp parallel for
	for (int64_t i = 0; i < n; i += incx) {
		if (x[i] < mi_val) {
			mi_val = x[i];
			idx = i;
		}
	}

	return idx;
}

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

			#pragma omp parallel for
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = ismax(*n, x, *inca);
			}
		}
		break;
		case DTYPE_FLOAT64:
		{
			double* x = (double*) a;

			#pragma omp parallel for
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = idmax(*n, x, *inca);
			}
		}
		break;
	}
}

PYMIC_KERNEL
void argmin(const int64_t* dtype, const void* a, const int64_t* niter, const int64_t* n, const int64_t* inciter, const int64_t* inca, int64_t* out) {
	int64_t i;
	switch (*dtype) {
		case DTYPE_FLOAT32:
		{
			float* x = (float*) a;

			#pragma omp parallel for
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = ismin(*n, x, *inca);
			}
		}
		break;
		case DTYPE_FLOAT64:
		{
			double* x = (double*) a;

			#pragma omp parallel for
			for (i = 0; i < *niter; ++i, x += *inciter) {
				out[i] = idmin(*n, x, *inca);
			}
		}
		break;
	}
}
