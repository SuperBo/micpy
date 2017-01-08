#include "core.h"
#include <time.h>

#define PSEUDO_GEN VSL_BRNG_SFMT19937
#define QUASI_GEN VSL_BRNG_SOBOL


PYMIC_KERNEL
void gaussian_distribution(const int64_t* dtype, const double* mean, const double* standard_deviation, void* out, const int64_t* size) {
	int status;
	VSLStreamStatePtr streamn;
	status = vslNewStream(&streamn, PSEUDO_GEN, time(NULL));

	if (status != VSL_STATUS_OK) {
		vslDeleteStream(&streamn);
		return;
	}

	double a = (mean) ? *mean : 0.0;
	double sigma = (standard_deviation) ? *standard_deviation : 1.0;

	switch (*dtype) {
		case DTYPE_FLOAT32:
			{
			float* fout = (float*) out;
			status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, streamn,
						*size, fout, (float) a, (float) sigma);
			}
			break;
		case DTYPE_FLOAT64:
			status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, streamn,
						*size, out, a, sigma);
			break;
	}

	vslDeleteStream(&streamn);
}

PYMIC_KERNEL
void uniform_distribution(const int64_t* dtype, const double* left, const double* right, void* out, const int64_t* size) {
	int status;
	VSLStreamStatePtr streamu;
	status = vslNewStream(&streamu, PSEUDO_GEN, time(NULL));

	if (status != VSL_STATUS_OK) {
		vslDeleteStream(&streamu);
		return;
	}

	double a = (left) ? *left : 0.0;
	double b = (right) ? *right : 1.0;

	switch (*dtype) {
		case DTYPE_FLOAT32:
			{
			float* fout = (float*) out;
			status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streamu,
						*size, fout, (float) a, (float) b);
			}
			break;
		case DTYPE_FLOAT64:
			status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streamu,
						*size, out, a, b);
			break;
	}

	vslDeleteStream(&streamu);
}
