#include "core.h"


/*
 * Compute output of cross entropy layer follow a softmax layer
 */
void softmax_cross_entropy_forward_float32(const int64_t* m, const int64_t* n,
		const double* normalize_coeff, const int64_t* ignored_label,
		const float* x, const int32_t* t, float* y, float* loss)
{

	float cross = 0.0f;

	//Loop for each samples
	//#pragma omp parallel for reduction(+:cross)
	for (int64_t i = 0; i < *m; ++i) {
		if (t[i] < 0 || t[i] >= *n || t[i] == *ignored_label)
			continue;

		size_t offset = (*n) * i;
		float max, sum = 0.0f;

		//Calculate total sum of e^(x-m)
		if (y == NULL) {
			max = _max_fa_((x + offset), *n);

			#pragma omp simd aligned(y,x:64) reduction(+:sum)
			for (int64_t j = 0; j < *n; ++j) {
				//Calculate normalized e^(x-max)
				sum += expf(x[offset + j] - max);
			}
		}
		else {
			float* x_ = (float*)x + offset;
			flat_softmax_forward_float32(x_, &max, &sum, y+offset, n);
		}

		//Calculate output of cross entropy function
		cross += -x[offset + t[i]] + max + logf(sum);
	}

	//Normalize result
	float coeff;
	if (normalize_coeff)
		coeff = (float) *normalize_coeff;
	else
		coeff = 1.0f / (float)(*m);

	*loss = cross * coeff;
}

void softmax_cross_entropy_forward_float64(const int64_t* m, const int64_t* n,
		const double* normalize_coeff, const int64_t* ignored_label,
		const double* x, const int32_t* t, double* y, double* loss)
{
	double cross = 0.0;

	//#pragma omp parallel for reduction(+:cross)
	for (int64_t i = 0; i < *m; ++i) {
		if (t[i] < 0 || t[i] >= *n || t[i] == *ignored_label)
			continue;

		size_t offset = (*n) * i;
		double max, sum = 0.0;

		//Calculate total sum of e^(x-m)
		if (y == NULL) {
			max = _max_da_((x + offset), *n);
			#pragma omp simd aligned(y,x:64) reduction(+:sum)
			for (int64_t j = 0; j < *n; ++j) {
				//Calculate normalized e^(x-max)
				sum += exp(x[offset + j] - max);
			}
		}
		else {
			double* x_ = (double*)x + offset;
			flat_softmax_forward_float64(x_, &max, &sum, y+offset, n);
		}

		//Calculate output of cross entropy function
		cross += -x[offset + t[i]] + max + log(sum);
	}

	//Normalize result
	double coeff;
	if (normalize_coeff)
		coeff = *normalize_coeff;
	else
		coeff = 1.0 / (double)(*m);

	*loss = cross * coeff;
}

void softmax_cross_entropy_backward_float32(const int64_t* m, const int64_t* n,
		const double* normalize_coeff, const int64_t* ignored_label,
		const float* x, const int32_t* t, float* y, const float* gloss, float* gx)
{
	//Check if y is cached
	if (y) {
		cblas_scopy((*m)*(*n), y, 1, gx, 1);
	}
	else {
		softmax_forward_float32(x, gx, m, n);
	}

	int64_t iglabel = (ignored_label) ? *ignored_label : -1;

	for (int64_t i = 0; i < *m; ++i) {
		size_t offset = (*n) * i;

		// Set gx of ignored_label to zero
		if (t[i] < 0 || t[i] == iglabel) {
			#pragma omp simd
			for (int64_t j = 0; j < *n; ++j) {
				gx[offset + j] = 0.0f;
			}
			continue;
		}

		gx[offset + t[i]] -= 1.0f;
	}

	float coeff = (normalize_coeff) ? *normalize_coeff : 1.0f / (*m);

	#pragma omp simd aligned(gx:64)
	for (int64_t i = 0; i < (*m)*(*n); ++i) {
		gx[i] *= coeff * *(gloss);
	}
}

void softmax_cross_entropy_backward_float64(const int64_t* m, const int64_t* n,
		const double* normalize_coeff, const int64_t* ignored_label,
		const double* x, const int32_t* t, double* y, const double* gloss, double* gx)
{
	//Check if y is cached
	if (y) {
		cblas_dcopy((*m)*(*n), y, 1, gx, 1);
	}
	else {
		softmax_forward_float64(x, gx, m, n);
	}

	int64_t iglabel = (ignored_label) ? *ignored_label : -1;

	for (size_t i = 0; i < *m; ++i) {
		size_t offset = (*n) * i;

		// Set gx of ignored_label to zero
		if (t[i] < 0 || t[i] == *ignored_label) {
			#pragma omp simd aligned(gx:64)
			for (int64_t j = 0; j < *n; ++j) {
				gx[offset + j] = 0.0f;
			}
			continue;
		}

		gx[offset + t[i]] -= 1.0;
	}

	double coeff = (normalize_coeff) ? *normalize_coeff : 1.0 / (*m);

	#pragma omp simd aligned(gx:64)
	for (int64_t i = 0; i < (*m)*(*n); ++i) {
		gx[i] *= coeff * (*gloss);
	}
}
