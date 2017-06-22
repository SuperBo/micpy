#include "randomkit.h"
#include "distributions.h"
#include <mkl_vsl.h>

int rk_fill_bytes(rk_state *state, int device, long size, void *data)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) map(to: stream, size, data) \
                                      map(from: ret)
    {
        unsigned char *buffer = (unsigned char *) data;
        long n = size / 4;
        ret = viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD,
                            stream, n, (unsigned int *) data);
        if (ret != VSL_STATUS_OK) {
            ret = -1;
        }
        /* handle remaining part */
        int r = size % 4;
        if (r > 0) {
            int i;
            buffer += size - r;
            unsigned int buf;
            viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD,
                            stream, 1, &buf);
            for (i = 0; i < r; ++i) {
                buffer[i] = (unsigned char) (buf >> (8 * i));
            }
        }
    }

    return ret;
}

/*************************************************************************
 *                              DOUBLE FILL                              *
 *************************************************************************/

int rk_dfill_normal(rk_state *state, int device, long length,
                        void *data, double mean, double std_dev)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, mean, std_dev) map(from: ret)
    ret = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                        stream, length, (double *) data, mean, std_dev);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_standard_exponential(rk_state *state, int device, long length,
                        void *data)
{
    return rk_dfill_exponential(state, device, length, data, 1.0);
}

int rk_dfill_exponential(rk_state *state, int device, long length,
                        void* data, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, scale) map(from: ret)
    ret = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF,
                        stream, length, (double *) data, 0.0, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_uniform(rk_state *state, int device, long length,
                        void *data, double low, double high)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, low, high) map(from: ret)
    ret = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                        stream, length, (double *) data, low, high);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_standard_gamma(rk_state *state, int device, long length,
                        void *data, double shape)
{
    return rk_dfill_gamma(state, device, length, data, shape, 1.0);
}

int rk_dfill_gamma(rk_state *state, int device, long length,
                        void *data, double shape, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data,shape, scale) map(from: ret)
    ret = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM,
                        stream, length, (double *) data, shape, 0.0, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_beta(rk_state *state, int device, long length,
                        void *data, double a, double b)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, a, b) map(from: ret)
    ret = vdRngBeta(VSL_RNG_METHOD_BETA_CJA,
                        stream, length, (double *) data, a, b, 0.0, 1.0);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_laplace(rk_state *state, int device, long length,
                        void *data, double mean, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, mean, scale) map(from: ret)
    ret = vdRngLaplace(VSL_RNG_METHOD_LAPLACE_ICDF,
                        stream, length, (double*) data, mean, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_cauchy(rk_state *state, int device, long length,
                        void *data, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, scale) map(from: ret)
    ret = vdRngCauchy(VSL_RNG_METHOD_CAUCHY_ICDF,
                        stream, length, (double*) data, 0.0, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_weibull(rk_state *state, int device, long length,
                        void *data, double shape, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, shape, scale) map(from: ret)
    ret = vdRngWeibull(VSL_RNG_METHOD_WEIBULL_ICDF,
                        stream, length, (double*) data, shape, 0.0, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_gumbel(rk_state *state, int device, long length,
                        void *data, double loc, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, loc, scale) map(from: ret)
    ret = vdRngGumbel(VSL_RNG_METHOD_GUMBEL_ICDF,
                        stream, length, (double*) data, loc, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_lognormal(rk_state *state, int device, long length,
                        void *data, double mean, double sigma)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, mean, sigma) map(from: ret)
    ret = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2,
                        stream, length, (double*) data, mean, sigma, 0.0, 1.0);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_dfill_rayleigh(rk_state *state, int device, long length,
                        void *data, double scale)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, scale) map(from: ret)
    vdRngRayleigh(VSL_RNG_METHOD_RAYLEIGH_ICDF,
                        stream, length, (double*) data, 0.0, scale);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

/*************************************************************************
 *                              INTEGER FILL                             *
 *************************************************************************/

int rk_ifill_uniform(rk_state *state, int device, long length,
                        void *data, int low, int high)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, low, high) map(from: ret)
    ret = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                        stream, length, (int *) data, low, high);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}


int rk_ifill_binomial(rk_state *state, int device, long length,
                        void *data, int n, double p)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, n, p) map(from: ret)
    ret = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE,
                        stream, length, (int *) data, n, p);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_ifill_negative_binomial(rk_state *state, int device, long length,
                        void *data, double n, double p)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, n, p) map(from: ret)
    ret = viRngNegbinomial(VSL_RNG_METHOD_NEGBINOMIAL_NBAR,
                        stream, length, (int*) data, n, p);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_ifill_poisson(rk_state *state, int device, long length,
                        void *data, double lambda)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, lambda) map(from: ret)
    ret = viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM,
                        stream, length, (int*) data, lambda);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_ifill_geometric(rk_state *state, int device, long length,
                        void *data, double p)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, p) map(from: ret)
    ret = viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF,
                        stream, length, (int*) data, p);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_ifill_hypergeometric(rk_state *state, int device, long length,
                        void *data, int ngood, int nbad, int nsample)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, ngood, nbad, nsample) map(from: ret)
    ret = viRngHypergeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF,
                        stream, length, (int*) data, ngood+nbad, nsample , ngood);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}

int rk_ifill_bernoulli(rk_state *state, int device, long length,
                        void *data, double p)
{
    int ret;
    VSLStreamStatePtr stream = state->rng_streams[device];

    #pragma omp target device(device) \
            map(to: stream, length, data, p) map(from: ret)
    ret = viRngBernoulli(VSL_RNG_METHOD_GEOMETRIC_ICDF,
                        stream, length, (int*) data, p);

    return (ret == VSL_STATUS_OK) ? 0 : -1;
}