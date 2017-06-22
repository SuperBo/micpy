#ifndef _MPY_DISTR_
#define _MPY_DISTR_

#ifdef __cplusplus
extern "C" {
#endif

/* Random bytes */
int rk_fill_bytes(rk_state *state, int device, long size, void *data);

/* Normal distribution with mean=loc and standard deviation=scale. */
int rk_dfill_normal(rk_state *state, int device, long length,
                        void *data, double mean, double std_dev);

/* Standard exponential distribution (mean=1) computed by inversion of the
 * CDF. */
int rk_standard_exponential(rk_state *state, int device, long length,
                        void *data);

/* Exponential distribution. */
int rk_dfill_exponential(rk_state *state, int device, long length,
                        void* data, double scale);

/* Uniform distribution on interval [low, high). */
int rk_dfill_uniform(rk_state *state, int device, long length,
                        void *data, double low, double high);

/* Gamma distribution with shape and scale. */
int rk_dfill_gamma(rk_state *state, int device, long length,
                        void *data, double shape, double scale);

/* Beta distribution computed by combining two gamma variates (Devroye p. 432).
 */
int rk_dfill_beta(rk_state *state, int device, long length,
                        void *data, double a, double b);

/* Binomial distribution with n Bernoulli trials with success probability p.
 * When n*p <= 30, the "Second waiting time method" given by (Devroye p. 525) is
 * used. Otherwise, the BTPE algorithm of (Kachitvichyanukul and Schmeiser 1988)
 * is used. */
int rk_ifill_binomial(rk_state *state, int device, long length,
                        void *data, int n, double p);

/* Negative binomial distribution computed by generating a Gamma(n, (1-p)/p)
 * variate Y and returning a Poisson(Y) variate (Devroye p. 543). */
int rk_ifill_negative_binomial(rk_state *state, int device, long length,
                        void *data, double n, double p);

/* Poisson distribution with mean=lam.
 * When lam < 10, a basic algorithm using repeated multiplications of uniform
 * variates is used (Devroye p. 504).
 * When lam >= 10, algorithm PTRS from (Hoermann 1992) is used.
 */
int rk_ifill_poisson(rk_state *state, int device, long length,
                        void *data, double lambda);

/* Cauchy distribution
 * (Devroye p. 451). */
int rk_dfill_cauchy(rk_state *state, int device, long length,
                        void *data, double scale);

/* Weibull distribution via inversion (Devroye p. 262) */
int rk_dfill_weibull(rk_state *state, int device, long length,
                        void *data, double shape, double scale);

/* Laplace distribution */
int rk_dfill_laplace(rk_state *state, int device, long length,
                        void *data, double mean, double scale);

/* Gumbel distribution */
int rk_dfill_gumbel(rk_state *state, int device, long length,
                        void *data, double loc, double scale);

/* Log-normal distribution */
int rk_dfill_lognormal(rk_state *state, int device, long length,
                        void *data, double mean, double sigma);

/* Rayleigh distribution */
int rk_dfill_rayleigh(rk_state *state, int device, long length,
                        void *data, double scale);

/* Geometric distribution */
int rk_ifill_geometric(rk_state *state, int device, long length,
                        void *data, double p);

/* Hypergeometric distribution */
int rk_ifill_hypergeometric(rk_state *state, int device, long length,
                        void *data, int ngood, int nbad, int nsample);

int rk_ifill_bernoulli(rk_state *state, int device, long length,
                        void *data, double p);

int rk_ifill_uniform(rk_state *state, int device, long length,
                        void *data, int low, int high);

#ifdef __cplusplus
}
#endif

#endif