from .mpyrandom import RandomState

_rand = RandomState()
seed = _rand.seed
bytes = _rand.bytes
random_sample = _rand.random_sample
rand = _rand.rand
randn = _rand.randn
randint = _rand.randint

# Continuous Distributions
uniform = _rand.uniform
standard_normal = _rand.standard_normal
normal = _rand.normal
beta = _rand.beta
exponential = _rand.exponential
standard_exponential = _rand.standard_exponential
standard_gamma = _rand.standard_gamma
gamma = _rand.gamma
standard_cauchy = _rand.standard_cauchy
cauchy = _rand.cauchy
weibull = _rand.weibull
laplace = _rand.laplace
gumbel = _rand.gumbel
lognormal = _rand.lognormal
rayleigh = _rand.rayleigh

# Discrete Distributions
binomial = _rand.binomial
negative_binomial = _rand.negative_binomial
poisson = _rand.poisson
geometric = _rand.geometric
hypergeometric = _rand.hypergeometric
bernoulli = _rand.bernoulli
