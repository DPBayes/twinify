import jax.numpy as np
import jax

import numpyro.distributions as dist
from numpyro.primitives import sample
from numpyro.infer import SVI, ELBO
from numpyro.contrib.autoguide import AutoDiagonalNormal, AutoContinuousELBO
from numpyro.optim import SGD

import numpy as onp

data = .3 * onp.random.randn(100) + 1.

def model(x):
    mu = sample('mus', dist.Normal(0., 1.))
    sig = sample('sig', dist.LogNormal(0., 1.))
    sample('x', dist.Normal(mu, sig), obs=x)

guide = AutoDiagonalNormal(model)

optim = SGD(1e-4)
svi = SVI(model, guide, optim, ELBO())

rng_key = jax.random.PRNGKey(0) # works with 1, does not work with 0
svi_state = svi.init(rng_key, data)
init_svi_state = svi_state

for _ in range(10):
    svi_state, loss = jax.lax.fori_loop(0, 100, lambda i, args: svi.update(args[0], data), (svi_state, 0.))
    # svi_state, loss = svi.update(svi_state, data)
    print(loss)
