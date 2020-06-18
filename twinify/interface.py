import jax.numpy as np
from numpyro.primitives import deterministic

__all__ = ['get_feature', 'sample_combined']

def get_feature(x, i):
    """
    Wraps access to a feature value of an observation and deals gracefully
    if no observation is given.

    Args:
        x (jax.numpy.ndarray): Array holding all features of a single data instance or None.
        i (int): Index of feature value to return.
    Returns:
        x[i] if x is not None; otherwise None
    """
    return None if x is None else x[i]

def sample_combined(*feature_samples):
    """
    Combines individual feature samples into a single sample of the full
    observation vector expected by twinify.

    Args:
        *feature_samples: Iterable of individual feature values.
    Samples:
        site `x` holding the combined feature vector
    Returns:
        jax.numpy.ndarray: sampled values
    """
    return deterministic('x', np.array(feature_samples))
