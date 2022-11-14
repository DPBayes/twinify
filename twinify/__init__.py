import warnings as _warnings
from jax.config import config as _config
_config.update("jax_enable_x64", True)
import jax.numpy as _jnp
if _jnp.array([1.]).dtype != _jnp.float64:
    _warnings.warn(
        "Failed to set floating point precision to 64 bits. You may experience instabilities "
        "with twinify as a result.\n"
        "This can happen when the jax library is already initialized in your scripts before twinify is loaded. "
        "In that case, please enable 64 bit precision by including the following before the first usage of jax:\n"
        "   from jax.config import config\n"
        "   config.update('jax_enable_x64', True).",
        stacklevel=2
    )

import twinify.dpvi
import twinify.napsu_mq
from twinify.dataframe_data import DataDescription
from twinify.version import VERSION
__version__ = VERSION