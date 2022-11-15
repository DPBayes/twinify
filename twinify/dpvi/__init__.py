from typing import Callable
ModelFunction = Callable
GuideFunction = Callable

from collections import namedtuple
PrivacyLevel = namedtuple("PrivacyLevel", ["epsilon", "delta", "dp_noise"])


from twinify.dpvi.dpvi_model import DPVIModel, InferenceException
from twinify.dpvi.dpvi_result import DPVIResult

import twinify.dpvi.modelling
