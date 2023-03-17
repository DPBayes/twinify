# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2023- twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Iterable, Optional, Type
from numpyro.infer.autoguide import AutoGuide
from numpyro.handlers import condition, trace, seed, block
import jax.random

from twinify.dpvi import ModelFunction

__all__ = ['LoadableAutoGuide']

class ModelGuideMismatchException(Exception):

    def __init__(self, model_sites: Iterable[str], expected_sites: Iterable[str]):
        super().__init__(
            "The sample sites obtained from the model did not contain the expected observation sites,"
            " indicating that model and guide do not match!\n"
            f"Expected: {list(expected_sites)}\n"
            f"Model sites: {list(model_sites)}"
        )

class LoadableAutoGuide(AutoGuide):
    """
    Wraps any given AutoGuide class and allows it to be used in sampling without
    first running inference, given only a list of known observation sites in the model
    and previously learned parameters.

    The main intent is to facilitate sampling from the guide in a separate
    script than the inference took place, i.e., when the particular guide instance
    was never used for inference, which is not possible with regular AutoGuide instances.
    """
    # TODO: consider moving initialization into construction, in a RAII manner.
    #   Currently construction results in a half-initialized object, which invites highly error prone user code.

    def __init__(self,
            model: ModelFunction,
            observation_sites: Optional[Iterable[str]],
            base_guide_class: Type[AutoGuide],
            *base_guide_args: Any,
            **base_guide_kwargs: Any) -> None:
        """
        Creates a LoadableAutoGuide for the given `base_guide_class`. If `observation_sites`
        are not `None` and cover all sampling sites which `model` is conditioned on - i.e., those
        which should not be affected by `base_guide_class` - the created LoadableAutoGuide can be used
        for sampling without being used in inference first.

        If `observation_sites` are `None`, using the created LoadableAutoGuide in inference
        will behave exactly as using an instance of `base_guide_class` but will additionally
        extract the the sampling sites which `model` is conditioned on during inference and
        make them avaiable via the `observation_sites` property. These can then be used
        to initialise LoadableAutoGuide instances for sampling without inference later on.

        Args:
            model (ModelFunction): The model function.
            observation_sites (Iterable[str]): Optional collection of parameter site names that are observations in the model
                and should not be sampled from the guide.
            base_guide_class (Type[AutoGuide]): The AutoGuide subclass to wrap around (NOT a class instance!).
            base_guide_args: Positional arguments to pass into `base_guide_class`'s initializer.
            base_guide_kwargs: Keyword arguments to pass into `base_guide_class`'s initializer.
        """
        if base_guide_class == LoadableAutoGuide:
            raise ValueError("LoadableAutoGuide cannot wrap itself.")

        self._model = model
        self._base_guide_factory = lambda model: base_guide_class(model, *base_guide_args, **base_guide_kwargs)
        self._obs = frozenset(observation_sites) if observation_sites is not None else None
        self._guide = None

    @staticmethod
    def wrap_for_inference(base_guide_class: Type[AutoGuide]) -> Callable[[ModelFunction, Any], "LoadableAutoGuide"]:
        """
        Returns a callable accepting a model and arguments for the base guide class and returns a LoadableAutoGuide
        instance for `base_guide_class` (an AutoGuide subtype) set up for inference (but not yet initialized).

        Args:
            base_guide_class (Type[AutoGuide]): The AutoGuide subclass to wrap around (NOT a class instance!).

        Returns:
            Callable[[ModelFunction, Any...], LoadableAutoGuide]
        """
        def wrapped_for_inference(model: ModelFunction, *args, **kwargs):
            __doc__ = base_guide_class.__doc__
            return LoadableAutoGuide(model, None, base_guide_class, *args, **kwargs)
        
        wrapped_for_inference.__doc__ = base_guide_class.__doc__
        return wrapped_for_inference

    @staticmethod
    def wrap_for_sampling(base_guide_class: Type[AutoGuide], observation_sites: Iterable[str]) -> Callable[[ModelFunction, Any], "LoadableAutoGuide"]:
        """
        Returns a callable accepting a model and arguments for the base guide class and returns a LoadableAutoGuide
        instance for `base_guide_class` (an AutoGuide subtype) set up for sampling (but not yet initialized).

        Args:
            base_guide_class (Type[AutoGuide]): The AutoGuide subclass to wrap around (NOT a class instance!).
            observation_sites (Iterable[str]): Collection of parameter site names that are observations in the model
                and should not be sampled from the guide.

        Returns:
            Callable[[ModelFunction, Any...], LoadableAutoGuide]
        """
        def wrapped_for_sampling(model: Callable, *args, **kwargs):
            return LoadableAutoGuide(model, observation_sites, base_guide_class, *args, **kwargs)
        
        wrapped_for_sampling.__doc__ = base_guide_class.__doc__
        return wrapped_for_sampling

    @staticmethod
    def wrap_for_sampling_and_initialize(
            base_guide_class: Type[AutoGuide],
            observation_sites: Iterable[str],
            *model_args: Any, **model_kwargs) -> Callable[[ModelFunction, Any], "LoadableAutoGuide"]:
        """
        Returns a callable accepting a model and base guide arguments and returns a LoadableAutoGuide
        instance for `base_guide_class` (an AutoGuide subtype) ready for use for sampling without running inference.

        Equivalent to running:
        LoadableAutoGuide.wrap_for_sampling(base_guide_class, observation_sites)(model).initialize(*model_args, **model_kwargs).

        Args:
            base_guide_class (Type[AutoGuide]): The AutoGuide subclass to wrap around (NOT a class instance!).
            observation_sites (Iterable[str]): Collection of parameter site names that are observations in the model
                and should not be sampled from the guide.
            model_args (Any): Optional positional arguments to be passed through to the model.
            model_kwargs (Any): Optional keyword arguments to be passed through to the model.

        Returns:
            Callable[[ModelFunction, Any...], LoadableAutoGuide]
        """
        def wrapped_for_sampling_with_init(model: ModelFunction, *base_guide_args, **base_guide_kwargs):
            guide = LoadableAutoGuide(
                model, observation_sites, base_guide_class, *base_guide_args, **base_guide_kwargs
            )
            guide.initialize(*model_args, **model_kwargs)
            return guide
        
        wrapped_for_sampling_with_init.__doc__ = base_guide_class.__doc__
        return wrapped_for_sampling_with_init

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """
        Traces through the model function provided to this LoadableAutoGuide instance during construction
        and prepares this instance for usage.

        If observation sites were provided, traces through the model to determine the parameter space
        of the guide, resulting in an instance that can be used for sampling without being used in inference first.
        """
        if self._guide is not None: return

        if self._obs is not None:
            # We are given a set of observation sites that should be ignored by the guide
            # Here's the general strategy: NumPyro AutoGuide requires that all
            # parameters sites corresponding to observations need to be conditioned
            # on, but this is not case yet. To do so, we sample some data from the
            # prior predictive distribution (using the model function) and condition
            # the model on those for the guide initialisation.

            # We will probably be in the middle of a sampling stack so we block
            # all handlers sitting above the guide while we initialize it
            with block():

                # get plausible fake values for observed sites from prior
                with trace() as tr:
                    seed(self._model, jax.random.PRNGKey(0))(*args, **kwargs)
                    fake_obs = {site: val['value'] for site, val in tr.items() if site in self._obs}

                    if len(fake_obs) < len(self._obs):
                        model_sites = [site for site, val in tr.items() if val["type"] == "sample"]
                        raise ModelGuideMismatchException(model_sites, self._obs)

                # feed model conditioned on fake observations to guide
                guide = self._base_guide_factory(condition(self._model, fake_obs))
                # trigger guide initialisation with fake observatons
                seed(guide, jax.random.PRNGKey(0))(*args, **kwargs)

            self._guide = guide
        else:
            # We are not given a set of observation sites, so we assume this call is
            # part of inference. We don't need to do anything to the guide.
            # However, we trace through the model to collect all observation sites
            # that it is conditioned on.

            with block():
                with trace() as tr:
                    seed(self._model, jax.random.PRNGKey(0))(*args, **kwargs)
                self._obs = [name for name, site in tr.items() if site.get('is_observed', False)]

            guide = self._base_guide_factory(self._model) # initialise guide with model as normal
            self._guide = guide

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._guide is None:
            self.initialize(*args, **kwargs)

        return self._guide.__call__(*args, **kwargs)

    @property
    def base_guide(self) -> AutoGuide:
        """
        The instance of type base_guide_class this LoadableAutoGuide instance wraps around.

        Note: Requires initialization.
        """
        if not self.is_initialized:
            raise RuntimeError("The guide must be initialized from the model first! "\
                "You can call initialize(*model_args, **model_kwargs) to do so.")
        return self._guide

    def sample_posterior(self, rng_key, params, *args, **kwargs):
        return self.base_guide.sample_posterior(rng_key, params, *args, **kwargs)

    @property
    def observation_sites(self) -> Iterable[str]:
        """
        The observation sites in the model, i.e., sample sites that are not
        sampled from the guide.

        If observation sites were provided during construction, the collection returned by this
        function will be identical. If not, this function returns the observation sites
        identified from the model during initialization of the guide.
        """
        return self._obs

    def median(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the posterior median value of each latent variable.

        Note: Requires initialization.

        Args:
            params (Dict[str, Any]): A dict containing parameter values.
                The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
                method from :class:`~numpyro.infer.svi.SVI`.
        Returns: A dict mapping sample site name to median value.
        """

        return self.base_guide.median(params)

    def quantiles(self, params: Dict[str, Any], quantiles: Iterable[float]) -> Dict[str, Any]:
        """
        Returns posterior quantiles each latent variable. Example::

            print(guide.quantiles(params, [0.05, 0.5, 0.95]))

        Note: Requires initialization.

        Args:
            params (Dict[str, Any]): A dict containing parameter values.
                The parameters can be obtained using :meth:`~numpyro.infer.svi.SVI.get_params`
                method from :class:`~numpyro.infer.svi.SVI`.
            quantiles (Iterable[float]): A list of requested quantiles between 0 and 1.

        Returns:A dict mapping sample site name to an array of quantile values.
        """
        return self.base_guide.quantiles(params, quantiles)

    @property
    def is_initialized(self) -> bool:
        return self._guide is not None
