# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union, Iterable, BinaryIO, List, Dict, FrozenSet
import os
from dataclasses import dataclass, field
import pandas as pd

import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
import pickle

from twinify.mst import MST_selection
from twinify.base import InferenceModel, InferenceResult, InvalidFileFormatException
from twinify.napsu_mq.markov_network import MarkovNetwork
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.dataframe_data import DataFrameData, DataDescription, disallow_integers
import twinify.napsu_mq.privacy_accounting as privacy_accounting
import twinify.napsu_mq.maximum_entropy_inference as mei
from twinify.napsu_mq.private_pgm.domain import Domain
from twinify.napsu_mq.private_pgm.dataset import Dataset
from twinify.napsu_mq.utils import progressbar_choice
import d3p.random


@dataclass
class NapsuMQLaplaceApproximationConfig:
    """Configuration for the Laplace approximation.

    Args:
        max_iters (int): Maximum number of optimisation iterations. Default 100.
        num_samples (int): Number of posterior samples after inference. Default 10 000.
        tol (float): Optimiser error tolerance. Default 1e-2.
        max_retries (int): Maximum number of retries in case optimiser fails. Default 5.
    """
    max_iters: int = 100
    num_samples: int = 10000
    tol: float = 1e-2
    max_retries: int = 5


@dataclass 
class NapsuMQMCMCConfig:
    """Configuration for MCMC.

    Args:
        num_samples (int): Number of kept samples from MCMC. Default 2000.
        num_warmup (int): Number of warmup samples that are dropped. Default 800.
        num_chains (int): Number of MCMC chains to run. Default 4.
    """
    num_samples: int = 2000
    num_warmup: int = 800
    num_chains: int = 4


class NapsuMQInferenceConfig:
    """Configuration for NAPSU-MQ posterior inference.

    Args:
        method (str): Inference method, one of 'mcmc', 'laplace' 'laplace+mcmc'. Default 'mcmc'.
        laplace_approximation_config (NapsuMQLaplaceApproximationConfig): Configuration for Laplace approximation. Only used for methods 'laplace' and 'laplace+mcmc'.
        mcmc_config (NapsuMQMCMCConfig): Configuration for MCMC. Only used for methods 'mcmc' and 'laplace+mcmc'.
    """
    def __init__(
        self, method: str = "mcmc", 
        laplace_approximation_config: Optional[NapsuMQLaplaceApproximationConfig] = None, 
        mcmc_config: Optional[NapsuMQMCMCConfig] = None
    ) -> None:
        self._method = method
        self._check_method()

        if laplace_approximation_config is None:
            self._laplace_approximation_config = NapsuMQLaplaceApproximationConfig()
        else:
            self._laplace_approximation_config = laplace_approximation_config

        if mcmc_config is None:
            self._mcmc_config = NapsuMQMCMCConfig()
        else:
            self._mcmc_config = mcmc_config

    @property
    def method(self) -> str:
        return self._method
    
    @method.setter
    def method(self, new_method: str) -> None:
        self._method = new_method
        self._check_method()
        self._check_la_config()
        self._check_mcmc_config()

    @property 
    def laplace_approximation_config(self) -> NapsuMQLaplaceApproximationConfig:
        return self._laplace_approximation_config

    @laplace_approximation_config.setter 
    def laplace_approximation_config(self, new_config:Optional[NapsuMQLaplaceApproximationConfig]) -> None:
        self._laplace_approximation_config = new_config
        self._check_la_config()

    @property
    def mcmc_config(self) -> NapsuMQMCMCConfig:
        return self._mcmc_config

    @mcmc_config.setter
    def mcmc_config(self, new_config) -> None:
        self._mcmc_config = new_config
        self._check_mcmc_config()

    def _check_method(self) -> None:
        if self.method not in ["mcmc", "laplace", "laplace+mcmc"]:
            raise ValueError("method must be one of 'mcmc', 'laplace' or 'laplace+mcmc'. Received {}".format(self.method))

    def _check_la_config(self) -> None:
        if (self.method == "laplace" or self.method == "laplace+mcmc") and self.laplace_approximation_config is None:
            raise ValueError("laplace_approximation_config is required when method is '{}'".format(self.method))

    def _check_mcmc_config(self) -> None:
        if (self.method == "mcmc" or self.method == "laplace+mcmc") and self.mcmc_config is None:
            raise ValueError("mcmc_config is required when method is '{}'".format(self.method))


class NapsuMQModel(InferenceModel):
    """Implementation for NAPSU-MQ algorithm, differentially private synthetic data generation method for discrete sensitive data.

    reference: arXiv:2205.14485
    "Noise-Aware Statistical Inference with Differentially Private Synthetic Data", Ossi Räisä, Joonas Jälkö, Samuel Kaski & Antti Honkela
    """

    def __init__(
        self, queries: Optional[Iterable[FrozenSet[str]]] = None, 
        forced_queries_in_automatic_selection: Optional[Iterable[FrozenSet[str]]] = None,
        inference_config: NapsuMQInferenceConfig = NapsuMQInferenceConfig(),
    # required_marginals: Iterable[FrozenSet[str]] = tuple()
    ) -> None:
        """
        Args:
            queries (iterable of sets of str or None): Queries that NAPSU-MQ attempts to preserve. None selects automatically. Default None.
            forced_queries_in_automatic_selection (iterable of sets of str): Force queries to be included with automatic selection.
            inference_config (NapsuMQInferenceConfig): Configuration for inference
        """

        super().__init__()
        if forced_queries_in_automatic_selection is None:
            forced_queries_in_automatic_selection = tuple()
        self._forced_queries_in_automatic_selection = forced_queries_in_automatic_selection
        self._queries = queries
        self._inference_config = inference_config

    def fit(self, data: pd.DataFrame, rng: d3p.random.PRNGState, epsilon: float, delta: float,
            show_progress: bool = True,
            return_diagnostics: bool = False) -> 'NapsuMQResult':
        """Fit differentially private NAPSU-MQ model from data.

        Args:
            data (pd.DataFrame): Pandas Dataframe containing discrete categorical data
            rng (d3p.random.PRNGState): d3p PRNG key
            epsilon (float): Epsilon for differential privacy mechanism
            delta (float): Delta for differential privacy mechanism
            show_progress (bool): Show progressbar for MCMC
            return_diagnostics (bool): Return diagnostics from inference

        Returns:
            NapsuMQResult: Class containing learned probabilistic model with posterior values
        """
        inference_config = self._inference_config

        dataframe = DataFrameData(data, integers_handler=disallow_integers)
        n, d = dataframe.int_df.shape

        if self._queries is None:
            domain_key_list = list(dataframe.columns)
            domain_value_count_list = [len(dataframe.values_by_col[key]) for key in domain_key_list]
            domain = Domain(domain_key_list, domain_value_count_list)
            self._queries = MST_selection(
                Dataset(dataframe.int_df, domain), epsilon, delta,
                cliques_to_include=self._forced_queries_in_automatic_selection
            )

        queries = FullMarginalQuerySet(self._queries, dataframe.column_domain_sizes)
        queries = queries.get_canonical_queries()
        mnjax = MarkovNetwork(dataframe.values_by_col, queries)
        suff_stat = np.sum(queries.flatten()(dataframe.int_df.to_numpy()), axis=0)

        # determine Gaussian mech DP noise level for given privacy level
        sensitivity = np.sqrt(2 * len(self._queries))
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)

        # add DP noise (according to Gaussian mechanism)
        inference_rng, dp_rng = d3p.random.split(rng, 2)
        dp_noise = d3p.random.normal(dp_rng, suff_stat.shape) * sigma_DP
        dp_suff_stat = suff_stat + dp_noise

        inference_rng = d3p.random.convert_to_jax_rng_key(inference_rng)

        if inference_config.method == "mcmc":
            mcmc_config = inference_config.mcmc_config
            mcmc = mei.run_numpyro_mcmc(
                inference_rng, dp_suff_stat, n, sigma_DP, mnjax, 
                num_samples=mcmc_config.num_samples, 
                num_warmup=mcmc_config.num_warmup, 
                num_chains=mcmc_config.num_chains,
                show_progressbar=show_progress,
            )
            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = posterior_values.lambdas.values.transpose()
            diagnostics = inf_data
        elif inference_config.method in ["laplace", "laplace+mcmc"]:
            # Do Laplace approximation
            approx_rng, approx_sample_rng, mcmc_rng = jax.random.split(inference_rng, 3)
            laplace_approx_config = inference_config.laplace_approximation_config
            laplace_approx, laplace_success = mei.run_numpyro_laplace_approximation(
                approx_rng, dp_suff_stat, n, sigma_DP, mnjax, 
                max_retries=laplace_approx_config.max_retries,
                tol=laplace_approx_config.tol,
                max_iters=laplace_approx_config.max_iters
            )
            if inference_config.method == "laplace+mcmc":
                mcmc_config = inference_config.mcmc_config
                mcmc, backtransform = mei.run_numpyro_mcmc_normalised(
                    mcmc_rng, dp_suff_stat, n, sigma_DP, mnjax, laplace_approx, 
                    num_samples=mcmc_config.num_samples, 
                    num_warmup=mcmc_config.num_warmup,
                    num_chains=mcmc_config.num_chains,
                    show_progressbar=show_progress,
                )
                inf_data = az.from_numpyro(mcmc, log_likelihood=False)
                posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
                posterior_values = backtransform(posterior_values.norm_lambdas.values.transpose())
                diagnostics = (laplace_success, inf_data)
            else:
                posterior_values = laplace_approx.sample(approx_sample_rng, (laplace_approx_config.num_samples,))
                diagnostics = laplace_success
        else:
            raise ValueError("inference_config.method must be one of 'mcmc', 'laplace' or 'laplace+mcmc'")
                
        result = NapsuMQResult(dataframe.values_by_col, queries, posterior_values, dataframe.data_description)
        if return_diagnostics:
            return result, diagnostics
        else:
            return result


class NapsuMQResult(InferenceResult):
    """
    NAPSU-MQ result class containing learned differentially private probabilistic model from data.
    Contains functions to generate differentially private synthetic datasets from the original dataset.
    """

    def __init__(self, dataframe_domain: Dict[str, List[int]], queries: 'FullMarginalQuerySet', posterior_values: jnp.ndarray,
                 data_description: DataDescription) -> None:
        super().__init__()
        self._dataframe_domain = dataframe_domain
        self._queries = queries
        self._posterior_values = np.asarray(posterior_values)
        self._data_description = data_description

    @property
    def dataframe_domain(self) -> Dict[str, List[int]]:
        return self._dataframe_domain

    @property
    def queries(self) -> 'FullMarginalQuerySet':
        return self._queries

    @property
    def posterior_values(self) -> np.ndarray:
        return self._posterior_values

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    def _store_to_io(self, write_io: BinaryIO) -> None:
        assert write_io.writable()
        result = pickle.dumps(self)
        return NapsuMQResultIO.store_to_io(write_io, result)

    @classmethod
    def _load_from_io(cls, read_io: BinaryIO, **kwargs) -> 'NapsuMQResult':
        return NapsuMQResultIO.load_from_io(read_io)

    def generate(
            self,
            rng: d3p.random.PRNGState,
            num_parameter_samples: int,
            num_data_per_parameter_sample: int = 1,
            single_dataframe: bool = True,
            show_progress: bool = False) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
        jax_rng = d3p.random.convert_to_jax_rng_key(rng)
        mnjax = MarkovNetwork(self._dataframe_domain, self._queries)
        posterior_values = jnp.array(self.posterior_values)
        jax_rng, ind_rng = jax.random.split(jax_rng)
        inds = jax.random.choice(key=ind_rng, a=posterior_values.shape[0], shape=[num_parameter_samples])
        posterior_sample = posterior_values[inds, :]
        rng, *data_keys = jax.random.split(jax_rng, num_parameter_samples + 1)
        syn_datasets = [mnjax.sample(syn_data_key, jnp.array(posterior_value), num_data_per_parameter_sample) for
                        syn_data_key, posterior_value
                        in progressbar_choice(list(zip(data_keys, posterior_sample)), show_progress)]

        dataframes = [self.data_description.map_to_categorical(syn_data) for syn_data in syn_datasets]

        if single_dataframe is True:
            combined_dataframe = pd.concat(dataframes, ignore_index=True)
            return combined_dataframe
        else:
            return dataframes


class NapsuMQResultIO:
    IDENTIFIER = "NapsuMQ".encode("utf8")
    CURRENT_IO_VERSION = 1
    # Replace with twinify.serialization.ENDIANESS with merge
    CURRENT_IO_VERSION_BYTES = CURRENT_IO_VERSION.to_bytes(1, 'big')

    @staticmethod
    def load_from_io(read_io: BinaryIO) -> NapsuMQResult:
        assert read_io.readable()

        if not NapsuMQResultIO.is_file_stored_result_from_io(read_io, reset_cursor=False):
            raise InvalidFileFormatException(NapsuMQResult, "Stored data does not have correct type identifier.")

        current_version = int.from_bytes(read_io.read(1), 'big')
        if current_version != NapsuMQResultIO.CURRENT_IO_VERSION:
            raise InvalidFileFormatException(NapsuMQResult, "Stored data uses an unknown storage format version.")

        result_binary = read_io.read()
        result = pickle.loads(result_binary)

        return result

    @staticmethod
    def is_file_stored_result_from_io(read_io: BinaryIO, reset_cursor: bool) -> bool:
        assert read_io.readable()
        assert read_io.seekable()

        identifier = read_io.read(len(NapsuMQResultIO.IDENTIFIER))
        if reset_cursor:
            read_io.seek(-len(identifier), os.SEEK_CUR)

        if identifier == NapsuMQResultIO.IDENTIFIER:
            return True

        return False

    @staticmethod
    def store_to_io(write_io: BinaryIO, result) -> None:
        assert write_io.writable()

        write_io.write(NapsuMQResultIO.IDENTIFIER)
        write_io.write(NapsuMQResultIO.CURRENT_IO_VERSION_BYTES)
        write_io.write(result)
