# Copyright 2022 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union, Iterable, BinaryIO, Callable, Mapping, List
import os
import pandas as pd

import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
from mbi import Domain, Dataset
import dill

from twinify.mst import MST_selection
from twinify.base import InferenceModel, InferenceResult, InvalidFileFormatException
from twinify.napsu_mq.markov_network import MarkovNetwork
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.dataframe_data import DataFrameData, DataDescription
import twinify.napsu_mq.privacy_accounting as privacy_accounting
import twinify.napsu_mq.maximum_entropy_inference as mei
import d3p.random


class NapsuMQModel(InferenceModel):
    """Implementation for NAPSU-MQ algorithm, differentially private synthetic data generation method for discrete sensitive data.

    reference: arXiv:2205.14485
    "Noise-Aware Statistical Inference with Differentially Private Synthetic Data", Ossi Räisä, Joonas Jälkö, Samuel Kaski & Antti Honkela
    """

    def __init__(self, column_feature_set, use_laplace_approximation=True) -> None:
        super().__init__()
        if column_feature_set is None:
            column_feature_set = []
        self.column_feature_set = column_feature_set
        self.use_laplace_approximation = use_laplace_approximation

    def fit(self, data: pd.DataFrame, rng: d3p.random.PRNGState, epsilon: float, delta: float, query_sets: List = None,
            **kwargs) -> 'NapsuMQResult':
        """Fit differentially private NAPSU-MQ model from data.

        Args:
            data (pd.DataFrame): Pandas Dataframe containing discrete categorical data
            rng (d3p.random.PRNGState): d3p PRNG key
            epsilon (float): Epsilon for differential privacy mechanism
            delta (float): Delta for differential privacy mechanism
            query_sets (List): Marginal queries

        Returns:
            NapsuMQResult: Class containing learned probabilistic model with posterior values
        """
        column_feature_set = self.column_feature_set
        use_laplace_approximation = self.use_laplace_approximation

        dataframe = DataFrameData(data)
        n, d = dataframe.int_df.shape

        if query_sets is None:
            domain_key_list = list(dataframe.columns)
            domain_value_count_list = [len(dataframe.values_by_col[key]) for key in domain_key_list]
            domain = Domain(domain_key_list, domain_value_count_list)
            query_sets = MST_selection(Dataset(dataframe.int_df, domain), epsilon, delta,
                                       cliques_to_include=column_feature_set)

        queries = FullMarginalQuerySet(query_sets, dataframe.values_by_col)
        queries = queries.get_canonical_queries()
        mnjax = MarkovNetwork(dataframe.values_by_col, queries)
        suff_stat = np.sum(queries.flatten()(dataframe.int_df.to_numpy()), axis=0)

        # determine Gaussian mech DP noise level for given privacy level
        sensitivity = np.sqrt(2 * len(query_sets))
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)

        # add DP noise (according to Gaussian mechanism)
        inference_rng, dp_rng = d3p.random.split(rng, 2)
        dp_noise = d3p.random.normal(dp_rng, suff_stat.shape) * sigma_DP
        dp_suff_stat = suff_stat + dp_noise

        inference_rng = d3p.random.convert_to_jax_rng_key(inference_rng)

        if use_laplace_approximation is True:
            approx_rng, mcmc_rng = jax.random.split(inference_rng, 2)
            laplace_approx, success = mei.run_numpyro_laplace_approximation(approx_rng, dp_suff_stat, n, sigma_DP,
                                                                            mnjax)
            mcmc, backtransform = mei.run_numpyro_mcmc_normalised(
                mcmc_rng, dp_suff_stat, n, sigma_DP, mnjax, laplace_approx, num_samples=2000, num_warmup=800,
                num_chains=4
            )
            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = backtransform(posterior_values.norm_lambdas.values.transpose())

        else:
            mcmc = mei.run_numpyro_mcmc(
                inference_rng, dp_suff_stat, n, sigma_DP, mnjax, num_samples=2000, num_warmup=800, num_chains=4
            )
            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = posterior_values.lambdas.values.transpose()

        return NapsuMQResult(mnjax, posterior_values, dataframe.data_description)


class NapsuMQResult(InferenceResult):
    """
    NAPSU-MQ result class containing learned differentially private probabilistic model from data.
    Contains functions to generate differentially private synthetic datasets from the original dataset.
    """

    def __init__(self, markov_network: MarkovNetwork, posterior_values: jnp.ndarray,
                 data_description: DataDescription) -> None:
        super().__init__()
        self._markov_network = markov_network
        self._posterior_values = posterior_values
        self._data_description = data_description

    @property
    def markov_network(self) -> MarkovNetwork:
        return self._markov_network

    @property
    def posterior_values(self) -> jnp.ndarray:
        return self._posterior_values

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    def _store_to_io(self, write_io: BinaryIO) -> None:
        assert write_io.writable()
        result = dill.dumps(self)
        return NapsuMQResultIO.store_to_io(write_io, result)

    @classmethod
    def _load_from_io(cls, read_io: BinaryIO, **kwargs) -> 'NapsuMQResult':
        return NapsuMQResultIO.load_from_io(read_io)

    def generate(
            self,
            rng: d3p.random.PRNGState,
            num_parameter_samples: int,
            num_data_per_parameter_sample: int = 1,
            single_dataframe: bool = True) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
        jax_rng = d3p.random.convert_to_jax_rng_key(rng)
        mnjax = self._markov_network
        posterior_values = self.posterior_values
        inds = jax.random.choice(key=jax_rng, a=posterior_values.shape[0], shape=[num_parameter_samples])
        posterior_sample = posterior_values[inds, :]
        rng, *data_keys = jax.random.split(jax_rng, num_parameter_samples + 1)
        syn_datasets = [mnjax.sample(syn_data_key, jnp.array(posterior_value), num_data_per_parameter_sample) for
                        syn_data_key, posterior_value
                        in list(zip(data_keys, posterior_sample))]

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
        result = dill.loads(result_binary)

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
        write_io.close()
