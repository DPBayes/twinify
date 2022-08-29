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
from typing import Optional, Union, Iterable, BinaryIO, Callable, Mapping
import os
import pandas as pd

import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from mbi import Domain, Dataset
import dill

from twinify.napsu_mq.mst import MST_selection
from twinify.base import InferenceModel, InferenceResult, InvalidFileFormatException
from twinify.napsu_mq.markov_network_jax import MarkovNetworkJax
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.napsu_mq.dataframe_data import DataFrameData
import twinify.napsu_mq.privacy_accounting as privacy_accounting
import twinify.napsu_mq.maximum_entropy_inference as mei

from d3p.random import PRNGState


class NapsuMQModel(InferenceModel):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: pd.DataFrame, rng: PRNGState, epsilon: float, delta: float, **kwargs) -> 'NapsuMQResult':
        column_feature_set = kwargs['column_feature_set'] if 'column_feature_set' in kwargs else []
        use_laplace_approximation = kwargs[
            'use_laplace_approximation'] if 'use_laplace_approximation' in kwargs else False

        dataframe = DataFrameData(data)
        category_mapping = DataFrameData.get_category_mapping(data)
        n, d = dataframe.int_array.shape
        domain_key_list = list(dataframe.values_by_col.keys())
        domain_value_count_list = [len(dataframe.values_by_col[key]) for key in domain_key_list]
        domain = Domain(domain_key_list, domain_value_count_list)
        query_sets = MST_selection(Dataset(dataframe.int_df, domain), epsilon, delta,
                                   cliques_to_include=column_feature_set)
        queries = FullMarginalQuerySet(query_sets, dataframe.values_by_col)
        queries = queries.get_canonical_queries()
        mnjax = MarkovNetworkJax(dataframe.values_by_col, queries)
        suff_stat = np.sum(queries.flatten()(dataframe.int_array), axis=0)
        sensitivity = np.sqrt(2 * len(query_sets))
        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)
        dp_suff_stat = jnp.asarray(np.random.normal(loc=suff_stat, scale=sigma_DP))

        if use_laplace_approximation is True:
            laplace_approx, success = mei.run_numpyro_laplace_approximation(rng, dp_suff_stat, n, sigma_DP, mnjax)
            mcmc, backtransform = mei.run_numpyro_mcmc_normalised(
                rng, dp_suff_stat, n, sigma_DP, mnjax, laplace_approx, num_samples=2000, num_warmup=800, num_chains=4
            )
            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = backtransform(posterior_values.norm_lambdas.values.transpose())

        else:
            mcmc = mei.run_numpyro_mcmc(
                rng, dp_suff_stat, n, sigma_DP, mnjax, num_samples=2000, num_warmup=800, num_chains=4
            )
            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = posterior_values.lambdas.values.transpose()

        return NapsuMQResult(mnjax, posterior_values, category_mapping)


class NapsuMQResult(InferenceResult):

    def __init__(self, markov_network: MarkovNetworkJax, posterior_values: jnp.ndarray,
                 category_mapping: Mapping) -> None:
        super().__init__()
        self._markov_network = markov_network
        self._posterior_values = posterior_values
        self._category_mapping = category_mapping

    @property
    def markov_network(self) -> MarkovNetworkJax:
        return self._markov_network

    @property
    def posterior_values(self) -> jnp.ndarray:
        return self._posterior_values

    @property
    def category_mapping(self) -> Mapping:
        return self._category_mapping

    def _store_to_io(self, write_io: BinaryIO) -> None:
        assert write_io.writable()
        result = dill.dumps(self)
        return NapsuMQResultIO.store_to_io(write_io, result)

    @classmethod
    def _load_from_io(cls, read_io: BinaryIO) -> 'NapsuMQResult':
        return NapsuMQResultIO.load_from_io(read_io)

    def _generate(self,
                  rng: PRNGState,
                  dataset_size: int,
                  num_datasets: Optional[int] = 1,
                  ) -> Iterable[pd.DataFrame]:
        mnjax = self._markov_network
        posterior_values = self.posterior_values
        inds = jax.random.choice(key=rng, a=posterior_values.shape[0], shape=[num_datasets])
        posterior_sample = posterior_values[inds, :]
        rng, *data_keys = jax.random.split(rng, num_datasets + 1)
        syn_datasets = [mnjax.sample(syn_data_key, jnp.array(posterior_value), dataset_size) for
                        syn_data_key, posterior_value
                        in list(zip(data_keys, posterior_sample))]

        categorical_syn_data_dfs = [DataFrameData.apply_category_mapping(syn_data, self._category_mapping) for syn_data
                                    in syn_datasets]

        return categorical_syn_data_dfs

    def generate_extended(self, rng: PRNGState, num_data_per_parameter_sample: int, num_parameter_samples: int,
                          single_dataframe: Optional[bool] = False) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:

        dataframes = self._generate(rng, num_data_per_parameter_sample, num_parameter_samples)

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
