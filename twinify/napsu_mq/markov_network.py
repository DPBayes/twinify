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
from typing import Optional, List, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from twinify.napsu_mq.junction_tree import JunctionTree
from twinify.napsu_mq.undirected_graph import UndirectedGraph, greedy_ordering
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.napsu_mq.log_factor import LogFactor
import functools


class MarkovNetwork:
    """Jax implementation of a Markov network representation for MED."""

    def __init__(self, domain: Dict, queries: FullMarginalQuerySet, elimination_order: Optional[Iterable] = None,
                 debug_checks: Optional[bool] = True):
        """Create the Markov network.
        Args:
            domain (dict): A dict containing the possible values for each variable.
            queries (FullMarginalQuerySet): The queries forming the sufficient statistic.
            elimination_order (list, optional): Elimination order for variable elimination. Defaults to using a greedy ordering.
            debug_checks (bool, optional): Whether to enable debug checks for factor computations. Defaults to True.
        """

        self.debug_checks = debug_checks
        self.domain = domain
        self.queries = queries
        self.d = len(self.domain.keys())
        variables_in_queries = set.union(*[set(feature_set) for feature_set in self.queries.feature_sets])
        self.variables_not_in_queries = set(domain.keys()).difference(variables_in_queries)

        self.graph = UndirectedGraph.from_clique_list(
            queries.feature_sets + [(var,) for var in self.variables_not_in_queries])
        if elimination_order is None:
            elimination_order = greedy_ordering(self.graph)
        self.elimination_order = elimination_order
        self.junction_tree = JunctionTree.from_variable_elimination(queries.feature_sets, self.elimination_order)
        self.junction_tree.remove_redundant_nodes()

        self.flat_queries = self.queries.flatten()
        self.variable_associations = self.flat_queries.get_variable_associations(use_features=True)
        self.suff_stat_d = len(self.flat_queries.queries)
        self.lambda_d = self.suff_stat_d

        self.suff_stat_mean = jax.jit(jax.grad(self.lambda0))
        self.suff_stat_cov = jax.jit(jax.hessian(self.lambda0))
        self.suff_stat_mean_bp = jax.jit(self.suff_stat_mean_bp)
        self.suff_stat_cov_bp = jax.jit(jax.jacrev(self.suff_stat_mean_bp))
        self.log_factor_class = LogFactor

    def suff_stat_mean_and_cov_bp(self, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sufficient statistic mean and covariance for given parameter values with belief propagation.
        Args:
            lambdas (numpy array): The parameter values.
        Returns:
            (numpy array, numpy array): (mean, covariance)
        """
        return self.suff_stat_mean_bp(lambdas), self.suff_stat_cov_bp(lambdas)

    def suff_stat_mean_and_cov(self, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sufficient statistic mean and covariance for given parameter values with.
        Args:
            lambdas (numpy array): The parameter values.
        Returns:
            (numpy array, numpy array): (mean, covariance)
        """
        return self.suff_stat_mean(lambdas), self.suff_stat_cov(lambdas)

    def marginal_distribution_logits(self, factors: Iterable[LogFactor], variables: List):
        to_eliminate = [var for var in self.elimination_order if var not in variables]
        result_factor = self.variable_elimination(factors, to_eliminate)
        proj_factor = result_factor.project(variables)
        proj_factor.ensure_batch_is_first_dim()
        return proj_factor.values

    def variable_elimination(self, factors: Iterable[LogFactor], variables_to_eliminate: Iterable) -> 'LogFactor':
        """Run variable elimination.
        Args:
            factors (list(LogFactor)): The factors used in variable elimination from self.compute_factors.
            variables_to_eliminate (list): The variables to eliminate.
        Returns:
            LogFactor: The result of variable elimination as a LogFactor.
        """
        for variable in variables_to_eliminate:
            factors = self.eliminate_var(factors, variable)

        return LogFactor.list_product(factors)

    @staticmethod
    def eliminate_var(factors: Iterable['LogFactor'], variable):
        factors_in_prod = [factor for factor in factors if variable in factor.scope]
        factors_not_in_prod = [factor for factor in factors if variable not in factor.scope]

        if len(factors_in_prod) == 0:
            return factors_not_in_prod
        log_prod_factor = LogFactor.list_product(factors_in_prod)

        summed_log_factor = log_prod_factor.marginalise(variable)
        factors_not_in_prod.append(summed_log_factor)
        return factors_not_in_prod

    def belief_propagation(self, factors: Iterable[LogFactor]) -> Dict:
        """Run belief propagation.
        Args:
            factors (list(LogFactor)): The factors used in belief propagation from self.compute_factors
        Returns:
            dict: Dict containing the LogFactor for each set factor scope in the Markov network.
        """

        self.junction_tree.init_factor_assignments()
        self.junction_tree.init_node_orders()

        for node in self.junction_tree.upward_order:
            node.reset()
            node.potential = LogFactor.list_product(
                factor for factor in factors if factor.scope in self.junction_tree.factors_in_node[node.variables]
            )
        for node in self.junction_tree.upward_order:
            if node.parent is not None:
                self.bp_message(node, node.parent)
        for node in self.junction_tree.downward_order:
            for child in node.children:
                self.bp_message(node, child)
        for node in self.junction_tree.downward_order:
            node.result = LogFactor.list_product([node.potential] + [message for _, message in node.messages])

        return {node.variables: node.result for node in self.junction_tree.downward_order}

    def bp_message(self, sender, receiver):
        product = LogFactor.list_product(
            [sender.potential] + [message for mes_sender, message in sender.messages if mes_sender is not receiver])
        edges = self.junction_tree.edges
        n1 = sender.variables
        n2 = receiver.variables
        separator = edges[(n1, n2)] if (n1, n2) in edges.keys() else edges[(n2, n1)]
        for variable in set(sender.variables).difference(set(separator)):
            product = product.marginalise(variable)
        receiver.messages.append((sender, product))

    @functools.partial(jax.jit, static_argnums=0)
    def lambda0(self, lambdas: jnp.ndarray) -> jnp.ndarray:
        factors = self.compute_factors(lambdas)
        result_factor = self.variable_elimination(factors, self.elimination_order)
        return result_factor.values

    def suff_stat_mean_bp(self, lambdas: jnp.ndarray) -> jnp.ndarray:
        factors = self.compute_factors(lambdas)
        result_factors = self.belief_propagation(factors)
        result: jnp.ndarray = jnp.zeros(self.suff_stat_d)
        for clique, indices in self.variable_associations.items():
            node_variables = self.junction_tree.node_for_factor[clique]
            factor = result_factors[node_variables]
            for variable in set(node_variables).difference(clique):
                factor = factor.marginalise(variable)
            result = result.at[jnp.array(indices)].set(factor.query(self.queries.queries[clique]))
        return result

    def sample(self, rng: jax.random.PRNGKey, lambdas: jnp.ndarray, n_sample: Optional[int] = 1) -> pd.DataFrame:
        """Sample the distribution with given parameter values.
        Args:
            lambdas (numpy array): The parameter values.
            n_sample (int, optional): The number of samples to generate. Defaults to 1.
        Returns:
            array (Pandas Dataframe): The generated samples.
        """
        n_cols = len(self.domain.keys())
        cols = self.domain.keys()
        data = np.zeros((n_sample, n_cols), dtype=np.int64)
        df = pd.DataFrame(data, columns=cols, dtype=int)
        order = self.elimination_order[::-1]
        batch_factors = [factor.add_batch_dim(n_sample) for factor in self.compute_factors(lambdas)]
        for variable in order:
            marginal = self.marginal_distribution_logits(batch_factors, [variable])
            rng, key = jax.random.split(rng)
            values = jax.random.categorical(key, marginal)
            batch_factors = [factor.batch_condition(variable, values) if variable in factor.scope else factor for factor
                             in batch_factors]
            df.loc[:, variable] = values

        return df

    def log_factor_vector(self, lambdas: jnp.ndarray, variables: Iterable) -> jnp.ndarray:
        vec = jnp.zeros(tuple(len(self.domain[var]) for var in variables))
        for query_ind in self.variable_associations[variables]:
            query_val = jnp.array(self.flat_queries.queries[query_ind].value)
            vec = vec.at[tuple(query_val)].set(lambdas[query_ind])
        return vec

    def compute_factors(self, lambdas: jnp.ndarray) -> List[LogFactor]:
        """Compute the LogFactor objects used by other methods for given parameters.
        Args:
            lambdas (numpy array): The parameters.
        Returns:
            list(LogFactor): The resulting factors.
        """
        return [
                   LogFactor(factor_scope, self.log_factor_vector(lambdas, factor_scope), self.debug_checks)
                   for factor_scope in self.variable_associations.keys()
               ] + [
                   LogFactor((variable,), jnp.zeros(len(self.domain[variable])), self.debug_checks)
                   for variable in self.variables_not_in_queries
               ]
