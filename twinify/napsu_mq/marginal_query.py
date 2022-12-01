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

from typing import Iterable, List, Optional, Tuple, Dict, Set

import itertools
import numpy as np

from tqdm import tqdm

from twinify.napsu_mq.utils import powerset


class MarginalQuery:
    def __init__(self, inds: Iterable[int], value: Iterable[int], features: Optional[Iterable] = None):
        """Create the marginal query object.
        Args:
            inds (iterable(int)): Indices of the marginal query. Converted to list.
            value (iterable(int)): Values of the marginal query. Converted to np.ndarray.
            features (list, optional): Variable names corresponding to the indices. Defaults to inds.
        """
        self.inds = list(inds)
        self.features = self.inds if features is None else features
        self.value = value if type(value) is np.ndarray else np.array(value)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x[:, self.inds] == self.value).all(axis=1).astype(int).reshape((-1, 1))

    def __str__(self) -> str:
        return "{} = {}".format(self.inds, [val.item() for val in self.value])

    def as_tuple(self) -> Tuple:
        """Convert to tuple for easy equality comparisons.
        Returns:
            tuple: (indices, value)
        """
        return tuple(self.inds), self.value_tuple()

    def value_tuple(self) -> Tuple:
        """Convert self.value to tuple.
        Returns:
            tuple: self.value as a tuple.
        """
        return tuple([val.item() for val in self.value])


class FullMarginalQuerySet:
    """A full marginal query set."""

    def __init__(self, feature_sets: Iterable[Tuple], values_by_feature: Dict):
        """Create the full marginal query set.
        Args:
            feature_sets (list(tuple)): The tuples of variables that are used as indices.
            values_by_feature (dict): A dict containing a list of possible values for each variable.
        """
        self.feature_sets = list(feature_sets)
        self.values_by_feature = values_by_feature

        self.feature_by_index = list(self.values_by_feature.keys())

        self.int_feature_sets = [
            tuple(self.feature_by_index.index(feature) for feature in feature_set)
            for feature_set in self.feature_sets
        ]

        self.values_by_int_feature = {
            self.feature_by_index.index(feature): values
            for feature, values in self.values_by_feature.items()
        }

        self.queries = {feature_set: QueryList(all_marginals_for_feature_set(feature_set, values_by_feature)) for
                        feature_set in feature_sets}

    def query(self, x: np.ndarray) -> Dict:
        """Run all marginal queries on output.
        Args:
            x (np.ndarray): The output.
        Returns:
            dict: A dictionary containing the feature sets as keys and query results as values.
        """
        return {feature_set: self.queries[feature_set](x) for feature_set in self.feature_sets}

    def query_sum(self, x: np.ndarray) -> Dict:
        """As self.query, except the results are summed over datapoints.
        Args:
            x (np.ndarray): The output.
        Returns:
            dict: A dictionary containing the feature sets as keys and summed query results as values.
        """
        return {feature_set: result.sum(axis=0) for feature_set, result in self.query(x).items()}

    def query_feature_set(self, feature_set: Tuple, x: np.ndarray) -> np.ndarray:
        """Query one feature set.
        Args:
            feature_set (tuple): The feature set to query.
            x (np.ndarray): The output.
        Returns:
            np.ndarray: The query results.
        """
        return self.queries[feature_set](x)

    def query_feature_set_sum(self, feature_set: Tuple, x: np.ndarray) -> np.ndarray:
        """As self.query_feature_set, except results are summed over datapoints.
        Args:
            feature_set (tuple): The feature set to query.
            x (np.ndarray): The output.
        Returns:
            np.ndarray: The summed query results.
        """
        return self.query_feature_set(feature_set, x).sum(axis=0)

    def flatten(self) -> 'QueryList':
        """Convert self to a QueryList of the contained queries.
        Returns:
            QueryList: The resulting QueryList.
        """
        return QueryList(itertools.chain.from_iterable(query_list.queries for query_list in self.queries.values()))

    def get_canonical_queries(self) -> 'FullMarginalQuerySet':
        """Find the canonical queries for the queries in self.
        Returns:
            FullMarginalQuerySet: The canonical queries as a FullMarginalQuerySet.
        """
        d = len(self.values_by_feature.keys())
        base_value = np.zeros(d, dtype=int)
        clique_list = list(itertools.chain.from_iterable(powerset(features) for features in self.int_feature_sets))
        clique_set = set([tuple(val) for val in clique_list])
        canonical_queries = {}

        for clique in tqdm(clique_set):
            clique = set(clique)
            clique_ordered = tuple(clique)
            if clique == set():
                continue
            index_conversion = np.full((d,), -1)
            for i, variable in enumerate(clique_ordered):
                index_conversion[variable] = i
            conv_clique_indices = index_conversion[list(clique)]

            clique_product = itertools.product(*(self.values_by_int_feature[variable] for variable in clique_ordered))
            for val in tqdm(clique_product):
                value = np.zeros(len(clique), dtype=int)
                value[conv_clique_indices] = np.asarray(val, dtype=int)
                counter = np.zeros(tuple([len(self.values_by_int_feature[variable]) for variable in clique_ordered]),
                                      dtype=int)
                subsets = powerset(clique)
                for subset in subsets:
                    multiplier = (-1) ** (len(clique) - len(subset))
                    base_indices = list(clique.difference(subset))
                    conv_base_indices = index_conversion[base_indices]
                    completed_value = value.copy()
                    completed_value[conv_base_indices] = base_value[conv_base_indices]
                    counter[tuple(completed_value)] += multiplier

                clique_ordered_not_int = tuple(self.feature_by_index[feature] for feature in clique_ordered)
                if not (counter == 0).all():
                    if clique_ordered_not_int not in canonical_queries.keys():
                        canonical_queries[clique_ordered_not_int] = []
                    canonical_queries[clique_ordered_not_int].append(
                        MarginalQuery(clique_ordered, val, features=clique_ordered_not_int)
                    )

        added_query_tuples = {query.as_tuple() for query in itertools.chain.from_iterable(canonical_queries.values())}
        not_original_clique_queries = list(itertools.chain.from_iterable(
            queries for clique, queries in canonical_queries.items() if clique not in self.feature_sets))
        original_clique_queries = {clique: queries for clique, queries in canonical_queries.items() if
                                   clique in self.feature_sets}

        for clique in tqdm(self.feature_sets):
            if clique not in original_clique_queries.keys():
                original_clique_queries[clique] = []

        for query in tqdm(not_original_clique_queries):
            original_clique = None
            for clique in self.feature_sets:
                if set(query.features).issubset(set(clique)):
                    original_clique = clique
                    break
            features_to_sum = set(clique).difference(set(query.features))
            values_to_sum = itertools.product(*(self.values_by_feature[feature] if feature in features_to_sum else (
                query.value[query.features.index(feature)],) for feature in original_clique))
            new_queries = [
                MarginalQuery(tuple(self.feature_by_index.index(feature) for feature in original_clique), value,
                              features=original_clique)
                for value in values_to_sum
            ]
            for new_query in new_queries:
                if new_query.as_tuple() not in added_query_tuples:
                    added_query_tuples.add(new_query.as_tuple())
                    original_clique_queries[original_clique].append(new_query)

        canonical_queries = {key: QueryList(queries) for key, queries in original_clique_queries.items()}
        new_fmqs = FullMarginalQuerySet([], self.values_by_feature)
        new_fmqs.queries = canonical_queries
        new_fmqs.feature_sets = list(canonical_queries.keys())
        return new_fmqs


class QueryList:
    """A list of MarginalQuery objects."""

    def __init__(self, queries: Iterable[MarginalQuery]):
        """Create the QueryList.
        Args:
            queries (iterable(MarginalQuery)): The MarginalQuery objects.
        """
        self.queries = list(queries)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([query(x) for query in self.queries], axis=1)

    def __str__(self):
        return "\n".join(query.__str__() for query in self.queries)

    def evaluate_subset(self, x: np.ndarray, subset: Iterable[int]) -> np.ndarray:
        """Evaluate a subset of the queries.
        Args:
            x (np.ndarray): The output.
            subset (iterable(int)): The indices for the queries to evaluate.
        Returns:
            np.ndarray: The query results.
        """
        return np.concatenate([self.queries[i](x) for i in subset], axis=1)

    def get_subset(self, subset: Iterable[int]) -> 'QueryList':
        """Get a subset of the queries.
        Args:
            subset (iterable(int)): The indices for the queries to return.
        Returns:
            QueryList: The selected queries as a new QueryList object.
        """
        return QueryList([self.queries[i] for i in subset])

    def as_tuple_set(self) -> Set[Tuple]:
        """Get a set of tuples representing the queries.
        Returns:
            set(tuple): A set of tuples of each query.
        """
        return {query.as_tuple() for query in self.queries}

    def get_variable_associations(self, use_features: Optional[bool] = False) -> Dict:
        """Get a dict containing a list of indices into self.queries for each set of indices that self.queries look at.
        Args:
            use_features (bool, optional): Use query features as keys instead of query indices. Defaults to False.
        Returns:
            dict: A dict with tuples of query indices or features as keys and a list of indices of queries with that set of indices or features.
        """
        result = {}
        for i, query in enumerate(self.queries):
            inds_tuple = tuple(query.inds if not use_features else query.features)
            if inds_tuple in result.keys():
                result[inds_tuple].append(i)
            else:
                result[inds_tuple] = [i]
        return result


def join_query_sets(query_sets: Iterable[QueryList]) -> QueryList:
    """Combine several query lists.
    Args:
        query_sets (iterable(QueryList)): The QueryLists to combine.
    Returns:
        QueryList: The combined QueryList.
    """
    return QueryList(list(itertools.chain.from_iterable(query_set.queries for query_set in query_sets)))


def all_marginals_for_feature_set(feature_set: Iterable, values_by_feature: Dict) -> List[MarginalQuery]:
    inds = [list(values_by_feature.keys()).index(feature) for feature in feature_set]
    all_values = list(itertools.product(*[sorted(values_by_feature[feature]) for feature in feature_set]))
    return [MarginalQuery(inds, value, feature_set) for value in all_values]


def all_marginals(feature_sets: Iterable[Tuple], values_by_feature: Dict) -> QueryList:
    """Get all marginal queries for given sets of variables.
    Args:
        feature_sets (list(tuple)): The sets of variables.
        values_by_feature (dict): A dict containing the possible values for each variable.
    Returns:
        QueryList: All marginals for the given sets of variables as a QueryList.
    """
    queries = []
    for feature_set in feature_sets:
        queries_for_feature_set = all_marginals_for_feature_set(feature_set, values_by_feature)
        for query in queries_for_feature_set:
            queries.append(query)
    return QueryList(queries)


def column_feature_sets_to_indices(column_feature_sets: Iterable[Tuple[str, str]],
                                   columns: List[str]) -> List[Tuple[int, int]]:
    int_feature_sets = []
    for pair in column_feature_sets:
        if pair[0] not in columns:
            raise ValueError(f"Variable {pair[0]} not found in the column list")
        if pair[1] not in columns:
            raise ValueError(f"Variable {pair[1]} not found in the column list")

        int_feature_sets.append((
            columns.index(pair[0]), columns.index(pair[1])
        ))

    return int_feature_sets
