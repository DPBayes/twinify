# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © Ryan McKenna, © 2022- twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Originally from https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
# Modified by the twinify Developers under the Apache 2.0 license
# Modifications contain adding typing hints and changing privacy accounting method

from typing import Iterable, List, Tuple, Callable, Mapping

import numpy as np
import scipy.sparse
from twinify.napsu_mq.private_pgm.inference import FactoredInference
from twinify.napsu_mq.private_pgm.dataset import Dataset
from twinify.napsu_mq.private_pgm.domain import Domain
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools
import twinify.napsu_mq.privacy_accounting as accounting
from scipy.special import logsumexp

"""
This is a generalization of the winning mechanism from the
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.
"""


def MST_selection(data: Dataset, epsilon: float, delta: float, cliques_to_include: Iterable[Tuple[str, str]] = []) -> List:
    """Select marginal queries from dataset and cliques

    Args:
        data (Dataset): Dataset for selection
        epsilon (float): Epsilon for DP mechanism
        delta (float): Delta for DP mechanism
        cliques_to_include (List[Tuple]): Cliques to include in the queries

    Returns:
        Marginal queries for probabilistic model
    """

    rho = accounting.eps_delta_budget_to_rho_budget(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    cliques = select(data, rho / 3.0, log1, cliques=cliques_to_include)
    return cliques


def MST(data: Dataset, epsilon: float, delta: float) -> Dataset:
    """MST algorithm to generate synthetic data from sensitive dataset

    Args:
        data (Dataset): Dataset for MST
        epsilon (float): Epsilon for DP mechanism
        delta (float): Delta for DP mechanism

    Returns:
        Synthetic dataset
    """
    rho = accounting.eps_delta_budget_to_rho_budget(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    cliques = select(data, rho / 3.0, log1)
    log2 = measure(data, cliques, sigma)
    engine = FactoredInference(data.domain, iters=5000)
    est = engine.estimate(log1 + log2)
    synth = est.synthetic_data()
    return undo_compress_fn(synth)


def measure(data: Dataset, cliques: List, sigma: float, weights: np.ndarray = None) -> List[
    Tuple[scipy.sparse.coo_matrix, np.ndarray, float, List]]:
    """Measure marginals with noisy measurements

    Args:
        data (Dataset): Dataset
        cliques (List): Cliques to measure
        sigma (float): Noise scale
        weights (np.ndarray): Weight for each measurement

    Returns:
        Measurement log of noisy measurements from marginals
    """

    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    return measurements


def compress_domain(data: Dataset, measurements: Iterable[Tuple]) -> Tuple[Dataset, List, Callable]:
    """Compress domain for dataset

    Args:
        data (Dataset): Dataset to compress
        measurements (List[Tuple]): Measurement log

    Returns:
        Dataset in new domain, new measurements, function to undo compression
    """

    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3 * sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append((Q, y, sigma, proj))
        else:  # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append((I2, y2, sigma, proj))
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn


def exponential_mechanism(q: np.ndarray, eps: float, sensitivity: float, sampling_func: Callable = np.random.choice,
                          monotonic=False) -> np.ndarray:
    """Exponential mechanism for differential privacy

    Args:
        q (np.ndarray):
        eps (float): Epsilon for DP mechanism
        sensitivity (float): Sensitivity of the function
        sampling_func (function): Function to generate random sample
        monotonic: Is monotonic function

    Returns:
        Selected highly weighted pair with differential privacy
    """

    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    return sampling_func(q.size, p=probas)


def select(data: Dataset, rho: float, measurement_log: Iterable[Tuple],
           cliques: Iterable[Tuple[str, str]] = []) -> List:
    """Select the low dimensional marginals from dataset and measurements

    Args:
        data (Dataset): Dataset for selection
        rho (float): Privacy parameter for rho-zCDP
        measurement_log (Iterable[Tuple]): Log of noisy measurements
        cliques (Iterable[Tuple]): Cliques to include

    Returns:
        List of marginal queries
    """
    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8 * rho / (r - 1))
    for i in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.union(*e)

    return list(T.edges)


def transform_data(data: Dataset, supports: Mapping) -> Dataset:
    """Transform dataset to new domain

    Args:
        data (Dataset): Dataset
        supports (Support): Support

    Returns:
        Dataset in new domain
    """
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def reverse_data(data: Dataset, supports: Mapping) -> Dataset:
    """Dataset expressed in new domain

    Args:
        data (Dataset): Dataset
        supports (Mapping): Supports for new domain

    Returns:
        Dataset with new domain
    """
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)
