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

# Originally from https://github.com/ryan112358/private-pgm/blob/557c077708d3559212a8f65dff3eccd3fd244abb/src/mbi/inference.py
# Modified by the twinify Developers under the Apache 2.0 license
# Modified import statements and removed support for torch backend

import numpy as np
from twinify.napsu_mq.private_pgm.graphical_model import GraphicalModel
from twinify.napsu_mq.private_pgm.callbacks import Logger
from twinify.napsu_mq.private_pgm.clique_vector import CliqueVector
from twinify.napsu_mq.private_pgm.factor import Factor
from scipy.sparse.linalg import eigsh, lsmr, aslinearoperator
from scipy import sparse
from collections import defaultdict


class FactoredInference:
    def __init__(self, domain, backend='numpy', structural_zeros={}, metric='L2', log=False, iters=1000,
                 warm_start=False, elim_order=None):
        """
        Class for learning a GraphicalModel from  noisy measurements on a data distribution

        :param domain: The domain information (A Domain object)
        :param backend: numpy or torch backend
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.backend = backend
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.elim_order = elim_order
        self.Factor = Factor

        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom, fact)

    def estimate(self, measurements, total=None, engine='MD', callback=None, options={}):
        """
        Estimate a GraphicalModel from the given measurements
        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param engine: the optimization algorithm to use, options include:
            MD - Mirror Descent with armijo line search
            RDA - Regularized Dual Averaging
            IG - Interior Gradient
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }

        :return model: A GraphicalModel that best matches the measurements taken
        """
        measurements = self.fix_measurements(measurements)
        options['callback'] = callback
        if callback is None and self.log:
            options['callback'] = Logger(self)
        if engine == 'MD':
            self.mirror_descent(measurements, total, **options)
        elif engine == 'RDA':
            self.dual_averaging(measurements, total, **options)
        elif engine == 'IG':
            self.interior_gradient(measurements, total, **options)
        return self.model

    def fix_measurements(self, measurements):
        assert type(measurements) is list, 'measurements must be a list, given ' + measurements
        assert all(len(m) == 4 for m in measurements), \
            'each measurement must be a 4-tuple (Q, y, noise,proj)'
        ans = []
        for Q, y, noise, proj in measurements:
            assert Q is None or Q.shape[0] == y.size, 'shapes of Q and y are not compatible'
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            assert np.isscalar(noise), 'noise must be a real value, given ' + str(noise)
            assert all(a in self.domain for a in proj), str(proj) + ' not contained in domain'
            assert Q.shape[1] == self.domain.size(proj), 'shapes of Q and proj are not compatible'
            ans.append((Q, y, noise, proj))
        return ans

    def interior_gradient(self, measurements, total, lipschitz=None, c=1, sigma=1, callback=None):
        """ Use the interior gradient algorithm to estimate the GraphicalModel
            See https://epubs.siam.org/doi/pdf/10.1137/S1052623403427823 for more information
        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param c, sigma: parameters of the algorithm
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipschitz is not None, 'lipschitz constant must be supplied'
        self._setup(measurements, total)
        # what are c and sigma?  For now using 1
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        if self.log:
            print('Lipchitz constant:', L)

        theta = model.potentials
        x = y = z = model.belief_propagation(theta)
        c0 = c
        l = sigma / L
        for k in range(1, self.iters + 1):
            a = (np.sqrt((c * l) ** 2 + 4 * c * l) - l * c) / 2
            y = (1 - a) * x + a * z
            c *= (1 - a)
            _, g = self._marginal_loss(y)
            theta = theta - a / c / total * g
            z = model.belief_propagation(theta)
            x = (1 - a) * x + a * z
            if callback is not None:
                callback(x)

        model.marginals = x
        model.potentials = model.mle(x)

    def dual_averaging(self, measurements, total=None, lipschitz=None, callback=None):
        """ Use the regularized dual averaging algorithm to estimate the GraphicalModel
            See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf
        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipschitz is not None, 'lipschitz constant must be supplied'
        self._setup(measurements, total)
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        print('Lipchitz constant:', L)
        if L == 0: return

        theta = model.potentials
        gbar = CliqueVector({cl: self.Factor.zeros(domain.project(cl)) for cl in cliques})
        w = v = model.belief_propagation(theta)
        beta = 0

        for t in range(1, self.iters + 1):
            c = 2.0 / (t + 1)
            u = (1 - c) * w + c * v
            _, g = self._marginal_loss(u)  # not interested in loss of this query point
            gbar = (1 - c) * gbar + c * g
            theta = -t * (t + 1) / (4 * L + beta) / self.model.total * gbar
            v = model.belief_propagation(theta)
            w = (1 - c) * w + c * v

            if callback is not None:
                callback(w)

        model.marginals = w
        model.potentials = model.mle(w)

    def mirror_descent(self, measurements, total=None, stepsize=None, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param stepsize: The step size function for the optimization (None or scalar or function)
            if None, will perform line search at each iteration (requires smooth objective)
            if scalar, will use constant step size
            if function, will be called with the iteration number
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        """
        assert not (self.metric == 'L1' and stepsize is None), \
            'loss function not smooth, cannot use line search (specify stepsize)'

        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)
        if ans[0] == 0:
            return ans[0]

        nols = stepsize is not None
        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total ** 2
            stepsize = lambda t: 2.0 * alpha

        for t in range(1, self.iters + 1):
            if callback is not None:
                callback(mu)
            omega, nu = theta, mu
            curr_loss, dL = ans
            # print('Gradient Norm', np.sqrt(dL.dot(dL)))
            alpha = stepsize(t)
            for i in range(25):
                theta = omega - alpha * dL
                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)
                if nols or curr_loss - ans[0] >= 0.5 * alpha * dL.dot(nu - mu):
                    break
                alpha *= 0.5

        model.potentials = theta
        model.marginals = mu

        return ans[0]

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals
        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.datavector()
                diff = c * (Q @ x - y)
                if metric == 'L1':
                    loss += abs(diff).sum()
                    sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff)
                    grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)

    def _setup(self, measurements, total):
        """ Perform necessary setup for running estimation algorithms

        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, noise ** 2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)

        # if not self.warm_start or not hasattr(self, 'model'):
        # initialize the model and parameters
        cliques = [m[3] for m in measurements]
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())

        model = GraphicalModel(self.domain, cliques, total, elimination_order=self.elim_order)

        model.potentials = CliqueVector.zeros(self.domain, model.cliques)
        model.potentials.combine(self.structural_zeros)
        if self.warm_start and hasattr(self, 'model'):
            model.potentials.combine(self.model.potentials)
        self.model = model

        # group the measurements into model cliques
        cliques = self.model.cliques
        # self.groups = { cl : [] for cl in cliques }
        self.groups = defaultdict(lambda: [])
        for Q, y, noise, proj in measurements:
            m = (Q, y, noise, proj)
            for cl in sorted(cliques, key=model.domain.size):
                # (Q, y, noise, proj) tuple
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break

    def _lipschitz(self, measurements):
        """ compute lipschitz constant for L2 loss
            Note: must be called after _setup
        """
        eigs = {cl: 0.0 for cl in self.model.cliques}
        for Q, _, noise, proj in measurements:
            for cl in self.model.cliques:
                if set(proj) <= set(cl):
                    n = self.domain.size(cl)
                    p = self.domain.size(proj)
                    Q = aslinearoperator(Q)
                    Q.dtype = np.dtype(Q.dtype)
                    eig = eigsh(Q.H * Q, 1)[0][0]
                    eigs[cl] += eig * n / p / noise ** 2
                    break
        return max(eigs.values())

    def infer(self, measurements, total=None, engine='MD', callback=None, options={}):
        import warnings
        message = "Function infer is deprecated.  Please use estimate instead."
        warnings.warn(message, DeprecationWarning)
        return self.estimate(measurements, total, engine, callback, options)