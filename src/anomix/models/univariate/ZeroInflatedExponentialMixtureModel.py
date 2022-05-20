import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import  Params, eps


class ZeroInflatedExponentialMixtureModel(MixtureModel):
    _param_signature_ = {'scale': {'type': float, 'max': np.inf, 'min': 0}}
    _support = {'min': 0 - eps, 'max': np.inf, 'dtype': float}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ZeroInflatedExponentialMixtureModel"
        self.zero_inflated = True
        self.min_components = 2
        self.n_variables = 1

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[
        np.ndarray, Union[float, np.float64]]:
        # **params are scale = 1 / lambda
        likelihood_zero = (data[:, 0] == 0).astype(np.int64)[:, np.newaxis]
        likelihood = scipy.stats.expon.pdf(data[:, 1:], **params)
        likelihood[data[:, 0] == 0] = 0  # put them to zero

        likelihood_weighted = np.concatenate((likelihood_zero, likelihood), axis=1) * weights

        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _fit_deterministic_(self, X: np.ndarray) -> Tuple[Params, np.ndarray]:
        # https://en.wikipedia.org/wiki/Exponential_distribution#Parameter_estimation
        params, weights, ll = self._fit_(X, self.min_components)
        return params, weights

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        pi = (X == 0).mean()
        weights = np.hstack((pi, (1 - pi) * np.ones(n_components - 1) / (n_components - 1)))

        lambdas = 1 / (X.mean() * (np.random.random(n_components - 1) + .5))
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=1)
        log_likelihoods = [-np.inf]

        for iteration in range(self.max_iterations):
            # calculate the maximum likelihood of each observation xi

            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, scale=1 / lambdas)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break
            r = (eps + likelihood_weighted) / (likelihood_weighted + self.eps).sum(axis=1)[..., np.newaxis]
            # Maximization
            lambdas = 1 / np.average(X_rep[:, 1:], weights=r[:, 1:], axis=0)
            weights = r.sum(axis=0) / r.sum()

        lambdas[lambdas == np.inf] = 1 / self.eps  # hotfix
        return {'scale': 1 / lambdas}, weights, log_likelihoods[-1]

    def _cdf_separate(self, X):
        zero_component = (X[:, 0] <= 0).astype(float)[..., np.newaxis]
        non_zero = scipy.stats.expon.cdf(X, **self.params)
        return np.concatenate((zero_component, non_zero), axis=-1)

    def _sf_separate(self, X):
        zero_component = (X[:, 0] >= 0).astype(float)[..., np.newaxis]
        non_zero = scipy.stats.expon.sf(X, **self.params)
        return np.concatenate((zero_component, non_zero), axis=-1)

    def _generate_component(self, component, n):
        return scipy.stats.expon.rvs(scale=self.params['scale'][component-1], size=n)
