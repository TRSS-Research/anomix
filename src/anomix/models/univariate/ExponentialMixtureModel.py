import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps


class ExponentialMixtureModel(MixtureModel):
    _param_signature_ = {'scale': {'type': float, 'max': np.inf, 'min': 0}}
    _support = {'min': 0-eps, 'max': np.inf, 'dtype': float}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ExponentialMixtureModel"
        self.n_variables = 1
        self.zero_truncated = True

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[
        np.ndarray, Union[float, np.float64]]:
        # **params are scale= 1 / lambda
        likelihood = scipy.stats.expon.pdf(data, **params)
        likelihood_weighted = likelihood * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _fit_deterministic_(self, X: np.ndarray) -> Tuple[Params, np.ndarray]:
        # https://en.wikipedia.org/wiki/Exponential_distribution#Parameter_estimation
        lambd = 1 / X.mean()  # MLE estimate
        lambd = lambd - lambd / X.shape[0]  # bias corrected MLE estimate
        scale = 1 / lambd  # scipy parameterization uses loc and scale
        weights = np.array([1])
        return {'scale': scale.reshape(-1)}, weights

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:

        weights = np.ones(n_components) / n_components
        lambdas = 1 / (X.mean() * (np.random.random(n_components) + .5))
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
            r = (likelihood_weighted + eps) / (likelihood_weighted + self.eps).sum(axis=1)[..., np.newaxis]

            # Maximization
            lambdas = 1 / np.average(X_rep, weights=r, axis=0)
            weights = r.sum(axis=0) / r.sum()

        return {'scale': 1 / lambdas}, weights, log_likelihoods[-1]

    def _generate_component(self, component, n):
        return scipy.stats.expon.rvs(scale=self.params['scale'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.expon.cdf(X, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.expon.sf(X, **self.params)