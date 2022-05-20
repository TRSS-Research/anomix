import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps


class ZeroInflatedPoissonMixtureModel(MixtureModel):
    _param_signature_ = {'mu': {'type': float, 'max': np.inf, 'min': 0},}
    _support = {'min': 0-eps, 'max': np.inf, 'dtype': int}
    pdf_thresholder = 10e-8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'ZeroInflatedPoissonMixtureModel'
        self.zero_inflated = True
        self.skip_deterministic = True
        self.min_components = 2
        self.n_variables = 1
        self.discrete = True

    def likelihood(self, weights: np.ndarray, data: np.ndarray,
                   log=False, **params) -> Union[np.ndarray, Union[float, np.float64]]:

        likelihood_poisson = scipy.stats.poisson.pmf(data[:, 1:], **params)
        likelihood_zero = (data[:, 0] == 0).astype(np.float64)[:, np.newaxis]
        likelihood_weighted = np.concatenate((likelihood_zero, likelihood_poisson), axis=1) * weights

        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _fit_deterministic(self, X):
        return self._fit_(X, self.min_components)

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        pi = (X == 0).mean()
        weights = np.hstack((pi, (1 - pi) * np.ones(n_components - 1) / (n_components - 1)))
        unique, counts = np.unique(X[X > 0], return_counts=True)
        try:
            rate = np.random.choice(unique, n_components - 1, replace=False,
                                    p=counts / counts.sum()).astype(np.float32)
        except ValueError:  # not enough variety
            rate = X[X != 0].mean() * (np.random.random(n_components - 1) / (n_components - 1) + .5)

        log_likelihoods = [-np.inf]
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=1)
        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, mu=rate)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)
            # if abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-2:
            #     break
            if self.check_convergence(log_likelihoods):
                break
            r = (eps + likelihood_weighted) / (likelihood_weighted + self.eps).sum(axis=1)[..., np.newaxis]

            # maximization
            rate = (r[:, 1:] * X_rep[:, 1:]).sum(axis=0) / (r[:, 1:] + self.eps).sum(axis=0)
            weights = r.sum(axis=0) / r.sum()

        return {'mu': rate}, weights, log_likelihoods[-1]

    def _generate_component(self, component, n):
        return scipy.stats.poisson.rvs(mu=self.params['mu'][component-1], size=n)

    def _cdf_separate(self, X):
        zero_component = (X[:, 0] <= 0).astype(float)[..., np.newaxis]
        non_zero = scipy.stats.poisson.cdf(X + self.eps, **self.params)
        return np.concatenate((zero_component, non_zero), axis=-1)

    def _sf_separate(self, X):
        zero_component = (X[:, 0] >= 0).astype(float)[..., np.newaxis]
        non_zero = scipy.stats.poisson.sf(X - self.eps, **self.params)
        return np.concatenate((zero_component, non_zero), axis=-1)

