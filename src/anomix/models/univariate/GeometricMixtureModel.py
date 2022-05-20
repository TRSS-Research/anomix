import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps

class GeometricMixtureModel(MixtureModel):
    _param_signature_ = {'p': {'type': float, 'max': 1+eps, 'min': 0}, }
    _support = {'min': 0, 'max': np.inf, 'dtype': int}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'GeometricMixtureModel'
        self.discrete = True
        self.n_variables = 1

    def _fit_deterministic_(self, X: np.ndarray) -> Tuple[Params, np.ndarray]:
        p = 1 / X.mean(axis=0)
        return {'p': p[..., np.newaxis]}, np.array([1])

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components
        p = (1 / X.mean(axis=0) * np.random.uniform(.25, 2, n_components))
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=-1)
        log_likelihoods = [-np.inf]

        for iteration in range(self.max_iterations):
            # calculate the maximum likelihood of each observation xi

            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, p=p)
            r = (likelihood_weighted + eps) / (likelihood_weighted + eps).sum(axis=1)[..., np.newaxis]

            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break
            # Maximization
            p = 1 / np.average(X_rep, weights=r, axis=0)
            weights = r.sum(axis=0) / r.sum()

        return {'p': p}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[
        np.ndarray, Union[float, np.float64]]:
        likelihoods = scipy.stats.geom.pmf(data, **params)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _generate_component(self, component, n):
        return scipy.stats.geom.rvs(p=self.params['p'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.geom.cdf(X+self.eps, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.geom.sf(X-self.eps, **self.params)