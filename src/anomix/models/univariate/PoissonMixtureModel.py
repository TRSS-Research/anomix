import numpy as np
import scipy.stats
from typing import Tuple
from ..base import MixtureModel
from ...utils.static import Params, eps


class PoissonMixtureModel(MixtureModel):
    _param_signature_ = {'mu': {'type': float, 'max': np.inf, 'min': 0},}
    _support = {'min': 0-eps, 'max': np.inf, 'dtype': int}
    pdf_thresholder = 10e-8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'PoissonMixtureModel'
        self.discrete = True
        self.n_variables = 1

    def likelihood(self, weights, data, log=False, **params):
        likelihoods = scipy.stats.poisson.pmf(data, **params)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _fit_deterministic_(self, X):
        rate = X.mean()
        weights = np.array([1])
        return {'mu': rate.reshape(1)}, weights

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components

        unique, counts = np.unique(X, return_counts=True)
        try:
            rates = np.random.choice(unique, n_components, replace=False, p=counts / counts.sum()).astype(np.float32)
        except ValueError as not_enough:
            rates = X.mean() * (np.random.random(n_components) / (n_components) + .5)

        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=1)
        log_likelihoods = [-np.inf]

        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, mu=rates)
            r = (eps + likelihood_weighted.T) / (likelihood_weighted + self.eps).sum(axis=1)  # todo fix

            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break
            # Maximization
            rates = (r * X).sum(axis=1) / r.sum(axis=1)
            weights = r.sum(axis=1) / r.sum()
            if rates.sum() == 0:  # cant both be zero
                rates[np.argmin(weights)] = .01

        return {'mu': rates}, weights, log_likelihoods[-1]


    def _generate_component(self, component, n):
        return scipy.stats.poisson.rvs(mu=self.params['mu'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.poisson.cdf(X + self.eps, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.poisson.sf(X - self.eps, **self.params)