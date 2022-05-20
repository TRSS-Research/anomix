import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps

class LogNormalMixtureModel(MixtureModel):
    """
        This class is the implementation of the Gaussian Mixture Models
        inspired by sci-kit learn implementation.
    """
    _param_signature_ = {'mu': {'type': float, 'max': np.inf, 'min': -np.inf},
                         'sigma': {'type': float, 'max': np.inf, 'min': 0}}
    _support = {'min':0-eps, 'max':np.inf, 'dtype': float}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_variables = 1
        self.name = 'LogNormalMixtureModel'
        self.zero_truncated = True

    def _fit_deterministic_(self, X) -> Tuple[Params, np.ndarray]:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        s, _, scale = scipy.stats.lognorm.fit(X, floc=0)
        sigma = s
        mu = np.log(scale)
        return {'mu': mu.reshape(1), 'sigma': sigma.reshape(1)}, np.array([1])

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components
        log_X = np.log(X)
        mu = log_X[np.random.choice(range(X.shape[0]), n_components)]
        sigma = np.random.random(n_components) * (log_X.var() / n_components)

        log_X_rep = np.repeat(log_X[..., np.newaxis], n_components, axis=1)
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=1)
        log_likelihoods = [-np.inf]
        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, mu=mu, sigma=sigma)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break

            r = likelihood_weighted / (likelihood_weighted + self.eps).sum(axis=1)[..., np.newaxis]
            # maximization

            mu = np.average(log_X_rep, axis=0, weights=r)
            sigma = np.sqrt(np.average(np.square((log_X_rep - mu)), weights=r, axis=0))
            weights = r.sum(axis=0) / r.sum()

        return {'mu': mu, 'sigma': sigma}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, mu, sigma, log=False
                   ) -> Union[np.ndarray, Union[float, np.float64]]:
        likelihoods = scipy.stats.lognorm.pdf(data, scale=np.exp(mu), s=sigma, loc=0)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted


    def _generate_component(self, component, n):
        return scipy.stats.lognorm.rvs(
            scale=np.exp(self.params['mu'][component]), s=self.params['sigma'][component], size=n)
    def _cdf_separate(self, X):
        return scipy.stats.lognorm.cdf(X, scale=np.exp(self.params['mu']), s=self.params['sigma'], loc=0)

    def _sf_separate(self, X):
        return scipy.stats.lognorm.sf(X, scale=np.exp(self.params['mu']), s=self.params['sigma'], loc=0)