import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.exceptions import ModelFailedToFit
from ...utils.static import Params, eps


class BetaMixtureModel(MixtureModel):
    _param_signature_ = {'a': {'type': float, 'max': np.inf, 'min': 0},
                         'b': {'type': float, 'max': np.inf, 'min': 0},}
    _support = {'min':0-eps, 'max':1+eps, 'dtype': float}

    def __init__(self, single_component_bimodal=False, **kwargs):
        super().__init__(**kwargs)
        self.name = 'BetaMixtureModel'
        self.zero_truncated = True
        self.n_variables = 1
        self.min_components = 1
        self.single_component_bimodal = single_component_bimodal
        self.eps = 1e-8


    def _fit_deterministic_(self, X: np.ndarray) -> Tuple[Params, np.ndarray]:
        try:
            a, b, loc, scale = scipy.stats.beta.fit(X, floc=0, fscale=1)
            if ((a < 1) & (b < 1)) and (self.single_component_bimodal is False): # is this good or bad?
                raise RuntimeError('a and b cannot both be less than 1')
        except RuntimeError as e:
            raise ModelFailedToFit(self.name)
        return {'a': a[..., np.newaxis], 'b': b[..., np.newaxis]}, np.array([1])

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components
        alphas, betas = 1 + np.random.random(size=n_components), 1 + np.random.random(size=n_components)
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=-1)
        log_likelihoods = [-np.inf]
        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, a=alphas, b=betas)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break

            r = (eps + likelihood_weighted) / (likelihood_weighted + eps).sum(axis=1)[..., np.newaxis]

            # maximization
            gmeans = scipy.stats.gmean(X_rep, weights=r, axis=0)
            gmeans_1minus = scipy.stats.gmean(1 - X_rep, weights=r, axis=0)
            alphas = .5 + gmeans / (2*(1-gmeans - gmeans_1minus))
            betas = .5 + gmeans_1minus / (2 * (1 - gmeans - gmeans_1minus))
            weights = r.mean(axis=0)

        return {'a': alphas, 'b':betas}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[
        np.ndarray, Union[float, np.float64]]:
        likelihoods = scipy.stats.beta.pdf(data, **params, loc=0, scale=1)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _generate_component(self, component, n):
        return scipy.stats.beta.rvs(a=self.params['a'][component], b=self.params['b'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.beta.cdf(X, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.beta.sf(X, **self.params)