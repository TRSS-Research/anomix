import numpy as np
import scipy.stats
from typing import Tuple, Union, Callable
from ..base import MixtureModel
from ...utils.static import Params, eps

BASE_N = 100


class BinomialMixtureModel(MixtureModel):
    _param_signature_ = {'p': {'type': float, 'max': 1 + eps, 'min': 0},}
    _support = {'min': 0, 'max': np.inf, 'dtype':int}
    def __init__(self, standardize=False, **kwargs):
        super().__init__(**kwargs)
        self.name = 'BinomialMixtureModel'
        self.discrete = True
        self.n_variables = 2
        self.standardize = standardize

    def _fit_deterministic_(self, X: np.ndarray) -> Tuple[Params, np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == 2
        ps = (X[:, 0] / X[:, 1])
        p = ps.mean()[..., np.newaxis]
        weights = np.array([1])
        return {'p': p}, weights

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components
        ps = (X[:, 0].sum() / X[:, 1].sum()) * np.random.uniform(.5, 1.5, size=n_components)
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=-1)
        log_likelihoods = [-np.inf]
        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, p=ps)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break

            r = (eps + likelihood_weighted) / (likelihood_weighted + eps).sum(axis=1)[..., np.newaxis]

            # maximization
            ps = np.average(X_rep[:, 0, :] / X_rep[:, 1, :], axis=0, weights=r)
            weights = r.mean(axis=0)

        return {'p': ps}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[
        np.ndarray, Union[float, np.float64]]:
        likelihoods = scipy.stats.binom.pmf(n=data[:, 1, :], k=data[:, 0, :], **params)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _generate_component(self, component, n: int, k: Union[Callable, int, np.ndarray]):
        if isinstance(k, int):
            k = np.array(k)
        elif callable(k):
            k = k(n)

        if len(k.shape) > 1:
            if k.shape[1] == 1:
                k = k.flatten()
            else:
                raise Exception(f'Dont know how to handle k of shape {k.shape}')
        elif len(k.shape) < 1:
            k = k[np.newaxis]

        if k.shape[0] == self.n_components:
            k = k[component]

        if k.shape == (n,):
            pass
        elif k.shape in [(1,), ()]:
            k = np.repeat(k, repeats=n)
        else:
            raise Exception(f'Dont know how to handle k of shape: {k.shape}')

        return np.stack((scipy.stats.binom.rvs(n=k, p=self.params['p'][component], size=n), k)).T

    def separate_pdf(self, x):
        x = np.stack((x, np.repeat(BASE_N, x.shape[0]))).T
        return super().separate_pdf(x)

    def _cdf_separate(self, X):
        if len(X.shape) == 1:
            X = X[np.newaxis]
        return scipy.stats.binom.cdf(X[:,0] + self.eps, X[:,1], **self.params)

    def _sf_separate(self, X):
        if len(X.shape) == 1:
            X = X[np.newaxis]
        return scipy.stats.binom.sf(X[:,0] - self.eps, X[:,1], **self.params)