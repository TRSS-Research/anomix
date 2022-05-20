import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps

BASE_N = 100

class ZeroInflatedBinomialMixtureModel(MixtureModel):
    _param_signature_ = {'p': {'type': float, 'max': 1 + eps, 'min': 0},}
    _support = {'min': 0 - eps, 'max': np.inf, 'dtype': int}
    def __init__(self, med_abs_dev=False, standardize=False, **kwargs):
        super().__init__(**kwargs)
        self.name = 'ZeroInflatedBinomialMixtureModel'
        self.skip_deterministic = True
        self.min_components = 2
        self.standardize = standardize
        self.zero_inflated = True
        self.discrete = True
        self.n_variables = 2

    def _fit_deterministic(self, X: np.ndarray):
        return self._fit_(X, self.min_components)

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        assert n_components > 1
        weights = np.ones(n_components) / n_components
        ps = (X[:, 0].sum() / X[:, 1].sum()) * np.random.uniform(.5, 1.5, size=n_components - 1)
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
            ps = np.average(X_rep[:, 0, 1:] / X_rep[:, 1, 1:], axis=0, weights=r[:, 1:])
            weights = r.mean(axis=0)

        return {'p': ps}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[
        np.ndarray, Union[float, np.float64]]:
        assert weights.shape[0] > 1
        likelihood_zero = (data[:, 0, 0] == 0).astype(np.int64)[:, np.newaxis]
        likelihood_binom = scipy.stats.binom.pmf(n=data[:, 1, 1:], k=data[:, 0, 1:], **params)
        likelihood_weighted = np.concatenate((likelihood_zero, likelihood_binom), axis=1) * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def generate_zero_inflated_component(self, n):
        zero_data, zero_labels = super().generate_zero_inflated_component(n)
        zero_data = self.add_n_to_k(zero_data)
        return zero_data, zero_labels

    def _generate_component(self, component, n, k):
        component -=1

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

        if k.shape[0] == self.n_components - 1:
            k = k[component]

        if k.shape == (n,):
            pass
        elif k.shape in [(1,), ()]:
            k = np.repeat(k, repeats=n)
        else:
            raise Exception(f'Dont know how to handle k of shape: {k.shape}')

        return np.stack((scipy.stats.binom.rvs(n=k, p=self.params['p'][component], size=n), k)).T

    def add_n_to_k(self, x, n=BASE_N):
        return np.stack((x, np.repeat(n, x.shape[0]))).T

    def separate_pdf(self, x):
        x = self.add_n_to_k(x)
        return super().separate_pdf(x)

    def _cdf_separate(self, X):
        """ should output  array of shape: (self.n_components, n_samples"""
        zero_component = (X[:, 0, 0] <= 0).astype(float)[..., np.newaxis]
        if len(X.shape) == 1:
            X = X[np.newaxis]
        non_zero = scipy.stats.binom.cdf(X[:, 0] + self.eps, X[:,1], **self.params)
        return np.concatenate((zero_component, non_zero), axis=-1)

    def _sf_separate(self, X):
        zero_component = (X[:, 0, 0] >= 0).astype(float)[..., np.newaxis]
        if len(X.shape) == 1:
            X = X[np.newaxis]
        non_zero = scipy.stats.binom.sf(X[:, 0] - self.eps, X[:,1], **self.params)
        return np.concatenate((zero_component, non_zero), axis=-1)
