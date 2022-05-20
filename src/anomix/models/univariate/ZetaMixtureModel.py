import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps
from scipy.special import zeta


def zeta_prime(x: np.ndarray) -> np.ndarray:
    h = 1e-6
    return (zeta(x + h, 1) - zeta(x - h, 1)) / (2 * h)


def fit_zeta_mle(d: np.ndarray) -> np.ndarray:
    x = np.linspace(1.01, 100, num=10000)
    critical = - zeta_prime(x) / zeta(x)
    return x[np.argmin(np.abs((critical - (1 / d.shape[0]) * np.log(d).sum())))]


class ZetaMixtureModel(MixtureModel):
    _param_signature_ = {'a': {'type': float, 'max': np.inf, 'min': 1},}
    _support = {'min': 1-eps, 'max': np.inf, 'dtype': int}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'ZetaMixtureModel'
        self.min_components = 1
        self.discrete = True
        self.n_variables = 1

    def _fit_deterministic_(self, X: np.ndarray):
        a = fit_zeta_mle(X)
        return {'a': a[..., np.newaxis]}, np.array([1])

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components
        a = np.array((1.01 * np.ones(n_components),
                      np.random.uniform(.5, 1.5, size=n_components) * fit_zeta_mle(X))).max(axis=0)
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=-1)
        log_likelihoods = [-np.inf]
        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, a=a)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods, ignore_assertion=True):
                break

            r = (eps + likelihood_weighted) / (likelihood_weighted + eps).sum(axis=1)[..., np.newaxis]

            a = list()
            chosen = list()
            data_available = np.ones(X.shape[0])
            for i in range(n_components - 1):
                p = r[:, i]
                mask = np.random.binomial(n=1, p=p) # randomly choose pieces according to weight
                actual_mask = (data_available * mask).astype(bool)
                data_chosen = X[actual_mask]
                chosen.append(data_chosen)
                data_available *= (1 - mask)
                a.append(fit_zeta_mle(data_chosen))
            chosen.append(X[data_available.astype(bool)])
            a.append(fit_zeta_mle(X[data_available.astype(bool)]))
            weights = r.mean(axis=0)

        return {'a': a}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params) -> Union[np.ndarray, Union[float, np.float64]]:
        likelihood = scipy.stats.zipf.pmf(data, **params)
        likelihood_weighted = likelihood * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _generate_component(self, component, n):
        return scipy.stats.zipf.rvs(a=self.params['a'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.zipf.cdf(X + self.eps, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.zipf.sf(X - self.eps, **self.params)

