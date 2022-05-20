import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps
from sklearn.mixture import GaussianMixture
import warnings


class ZeroInflatedNormalMixtureModel(MixtureModel):
    """
        This class is the implementation of the Gaussian Mixture Models
        inspired by sci-kit learn implementation.
    """
    _param_signature_ = {'loc': {'type': float, 'max': np.inf, 'min': -np.inf},
                         'scale': {'type': float, 'max': np.inf, 'min': 0}, }
    _support = {'min': -np.inf, 'max': np.inf, 'dtype': float}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_variables = 1
        self.min_components = 2
        self.zero_inflated = True
        self.name = 'ZeroInflatedNormalMixtureModel'

    def _fit_deterministic_(self, X):
        pi = (X == 0).mean()
        loc = X[X != 0].mean()
        scale = X[X != 0].std()
        weights = np.hstack((pi, (1 - pi)))
        return {'loc': loc.reshape(1), 'scale': scale.reshape(1)}, weights

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        pi = (X == 0).mean()
        weights = np.hstack((pi, (1 - pi) * np.ones(n_components - 1) / (n_components - 1)))
        loc = X[X != 0].mean() * (np.random.random(n_components - 1) / (n_components - 1) + .5)
        scale = X[X != 0].std() * (np.random.random(n_components - 1) / (n_components - 1) + .5)

        log_likelihoods = [-np.inf]
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=1)
        for iteration in range(self.max_iterations):
            # calculate the maximum likelihood of each observation xi

            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False,
                                                  loc=loc, scale=scale)
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break
            r = (eps + likelihood_weighted) / (likelihood_weighted + self.eps).sum(axis=1)[..., np.newaxis]

            # maximization
            loc = (r[:, 1:] * X_rep[:, 1:]).sum(axis=0) / (r[:, 1:] + self.eps).sum(axis=0)
            scale = np.sqrt(
                (r[:, 1:] * np.square(X_rep[:, 1:] - loc + self.eps ** 10)).sum(axis=0) / (r[:, 1:] + self.eps).sum(
                    axis=0))
            weights = r.sum(axis=0) / r.sum()

        return {'loc': loc, 'scale': scale}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray,
                   log=False, **params) -> Union[np.ndarray, Union[float, np.float64]]:

        likelihood_normal = scipy.stats.norm.pdf(x=data[:, 1:], **params)
        likelihood_zero = (data[:, 0] == 0).astype(np.int64)[:, np.newaxis]
        likelihood_weighted = np.concatenate((likelihood_zero, likelihood_normal), axis=1) * weights

        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _generate_component(self, component, n):
        return scipy.stats.norm.rvs(loc=self.params['loc'][component-1], scale=self.params['scale'][component-1], size=n)

    def _cdf_separate(self, X):
        non_zero = scipy.stats.norm.cdf(X, **self.params)
        zero_component = (X[:, 0] <= 0).astype(float)[..., np.newaxis]
        return np.concatenate((zero_component, non_zero), axis=-1)

    def _sf_separate(self, X):
        non_zero = scipy.stats.norm.sf(X, **self.params)
        zero_component = (X[:, 0] >= 0).astype(float)[..., np.newaxis]
        return np.concatenate((zero_component, non_zero), axis=1)


class ZeroInflatedNormalMixtureModelSklearn(ZeroInflatedNormalMixtureModel):
    """
        This class is the implementation of the Gaussian Mixture Models
        wrapping the sci-kit learn implementation.
    """

    def __init__(self, max_iterations=2000):
        # todo
        raise NotImplementedError
        super().__init__(max_iterations)
        self.name = 'ZeroInflatedNormalMixtureModelSklearn'
        self.skip_deterministic = True

    def _fit_deterministic(self, X):
        return self._fit_(X, self.min_components)

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        if len(X.shape) == 1:
            X = X[..., np.newaxis]
        pi = (X == 0).mean()
        X_nz = X[X != 0]

        model = GaussianMixture(n_components - 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_nz)
        loc = model.means_.reshape(-1)
        scale = np.sqrt(model.covariances_.reshape(-1))
        weights = np.hstack((pi, model.weights_))
        X_rep = np.repeat(X, n_components, axis=1)
        log_likelihood = self.likelihood(weights=weights, data=X_rep, log=True, loc=loc, scale=scale)
        return {'loc': loc, 'scale': scale}, weights, log_likelihood

    def _cdf_separate(self, X):

        return scipy.stats.norm.cdf(X, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.norm.sf(X, **self.params)