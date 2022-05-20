import numpy as np
import scipy.stats
from typing import Tuple, Union
from ..base import MixtureModel
from ...utils.static import Params, eps
from sklearn.mixture import GaussianMixture
import warnings


class NormalMixtureModel(MixtureModel):
    """
        This class is the implementation of the Gaussian Mixture Models
        inspired by sci-kit learn implementation.
    """
    _param_signature_ = {'loc': {'type': float, 'max': np.inf, 'min': -np.inf},
                         'scale': {'type': float, 'max': np.inf, 'min': 0}, }
    _support = {'min': -np.inf, 'max': np.inf, 'dtype': float}

    def __init__(self, tied_variance=False, zero_truncated=False, **kwargs):
        super().__init__(zero_truncated=zero_truncated, **kwargs)
        self.n_variables = 1
        self.name = 'NormalMixtureModel'
        self.tied_variance = tied_variance

    def _fit_deterministic_(self, X):
        loc = X.mean()
        scale = X.std()
        weights = np.array([1])
        return {'loc': loc.reshape(1), 'scale': scale.reshape(1)}, weights

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        weights = np.ones(n_components) / n_components
        means = X[np.random.choice(range(X.shape[0]), n_components)]
        variances = np.random.random(n_components) * (X.var() / n_components)

        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=1)
        log_likelihoods = [-np.inf]
        for iteration in range(self.max_iterations):
            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, loc=means,
                                                  scale=np.sqrt(variances))
            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods):
                break

            r = (eps + likelihood_weighted.T) / (likelihood_weighted + eps).sum(axis=1)

            # maximization
            means = (r * X).sum(axis=1) / (r + self.eps).sum(axis=1)
            variances = (r * np.square(X_rep - means + self.eps ** 10).T).sum(axis=1) / (r + self.eps).sum(axis=1)
            if self.tied_variance:
                variances = np.repeat(variances.mean(), n_components)
            weights = r.mean(axis=1)

        return {'loc': means, 'scale': np.sqrt(variances)}, weights, log_likelihoods[-1]

    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params
                   ) -> Union[np.ndarray, Union[float, np.float64]]:
        likelihoods = scipy.stats.norm.pdf(data, **params)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _cdf_separate(self, X):
        return scipy.stats.norm.cdf(X, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.norm.sf(X, **self.params)

    def _generate_component(self, component, n):
        return scipy.stats.norm.rvs(loc=self.params['loc'][component], scale=self.params['scale'][component], size=n)


class NormalLocationMixtureModel(NormalMixtureModel):
    """
        This class is the implementation of the Gaussian Mixture Models
        inspired by sci-kit learn implementation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, tied_variance=True)
        self.name = 'NormalLocationMixtureModel'


class NormalMixtureModelSklearn(NormalMixtureModel):
    """
        This class is the implementation of the Gaussian Mixture Models
        wrapping the sci-kit learn implementation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'NormalMixtureModelSklearn'
        self.skip_deterministic = True
        self.n_iter = 1

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        if len(X.shape) == 1:
            X = X[..., np.newaxis]
        model = GaussianMixture(n_components)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X)
        loc = model.means_.reshape(-1)
        scale = np.sqrt(model.covariances_.reshape(-1))
        weights = model.weights_
        X_rep = np.repeat(X, n_components, axis=1)
        log_likelihood = self.likelihood(weights=weights, data=X_rep, log=True, loc=loc, scale=scale)
        return {'loc': loc, 'scale': scale}, weights, log_likelihood
