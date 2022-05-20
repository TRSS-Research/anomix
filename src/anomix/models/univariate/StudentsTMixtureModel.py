import numpy as np
import scipy.stats
from typing import Tuple
from ..base import MixtureModel
from ...utils.static import Params
import warnings
from smm import SMM


class StudentsTMixtureModel(MixtureModel):
    _param_signature_ = {'loc': {'type': float, 'max': np.inf, 'min': -np.inf},
                         'scale': {'type': float, 'max': np.inf, 'min': 0},
                         'df': {'type': float, 'max': np.inf, 'min': 0},}
    _support = {'min': -np.inf, 'max': np.inf, 'dtype': float}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'StudentsTMixtureModel'
        self.n_iter = 1
        self.zero_truncated = True
        self.discrete = False
        self.n_variables = 1

    def likelihood(self, weights, data, log=False, **params):
        likelihoods = scipy.stats.t.pdf(data, **params)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _fit_deterministic_(self, X):
        df, loc, scale = scipy.stats.t.fit(X)
        return {'df': df[...,np.newaxis], 'loc': loc[...,np.newaxis], 'scale': scale[...,np.newaxis]}, np.array([1])

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        if len(X.shape) == 1:
            X = X[..., np.newaxis]
        model = SMM(n_components)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X)
        loc = model.means_.reshape(-1)
        scale = np.sqrt(model.covars_.reshape(-1))
        df = model.degrees_
        weights = model.weights_
        X_rep = np.repeat(X, n_components, axis=1)
        log_likelihood = self.likelihood(weights=weights, data=X_rep, log=True, loc=loc, scale=scale, df=df)
        return {'loc': loc, 'scale': scale, 'df': df}, weights, log_likelihood

    def _generate_component(self, component, n):
        return scipy.stats.t.rvs(df=self.params['df'][component], scale=self.params['scale'][component],
                                 loc=self.params['loc'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.t.cdf(X, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.t.sf(X, **self.params)