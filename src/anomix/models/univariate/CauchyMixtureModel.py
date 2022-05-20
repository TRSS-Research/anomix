import numpy as np
import scipy.stats
from typing import Tuple
from ..base import MixtureModel
from ...utils.exceptions import NotEnoughVariation
from ...utils.static import Params, eps


class CauchyMixtureModel(MixtureModel):
    _param_signature_ = {'loc': {'type': float, 'max': np.inf, 'min': -np.inf},
                         'scale':{'type': float, 'max': np.inf, 'min': 0},}
    _support = {'min': -np.inf, 'max': np.inf, 'dtype': float}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'CauchyMixtureModel'
        self.zero_truncated = True
        self.discrete = False
        self.n_variables = 1

    def likelihood(self, weights, data, log=False, **params):
        likelihoods = scipy.stats.cauchy.pdf(data, **params)
        likelihood_weighted = likelihoods * weights
        if log:
            return self.log_likelihood(likelihood_weighted)
        else:
            return likelihood_weighted

    def _fit_deterministic_(self, X):
        loc, scale = scipy.stats.cauchy.fit(X)
        return {'loc': loc[..., np.newaxis], 'scale': scale[..., np.newaxis]}, np.array([1])

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        # https://www.mdpi.com/2073-8994/11/9/1186/pdf
        X = X.copy()
        X.sort()

        weights = np.ones(n_components) / n_components
        locs = np.quantile(X, (np.arange(1, n_components + 1) / (n_components + 1)))
        locs *= ((np.random.random(n_components) + 99.5) / 100) # force some variation in case quantiles are identical
        scales = np.repeat(.5 * scipy.stats.iqr(X), n_components)
        if scales[0] == 0:
            raise NotEnoughVariation(model_type=self.name)

        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=-1)
        log_likelihoods = [-np.inf]
        for iteration in range(self.max_iterations):

            # Expectation
            likelihood_weighted = self.likelihood(weights=weights, data=X_rep, log=False, loc=locs, scale=scales)
            r = (eps + likelihood_weighted) / (likelihood_weighted + eps).sum(axis=1)[..., np.newaxis]

            log_likelihood = self.log_likelihood(likelihood_weighted)
            log_likelihoods.append(log_likelihood)

            if self.check_convergence(log_likelihoods, ignore_assertion=True):
                break

            # Maximization
            component_n = r.sum(axis=0).round().astype(int)
            if component_n.min() <= 5:
                break
            loc_list, scale_list = [],[]
            previous_component_end = 0
            for component_size in component_n:
                X_component = X[previous_component_end:component_size+previous_component_end]
                ixs = np.array((component_size/4, component_size/2, 3*component_size/4)).round().astype(int)
                q1, loc, q3 = X_component[ixs]
                scale_list.append((q3 - q1) / 2)
                loc_list.append(loc)
                previous_component_end += component_size
            locs = np.array(loc_list)
            scales = np.array(scale_list)
            weights = component_n / component_n.sum()
        return {'loc': locs, 'scale': scales}, weights, log_likelihoods[-1]

    def _generate_component(self, component, n):
        return scipy.stats.cauchy.rvs(loc=self.params['loc'][component], scale=self.params['scale'][component], size=n)

    def _cdf_separate(self, X):
        return scipy.stats.cauchy.cdf(X, **self.params)

    def _sf_separate(self, X):
        return scipy.stats.cauchy.sf(X, **self.params)