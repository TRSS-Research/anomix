import abc
import datetime as dt
import json
import logging
from typing import Optional, Tuple, Union, List, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt
from ..utils.exceptions import (
    SampleSizeTooSmall,
    AllZeros,
    LikelihoodNaNError,
    LikelihoodError,
    LikelihoodDecreasingError,
    ModelFailedToFit,
    UserInputError,
    DataError
)

from ..utils.static import Params, COMPONENT_MIN, MIN_ITERATIONS, DATETIME_FORMAT
from ..utils.utils import simplify_params, sort_params,checkfit

logger = logging.getLogger(__name__)
BASE_PLOT_MAX = 1000

class PresetParamsError(UserInputError):
    pass


class ParamSizeMismatchError(PresetParamsError):
    pass


class ParamTypeError(PresetParamsError):
    pass


class ParamBoundsError(PresetParamsError):
    pass


class IdenticalComponentError(PresetParamsError):
    pass


class MissingKeyError(UserInputError):
    pass


class MixtureModel(object):
    """
        Abstract base class for Mixture Models, to be subclassed with models designed for specific distributions.
         This base class contains generic high level functions which call the subclass specific functions (primarily
         `likelihood` and `_fit_`. `likelihood` reflects the mixture model likelihood for the data, while `_fit_`
         implements the specific distributions flavor of the EM algorithm.
            E-M algorithm overview:  https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
    """
    _param_signature_ = {'None': {'type': type(None), 'max': np.inf, 'min': -np.inf}}
    _support = {'min':-np.inf, 'max':np.inf, 'dtype':float}
    pdf_thresholder = 10e-5
    def __init__(self, **kwargs) -> None:
        """
            Initialize the model, set some basic class variables
        """
        self.eps = 1e-8
        self.tol = 1e-2
        self.n_iter = 5
        self.name = 'MixtureModel'


        # general model train configurations
        self.max_iterations = kwargs.get('max_iterations', 100)
        self.min_components = kwargs.get('min_components', 1)
        self.max_components = kwargs.get('max_components', 3)
        self.unimodal = kwargs.get('unimodal', False)
        self.sort_params = kwargs.get('sort_params', False)


        # default Values to be overwritten in subclasses
        self.skip_deterministic = False
        self.zero_truncated = False
        self.discrete = False
        self.zero_inflated = False

        # model parameters to be edited by train process
        self.params = dict()
        self.n_components = None
        self.weights = None
        self.sample_size = None
        self.n_variables = None
        self.converged = False
        self.isfit = False

        self.datetime_fit = None
        self.seconds_to_fit = 0

    def subset_zeros(self, X):
        """
            remove zeros from the data. Useful when zeros are representative of nullity.
             If multivariate, then remove if the model is 'zero truncated' then we drop any instance we observe a zero
        :param X: data to censor
        :return: censored data
        """
        if isinstance(self.zero_truncated, bool):
            if self.zero_truncated:  # if our model cannot handle zeros we will drop all observations of zero.
                if len(X.shape) == 1:
                    X = X[X != 0]
                else:
                    X = X[(X != 0).all(axis=1)]
        elif isinstance(self.zero_truncated, tuple):
            for i, truncated in enumerate(self.zero_truncated):
                if truncated:
                    X = X[X[:, i] != 0, :]
        return X

    def __repr__(self):
        return self.__str__()

    def fit(self, X: np.ndarray, overwrite=False) -> None:
        """
            Fit an mixture model of arbitrary number of components, and distributions.
        :param X: data to be fit.
        :param overwrite: bool, will not overwrite model if set to False
        :return: None
        """
        if self.isfit and not overwrite:
            raise ModelFailedToFit('Model already fit. If you want to overwrite, set overwrite=True)')

        self.datetime_fit = dt.datetime.now()
        logger.debug(f"Begin Fitting {self.name}")
        X = self.subset_zeros(X)

        if len(X.shape) == 1:  # set the number of variables and sample size for easy future reference.
            self.sample_size, self.n_variables = X.shape[0], 1
            if (X == 0).all():
                raise AllZeros(self.name)
        else:
            self.sample_size, self.n_variables = X.shape[0], X.shape[1]
            if self.n_variables > 1:
                if (X == 0).all(axis=1).any():
                    raise AllZeros(self.name)
            else:
                X = X.reshape(-1)

        self.check_data_supported(X)

        if isinstance(self.discrete, bool):
            if self.discrete:
                X = np.ceil(X).astype(np.int64)
        elif isinstance(self.discrete, tuple):
            for v in range(self.n_variables):
                if self.discrete[v]:
                    X[:, v] = np.ceil(X[:, v]).astype(np.int64)

        max_components = min(self.sample_size // COMPONENT_MIN, self.max_components)
        if max_components < self.min_components:
            # Too few samples to fit a model of any size
            raise SampleSizeTooSmall(self.name)

        elif max_components == self.min_components:
            # Only enough data to fit a unimodal model
            self.n_components = self.min_components

        if (self.n_components is None) and (self.min_components != max_components) and (not self.unimodal):
            # Only choose the number of components if we need to
            logger.debug(f"Finding Optimal Components")
            # choose the optimal number of components
            self.n_components = self.optimize_components(X, max_components)

        # Now fit the actual model
        logger.debug(f"Fitting model with {self.n_components}")
        self.converged = False  # need to reset convergence param
        params, weights, ll = self._fit(X, self.n_components)
        if params is None:
            raise ModelFailedToFit(self.name)
        self.params = params
        self.weights = weights
        self.isfit = True
        
        self.seconds_to_fit = (dt.datetime.now() - self.datetime_fit).seconds + \
                              (dt.datetime.now() - self.datetime_fit).microseconds / 1000000
        logger.debug(f'Completed')

    def preprocess_pred(self, X: Union[int, float, np.ndarray]):
        if not isinstance(X, np.ndarray):
            X = np.array([X])
        self.check_data_supported(X)
        return X

    def check_data_supported(self, X: np.ndarray):
        if len(X.shape) == 1:
            if self.n_variables == 1: pass
            elif self.n_variables != X.shape[0]:
                raise DataError(f'Not sure what to do with data of shape: {X.shape}. Expecting either (1, {self.n_variables}) or ({self.n_variables},)')
        elif X.shape[1] != self.n_variables:
            raise DataError(f'Not sure what to do with data of shape: {X.shape}. Expecting either (1, {self.n_variables}) or ({self.n_variables},)')

        if self.zero_truncated:
            if any(X == 0):
                raise DataError(f'{self.name} is zero truncated, but you provided zeros')
        if (X < self._support['min']).any() or (X > self._support['max']).any():
            raise DataError(f'Outside of Support range. Required: {self._support}, received: {X}')
        if not (X.astype(self._support['dtype']) == X).all():
            raise DataError(f'Cannot cast {X.dtype} to {self._support["dtype"]}')

    @abc.abstractmethod
    def _cdf_separate(self, X):
        raise NotImplementedError

    @abc.abstractmethod
    def _sf_separate(self, X):
        raise NotImplementedError

    def _cdf(self, X):
        return (self._cdf_separate(X) * self.weights) .sum()

    def _sf(self, X):
        return (self._sf_separate(X) * self.weights).sum()

    def cdf_separate(self, X):
        X = np.repeat(X[..., np.newaxis], repeats=self.n_components - self.zero_inflated, axis=-1)
        return self._cdf_separate(X)

    def sf_separate(self, X):
        X = np.repeat(X[..., np.newaxis], repeats=self.n_components - self.zero_inflated, axis=-1)
        return self._sf_separate(X)

    def cdf(self, X):
        X = np.repeat(X[..., np.newaxis], repeats=self.n_components - self.zero_inflated, axis=-1)
        return self._cdf(X)

    def sf(self, X):
        X = np.repeat(X[..., np.newaxis], repeats=self.n_components - self.zero_inflated, axis=-1)
        return self._sf(X)

    def _predict_proba(self, X: np.ndarray, test_lower_tail=True, test_upper_tail=True):
        if test_lower_tail:
            lower_tail = self.cdf_separate(X).reshape((X.shape[0], self.weights.shape[0]))
        else:
            lower_tail = np.ones((X.shape[0], self.weights.shape[0]))

        if test_upper_tail:
            upper_tail = self.sf_separate(X).reshape((X.shape[0], self.weights.shape[0]))
        else:
            upper_tail = np.ones((X.shape[0], self.weights.shape[0]))

        tails = np.stack((lower_tail, upper_tail))
        pvals = np.min(tails, axis=0) * self.weights
        assigned = pvals.argmax(axis=1)
        return pvals[np.ones_like(assigned, dtype=bool), assigned], assigned

    @checkfit
    def predict_proba(self, X, unweighted=False, test_lower_tail=True, test_upper_tail=True):
        X = self.preprocess_pred(X)
        assert test_lower_tail or test_upper_tail, 'Must specify at least one tail to test'
        pval, assigned = self._predict_proba(X, test_lower_tail=test_lower_tail, test_upper_tail=test_upper_tail)
        if unweighted:
            pval /= self.weights[assigned]
        return pval, assigned

    @checkfit
    def predict_anomaly(self, X: [int, float, np.ndarray], threshold: float=.8, unweighted: bool = True,
                        test_lower_tail: bool = True, test_upper_tail: bool = True) -> np.ndarray:
        """
            predict whether input is anomalous in comparison to the fit model, can specify to include the baseline
                component weights as part of the anomaly reporting, or to treat the components as unweighted, full distributions
        :param X: input to evaluate.
        :param threshold:
        :param unweighted:
        :param test_lower_tail: bool, if true tests if value is anomalous in the lower tail
        :param test_upper_tail: bool, if true tests if value is anomalous in the upper tail
        :return:
        """
        if not (test_upper_tail or test_lower_tail):
            raise UserInputError("Cannot predict anomaly without specifying at least one tail")
        X = self.preprocess_pred(X)
        pval, assigned = self.predict_proba(X, unweighted)
        alpha = (1-threshold) / (test_lower_tail + test_upper_tail)
        return pval < alpha

    @checkfit
    def predict(self, X: [int, float, np.ndarray], threshold: float=.8, unweighted: bool = True,
                test_lower_tail: bool = True, test_upper_tail: bool = True) -> np.ndarray:
        return self.predict_anomaly(X=X, threshold=threshold, unweighted=unweighted, test_lower_tail=test_lower_tail,
                                    test_upper_tail=test_upper_tail)

    def check_convergence(self, log_likelihood_list, ignore_assertion=False):
        current, previous = log_likelihood_list[-1], log_likelihood_list[-2]
        if np.isnan(current):
            raise LikelihoodNaNError()
        if len(log_likelihood_list) < MIN_ITERATIONS:
            return False
        if not ignore_assertion:
            if current + self.tol < previous:
                raise LikelihoodDecreasingError(f'{log_likelihood_list[-1]} < {log_likelihood_list[-2]}')
        if current - self.tol < previous:
            self.converged = True
            return True
        else:
            return False

    def get_optimal_loglikelihood(self, X, true_params, true_weights):
        n_components = true_weights.shape[0]
        X_rep = np.repeat(X[..., np.newaxis], n_components, axis=-1)
        return self.likelihood(data=X_rep, weights=true_weights, log=True, **true_params)

    def __str__(self):
        info = {'type': self.name,
                'max_iterations': self.max_iterations,
                'zero_truncated': self.zero_truncated
                }
        if self.isfit:
            params = simplify_params(self.params)
            info.update({'params': params,
                         'weights': self.weights.tolist(),
                         'n_components': int(self.n_components),
                         'n_variables': self.n_variables,
                         'sample_size': self.sample_size,
                         'datetime_fit': dt.datetime.strftime(self.datetime_fit, DATETIME_FORMAT),
                         'seconds_to_fit': self.seconds_to_fit,
                         'converged': self.converged,
                         })

        return json.dumps(info, indent=4, sort_keys=False)

    @checkfit
    def save(self, filepath):
        with open(filepath, 'w', encoding='utf8') as fo:
            fo.write(self.__str__())

    def _fit_deterministic(self, X: np.ndarray) -> Tuple[Union[Params, None], np.ndarray, float]:
        try:
            params, weights = self._fit_deterministic_(X)
        except ModelFailedToFit:
            return None, np.array([1]), -np.inf
        X_rep = np.repeat(X[..., np.newaxis], self.min_components, axis=-1)
        ll = self.likelihood(data=X_rep, weights=weights, log=True, **params)
        self.converged = True
        return params, weights, ll

    @abc.abstractmethod
    def _fit_deterministic_(self, X: np.ndarray) -> Tuple[Params, np.ndarray]:
        raise NotImplementedError

    def _fit_(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        raise NotImplementedError

    @staticmethod
    def check_weights(weights):
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=float)
        if not np.isclose(weights.sum(), 1).all():
            raise ValueError(f'weights must sum to 1. received: {weights} which sums to {weights.sum()}')
        if (weights < 0).any() or (weights > 1).any():
            raise ValueError(f'weights must all be between 0 and 1. received {weights}')
        return weights

    def preset(self, weights: Union[np.ndarray, Iterable] , overwrite=False, **params):
        if self.isfit:
            if overwrite:
                pass
            else:
                raise UserInputError("Model is already fit! Must specify 'overwrite=True' if you want to reset.")
        weights = self.check_weights(weights)
        self.check_params(weights, params)

        self.weights = weights
        self.params = params
        self.isfit = True
        self.n_components = len(weights)
        self.datetime_fit = dt.datetime.now()

    def check_params(self, weights: np.ndarray, params: dict):
        weights_size = weights.size - int(self.zero_inflated)

        for key in self._param_signature_:  # parameter (e.g loc, scale)
            if not key in params:
                raise MissingKeyError(f'Cannot Find {key}. Expecting Signature like: {self._param_signature_}')
            if not np.array(params[key]).size == weights_size:
                raise ParamSizeMismatchError(f'Number of Component mismatch between weights and {key}. Received weights: {weights}, {key}:{params[key]}')
            for i, arg in enumerate(params[key]):  # which component
                _max, _min = self._param_signature_[key]['max'], self._param_signature_[key]['min']
                _typed = self._param_signature_[key]['type'](arg)
                if not _typed == arg:
                    raise ParamTypeError(
                        f'{key} is of wrong type. Value changed when coerced (coerced to {_typed} from {arg}')
                if (arg >= _max) or (arg <= _min):
                    raise ParamBoundsError(
                        f'{key} is outside of bounds. Received {arg}, but bounded between ({_min:.2f}, {_max:.2f})')
        for i in range(1, weights_size):
            component_param_values = np.array([params[key][i] for key in self._param_signature_])
            other_param_values = np.array([[params[key][j] for key in self._param_signature_] for j in range(weights_size) if j != i ])
            if (component_param_values ==other_param_values).all(axis=1).any():
                raise IdenticalComponentError('Found identical component parameters, cannot be a mixture of identical components')

    def _fit(self, X: np.ndarray, n_components: int) -> Tuple[Params, np.ndarray, float]:
        """
            Fit a model multiple times and choose the best option one. This tries to accommodate for dependence on
            initial conditions.
        :param X: data
        :param n_components: number of componets we are fitting this time
        :return: params, weights, minimum log likelihood
        """
        if n_components == self.min_components and not self.skip_deterministic:
            return self._fit_deterministic(X)
        best_params, best_weights = None, None
        max_ll = -np.inf
        for _ in range(self.n_iter):
            # Fit the model several times, choosing the model with the highest log-likelihood
            try:
                params, weights, ll = self._fit_(X, n_components)
            except LikelihoodError as e:
                logger.warning("Likelihood error - " + e.message)
                continue
            if ll > max_ll:
                # if we have a better LL than before, update the 'best'
                best_params, best_weights, max_ll = params, weights, ll
        if self.sort_params:
            best_weights, best_params = sort_params(best_weights, params=best_params, zeros=self.zero_inflated)
        return best_params, best_weights, max_ll

    @abc.abstractmethod
    def likelihood(self, weights: np.ndarray, data: np.ndarray, log=False, **params
                   ) -> Union[np.ndarray, Union[float, np.float64]]:
        """
            likelihood function. this is model and distribution dependent.
        :param weights: generic mixture weights
        :param data: data to fit
        :param log: bool- if true return LogLikelihood
        :param params: kwargs of the distribution to be fit
        :return:
        """
        raise NotImplementedError

    def log_likelihood(self, likelihood_weighted):
        return np.log(likelihood_weighted.sum(axis=1) + self.eps).sum()

    @staticmethod
    def choose_from_loglikelihoods(llikelihoods: np.ndarray, n: int, modifier: float = 2):
        """
            modified BIC (seriously modified)
                This is modified to put more severe penalties on more components
            Chooses the index of the optimal choice criteria. we flipped LL to negative
        :param llikelihoods: array of n_components and associated loglikelihoods
        :param n: number of records
        :param modifier: float modifier to BIC. at 0, no modification.
        :return: index of optimal model
        """
        choice_criteria = - (2 + modifier) * llikelihoods[:, 1] + np.log(n) * (llikelihoods[:, 0] + 1) ** (1 + modifier)
        return np.argmin(choice_criteria)

    def optimize_components(self, X: np.ndarray, max_components) -> int:
        """
            train-test cross-validation to decide the optimal number of components in the model.
                This adds a lot of time to the modeling, we fit the model
                ((max_components-self.min_components) * self.n_iter) times
        :param X: data
        :param max_components: maximum components we are able to fit due to sample sizes
        :return: optimal number of components
        """
        np.random.shuffle(X)
        train = X[:X.shape[0] // 2, ...]
        test = X[X.shape[0] // 2:, ...]
        if (train == 0).all() or (test == 0).all():
            self.n_components = self.min_components
            return self.min_components
        test_llikelihood = []
        for n_components in range(self.min_components, max_components + 1):
            # Find the best model at a given number of components
            params, weights, train_ll = self._fit(train, n_components)
            # potential todo: break if one of the weights is below a certain threshold?
            if params is None:
                test_llikelihood.append((n_components, -np.inf))
                continue
            test_repeat = np.repeat(test[..., np.newaxis], n_components, axis=-1)  # j

            # test this 'best' model on unseen data and get the log-likelihood
            test_ll = self.likelihood(weights=weights, data=test_repeat, log=True, **params)
            test_llikelihood.append((n_components, test_ll))

        test_llikelihood = np.array(test_llikelihood)
        if (test_llikelihood.size == 0) or np.isinf(test_llikelihood[:, 1]).all():  # unable to fit, pass the buck
            logger.warning(f'No Valid Test Likelihoods Passed')
            return self.min_components

        # choose the best number of components based on the log likelihoods
        try:
            optim_n_components = self.choose_from_loglikelihoods(
                test_llikelihood, n=test.shape[0], modifier=2) + self.min_components
        except IndexError:
            logger.warning(f'Choose Component failed: {test_llikelihood}')
            optim_n_components = self.min_components
        self.n_components = optim_n_components
        return optim_n_components

    def generate_zero_inflated_component(self, n):
        component_n = round(self.weights[0] * n)
        return np.zeros(component_n), np.zeros(component_n)

    @checkfit
    def generate(self, n: int, **kwargs):
        """
            generate data given self
        :param n: number of samples
        :return:
        """
        data = []
        labels = []
        if self.zero_inflated:
            zero_data, zero_labels = self.generate_zero_inflated_component(n)
            data.append(zero_data)
            labels.append(zero_labels * 0)
        for i in range(int(self.zero_inflated), self.n_components):
            weight = self.weights[i]
            component_n = round(weight * n)
            data.append(self._generate_component(component=i, n=component_n, **kwargs))
            labels.append(np.ones(component_n) * i)
        data, labels = np.concatenate(data), np.concatenate(labels)
        ix = np.arange(data.shape[0])
        np.random.shuffle(ix)
        return labels[ix], data[ix]

    @abc.abstractmethod
    def _generate_component(self, component: int,  n: int, **kwargs):
        """
            Generate data for a given component index.
            Abstract method implemented by the subclass
            :param component: component index
            :param n: number of samples to generate
        """
        raise NotImplementedError

    def pdf(self, x):
        return self.separate_pdf(x).sum(axis=1)

    def separate_pdf(self, x):
        X = np.repeat(x[..., np.newaxis], repeats=self.n_components, axis=-1)
        return self.likelihood(weights=self.weights, data=X, log=False, **self.params)

    @checkfit
    def get_plotting_range(self):
        """
            identify the minimum and maximum over which we can plot, depending on the 'support' of the distribution
        :return:
        """
        _min = max(self._support['min'], -BASE_PLOT_MAX)
        _max = min(self._support['max'],  BASE_PLOT_MAX)
        X = self.get_plotting_array(_min, _max)
        pdf = self.pdf(X)
        pdf_threshold = pdf.max() * self.pdf_thresholder
        _min = X[np.argmax(pdf > pdf_threshold)]
        _max = X[pdf.shape[0] - np.argmax(pdf[::-1] > pdf_threshold) - 1]
        return _min, _max

    def get_plotting_array(self, _min: float, _max: float) -> np.ndarray:
        """
            get a range of values between _min and _max to plot over
        :return:
        """
        if self.discrete:
            X = np.arange(_min, _max)
        else:
            X = np.linspace(_min, _max, num=10001)
        if self.zero_inflated:
            if not (X == 0).any():
                ix = np.argwhere(X > 0)[0]
                X = np.insert(X, ix, 0)
        return X

    def plot_pdf(self, x_range: Optional[Tuple[float, float]] = None, show=True, smooth_discrete=True,
                 fig_ax: Optional[Tuple] = None):
        """
            Plot the composite likelihood of the mixtures.
            Infers a range if none is provided based on calculated likelihood and support
        :param x_range: minimum and maximum for the x range
        :param show: show figure or not
        :param smooth_discrete: smooth the likelihod for discrete distributions
        :param fig_ax: tuple of figure and axis to plot upon
        :return: (figure, axis)
        """
        if x_range is None:
            x_range = self.get_plotting_range()
        _min, _max = x_range
        X = self.get_plotting_array(_min, _max)
        if fig_ax is None:
            f, ax = plt.subplots(figsize=(12, 8))
        else:
            f, ax = fig_ax

        component_pdfs = self.separate_pdf(X)
        component_list = self.get_components()
        if self.discrete and smooth_discrete:
            from scipy.interpolate import make_interp_spline, BSpline, interp1d

            xnew = np.linspace(X.min(), X.max(), 300)

            component_pdfs = make_interp_spline(X, component_pdfs, k=2)(xnew)  # type: BSpline
            component_pdfs[component_pdfs < 0] = 0
            if self.zero_inflated:
                component_pdfs[np.argwhere(xnew == 0), 0] = self.weights[0]
                component_pdfs[np.argwhere(xnew != 0),0] = 0
            X = xnew
        pdf = component_pdfs.sum(axis=1)

        ax.plot(X, component_pdfs, label=[{key: f'{params[key]:.2f}' for key in params} for params in component_list],
                linewidth=5, alpha=1/(len(component_list) + 1))
        ax.plot(X, pdf, label=f'Composite Likelihood', alpha=1, linewidth=1)
        ax.set_xlabel('X')
        ax.set_ylabel('P(X)')
        ax.legend()
        if show:
            plt.show()
        return f, ax

    @checkfit
    def get_components(self) -> List[Dict[str, Union[bool, np.ndarray, float, int]]]:
        """
            get list of component parameters
        :return:
        """
        component_list = []
        if self.zero_inflated:
            component_list.append({'weight': self.weights[0], 'Zero Component': True})
        for i in range(int(self.zero_inflated), self.n_components):
            component_list.append({'weight': self.weights[i]} | {key: self.params[key][i-int(self.zero_inflated)] for key in self.params})
        return component_list


def initialize_weights_uniform(n_components):
    return np.ones(n_components) / n_components

