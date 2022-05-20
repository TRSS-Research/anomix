from typing import Dict
import numpy as np
from scipy.sparse import random as sparse_random
from .exceptions import ModelNotFit


def jiggle_likelihoods(likelihoods):
    m, n = likelihoods.shape
    return likelihoods + 1e-8 * np.random.random(size=likelihoods.shape) * sparse_random(m=m, n=n, density=.01).A


def transpose_params(params):
    params = params.copy()
    for p in params:
        params[p] = params[p].T
    return params


def sort_params(weights: np.ndarray, params: Dict, zeros=False):
    if params is None:
        return weights, params
    if zeros:
        zeroweight = weights[0]
        weights = weights[1:]

    order = weights.argsort()[::-1]
    params = params.copy()
    for name in params:
        if isinstance(params[name], dict):
            for subname in params[name]:
                params[name][subname] = params[name][subname][order]
        else:
            params[name] = params[name][order]

    if zeros:
        weights = np.hstack((zeroweight, weights[order]))
    else:
        weights = weights[order]
    return weights, params


def simplify_params(params):
    """
        replace numpy arrays with lists, for jsonification handling
    :param params:
    :return:
    """
    params = params.copy()
    for key in params:
        if isinstance(params[key], np.ndarray):
            params[key] = params[key].tolist()
        elif isinstance(params[key], dict):
            for subkey in params[key]:
                if isinstance(params[key][subkey], np.ndarray):
                    params[key][subkey] = params[key][subkey].tolist()
    return params


def checkfit(func):
    def checker(self, *args, **kwargs):
        if self.isfit:
            return func(self, *args, **kwargs)
        else:
            raise ModelNotFit(f'{self.name} Not yet fit. Attempted function: {func.__name__}', self.name)
    return checker

