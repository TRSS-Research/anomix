import itertools
import logging
import os
from functools import partial

import pytest
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from typing import Optional
import sys

from src.anomix.models.base import MixtureModel
from src.anomix.models.models import *
from src.anomix.utils.exceptions import ModelFailedToFit, DataError, ModelNotFit, UserInputError

strict_prediction_tests = True

logger = logging.getLogger('test-logger')
logger.setLevel(logging.DEBUG)
np.set_printoptions(precision=2)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs('logs/figs', exist_ok=True)
debug_handler = logging.FileHandler('logs/test-logs.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
logger.addHandler(debug_handler)

info_handler = logging.StreamHandler(sys.stdout)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)

# todo (potentially) repo of default values for each distribution?

plt.rcParams["figure.figsize"] = (14,10)


######################
## REUSABLE PARAMS ###
######################

loc_scale = [
    {'weights': np.array((1,)), 'loc': (0,), 'scale': (1,)},
    {'weights': np.array((1,)), 'loc': (45,), 'scale': (10,)},
    {'weights': np.array((.5, .5)), 'loc': (-10, 30), 'scale': (10, 5)},
    {'weights': np.array((.3, .7)), 'loc': (15, 45), 'scale': (10, 5)},
    {'weights': np.array((.5, .4, .1)), 'loc': (45, -10, -50), 'scale': (1, 5, 10)}
]

poisson_params = [
        {'weights': np.array([1]), 'mu': (.1,)},
        {'weights': np.array([1]), 'mu': (1,)},
        {'weights': np.array([1]), 'mu': (20,)},
        {'weights': np.array([.5, .5]), 'mu': (.1, 5)},
        {'weights': np.array([.3, .7]), 'mu': (1, 10)},
        {'weights': np.array((.5, .1, .4)), 'mu': (1, 5, 20)}
]

p_params = [
    {'weights': np.array([1]), 'p': (.1,)},
    {'weights': np.array([1]), 'p': (.5,)},
    {'weights': np.array([.5, .5]), 'p': (.1, .5)},
    {'weights': np.array([.3, .7]), 'p': (.01, .6)},
    {'weights': np.array((.5, .1, .4)), 'p': (.05, .4, .95)}
]


loc_scale_bad_params = [
        {'loc': np.array([0]), 'scale': np.array([-1])},
        {'loc': np.array([0]), 'scale': np.array([0])},
        {'loc': np.array([0, 0]), 'scale': np.array([1, 1])},
        {'loc': np.array([0, 20, 0]), 'scale': np.array([1, 560, 1])},
    ]

loc_scale_generic_params_by_n_components = [
    {'loc': np.array([0]), 'scale': np.array([1])},
    {'loc': np.array([0, 1]), 'scale': np.array([1, 2])},
    {'loc': np.array([0, 1, 2]), 'scale': np.array([1, 2, 3])},
    {'loc': np.array([0, 1, 2, 3]), 'scale': np.array([1, 2, 3, 4])},
]

p_generic_params_by_n_components = [
    { 'p': (.1,) },
    { 'p': (.1, .5)},
    { 'p': (.01, .2, .9)},
    { 'p': (.01, .1, .2, .95)}
]

p_bad_params = [
    { 'p': (0, )},
    { 'p': (1, )},
    { 'p': (.5, .5)},
    { 'p': (1.1, )},
    { 'p': (-.5, )}
]

poisson_generic_params_by_n_components = [
    {'mu': (1, )},
    {'mu': (1,5)},
    {'mu': (1,5,10)},
    {'mu': (1,5,10,25)},
]

poisson_bad_params = [
    {'mu': (0, )},
    {'mu': (-1, )},
    {'mu': (5, 5)},
]

#########################
# Support Functions #####
#########################


k_params = [{'k': scipy.stats.randint(low=20, high=150).rvs}, {'k': scipy.stats.randint(low=50, high=80).rvs}, {'k': 15}]
p_transform = lambda x: x[:, 0] / x[:, 1]
identity = lambda x: x


def update_zero_inflated(params: dict, zero_weight: Optional[float]=None):
    if zero_weight is None:
        zero_weight = np.random.random() * .75
    params = params.copy()
    weights = params['weights']
    weights = np.append(zero_weight, weights)
    weights = weights / weights.sum()
    params['weights'] = weights
    return params


def check_model_params(model, **params):
    _params = params.copy()
    if _params.get('weights') is not None:
        weights = _params.pop('weights')
        assert (model.weights == weights).all(), f'Weights differ: Expected {weights}, received {model.weights}'
    assert model.isfit, f'Model Not Fit!'
    assert model.params == _params, f'Model Params Wrong: Expected {_params}, recieved {model.params}'


def compare_dicts(prediction:dict, expected:dict, strict=True):
    for attribute_to_check in expected:
        if isinstance(expected[attribute_to_check], tuple):
            expected_value, threshold = expected[attribute_to_check]
        else:
            expected_value, threshold = expected[attribute_to_check], 0
        prediction_value = prediction[attribute_to_check]
        try:
            assert np.isclose(prediction_value, expected_value, atol=threshold), \
                f'{attribute_to_check} is incorrect. Expected {expected_value}, received {prediction_value}'
        except AssertionError as e:
            if strict:
                raise e
            else:
                logger.warning(str(e))

######################################
# TESTING CLASSES ####################
######################################


class BaseTester:
    # todo: predict with component (unweighted) level too
    model_class = MixtureModel
    initial_parameters_list = [dict(),]
    prediction_tests = [[tuple(),],]
    generate_kwargs_list = [dict(),]
    init_kwargs = dict()
    init_args = ()
    predict_kwargs = [dict(),]
    generic_params_by_n_components = [dict(),]
    bad_params = [dict(),]

    generic_weights_by_n_components = [
        np.array([1]) ,
        np.ones(2) / 2,
        np.ones(3) / 3,
        np.ones(4) / 4,
    ]

    bad_weights = [
        np.array([1.1]),
        np.array([2, -1]),
        np.array([.1, .5, .5]),
        np.array([.1, -.1, .5, .5]),
    ]

    @staticmethod
    def compare_datasets_function(*args, **kwargs):
        return scipy.stats.ks_2samp(*args, **kwargs)

    @staticmethod
    def transform_data_for_comparison(X):
        return identity(X)

    @staticmethod
    def model_plotting_function(*args, **kwargs):
        return plot_generic(*args, **kwargs)

    def run_tests(self):
        self.test_preset_overwrite()
        self.test_notfit_predict()
        self.test_fail_preset_bad_params()
        self.test_fail_preset_bad_weights()
        self.test_succeed_preset()
        self.test_predict()
        self.test_train()

    def test_train(self, plot=True):
        model_name = self.model_class.__name__
        n_errors = 0
        if not self.initial_parameters_list:
            logger.warning(f'{model_name} has no initial parameters. Not Testing')
        for i, (preset_parameters, generate_kwargs) in enumerate(zip(
                self.initial_parameters_list,
                itertools.cycle(self.generate_kwargs_list))):
            preset_model = self.model_class()
            preset_model.preset(**preset_parameters)
            preset_labels, preset_data = preset_model.generate(10000, **generate_kwargs)

            train_model = self.model_class()
            try:
                train_model.fit(preset_data)
            except ModelFailedToFit:
                logger.warning(f'Hard Failure on: {model_name} with parameters: {preset_parameters}.')
                continue
            test_labels, test_data = train_model.generate(1000, **generate_kwargs)

            test_data = self.transform_data_for_comparison(test_data)
            preset_data = self.transform_data_for_comparison(preset_data)

            print(f'comparing {model_name}, {preset_parameters}')
            estimate, pvalue = self.compare_datasets_function(preset_data, test_data)

            caption = f'P-value of {pvalue:.2E}\nExpected: {preset_parameters} \n     Received: (weights: {train_model.weights}), {train_model.params}'
            if pvalue < .01:
                caption = f'KS-test failed for {model_name}: ' + caption
                logger.warning('     ' + caption)
                n_errors += 1
                name = model_name + f'{i}-fail'
            else:
                caption = 'Success: ' + caption
                name = model_name + f'-{i}-success'
            if plot:
                plot_histogram_subplots(preset_data, test_data, fig_name='Data ' + name, caption=caption,
                                        plot_function=self.model_plotting_function)

                f, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, sharey=True)
                preset_model.plot_pdf(show=False, smooth_discrete=True, fig_ax=(f, ax1))
                ax1.set_title('Preset Model')
                train_model.plot_pdf(show=False, smooth_discrete=True, fig_ax=(f, ax2))
                ax2.set_title('Trained Model')
                savefig(f, 'Likelihoods ' + name)
                plt.close(f)
        logger.info(f'Completed {model_name}, with {n_errors} failures')

    def test_predict(self):
        model_name = self.model_class.__name__

        if not self.initial_parameters_list:
            logger.warning(f'{model_name} has not implemented initial parameters. Not Testing Predict')
            assert 1 == 0
        if not self.prediction_tests:
            logger.warning(f'{model_name} has no prediction tests. Not Testing')
            assert 1 == 0
        for i, (preset_parameters, test_parameter_list) in enumerate(
                zip(self.initial_parameters_list, self.prediction_tests)):

            model = self.model_class()
            model.preset(**preset_parameters)
            logger.debug(f'working on {model_name}: {preset_parameters}')
            for value, predict_kwargs, expected_response in test_parameter_list:
                if isinstance(expected_response, Exception):
                    with pytest.raises(expected_response.__class__):
                        model.predict(X=value, **predict_kwargs)
                else:
                    logger.debug(f'{value}, {predict_kwargs}')
                    prediction = model.predict(X=value, **predict_kwargs)
                    compare_dicts({'anomaly':prediction},
                                  expected_response,
                                  strict=strict_prediction_tests)

    def test_notfit_predict(self):
        model = self.model_class()
        with pytest.raises(ModelNotFit):
            model.predict(1000)

    def test_fail_preset_bad_weights(self):
        if not self.generic_params_by_n_components:
            logger.warning(f'{self.model_class.__name__} has not implemented generic_params_by_n_components. Not Testing')
        for bad_weights in self.bad_weights:
            model = self.model_class()
            n_components = bad_weights.shape[0]
            with pytest.raises(ValueError):
                model.preset(weights=bad_weights,
                             **self.generic_params_by_n_components[n_components - 1])

    def test_fail_preset_bad_params(self):
        if not self.bad_params:
            logger.warning(f'{self.model_class.__name__} has not implemented badParams. Not Testing')
        for bad_params in self.bad_params:
            model = self.model_class()
            n_components = None
            for key in model._param_signature_:
                if n_components is None:
                    n_components = len(bad_params[key])
                else:
                    assert n_components == len(bad_params[key])
            with pytest.raises(UserInputError):
                model.preset(weights=self.generic_weights_by_n_components[n_components],
                             **bad_params)

    def test_succeed_preset(self):
        if not self.initial_parameters_list:
            logger.warning(f'{self.model_class.__name__} has not implemented initial parameters. Not Testing')
        for params in self.initial_parameters_list:
            model = self.model_class()
            model.preset(**params)
            check_model_params(model, **params)

    def test_preset_overwrite(self):
        model = self.model_class()
        model.preset(**self.initial_parameters_list[0])
        with pytest.raises(UserInputError):
            model.preset(**self.initial_parameters_list[1])
        model.preset(**self.initial_parameters_list[1], overwrite=True)
        check_model_params(model, **self.initial_parameters_list[1])



# threshold, unweighted=True, test_lower_tail=True, test_upper_tail=True
class TestNormalMixtureModel(BaseTester):
    model_class = NormalMixtureModel
    generic_params_by_n_components = loc_scale_generic_params_by_n_components.copy()
    bad_params = loc_scale_bad_params.copy()

    initial_parameters_list = loc_scale.copy()

    prediction_tests = [
        # [{'weights': np.array((1,)), 'loc': (0,), 'scale': (1,)},
        [
            (-1, {'threshold': .8, 'unweighted': True}, {'anomaly': False}),
            (-2.1, {'threshold':.98, 'unweighted': True}, {'anomaly': False}),
            (2, {'threshold':.98, 'unweighted': False}, {'anomaly': False}),
            (8.1, {'threshold':.999, 'unweighted': True}, {'anomaly': True}),
            (-10, {'threshold':.9, 'unweighted': True}, {'anomaly': True}),
        ],
#             {'weights': np.array((1,)), 'loc': (45,), 'scale': (10,)},
        [
            (45, {'threshold':.8, 'unweighted': True}, {'anomaly': False}),
            (55, {'threshold':.9, 'unweighted': True}, {'anomaly': False}),
            (-50, {'threshold':.9, 'unweighted': True}, {'anomaly': True}),
            (150, {'threshold':.99, 'unweighted': True}, {'anomaly': True}),
         ],
 #             {'weights': np.array((.5, .5)), 'loc': (-10, 30), 'scale': (10, 5)},
        [
            (-20, {'threshold': .8, 'unweighted': True}, {'anomaly': False}),
            (-20, {'threshold': .8, 'unweighted': False}, {'anomaly': True}),
            (0.1, {'threshold': .8, 'unweighted': True}, {'anomaly': False}),
            (30, {'threshold': .9, 'unweighted': False}, {'anomaly': False}),
            (25, {'threshold': .98, 'unweighted': False}, {'anomaly': False}),
            (-100, {'threshold':.99}, {'anomaly': True}),
            (15, {'threshold':.95}, {'anomaly': True}),
            (150, {'threshold':.99}, {'anomaly': True}),
         ],

#             {'weights': np.array((.3, .7)), 'loc': (15, 45), 'scale': (10, 5)},
        [
            (5, {'threshold':.7}, {'anomaly': False}),
            (25, {'threshold':.8}, {'anomaly': False}),
            (35, {'threshold':.98}, {'anomaly': False}),
            (55, {'threshold':.9}, {'anomaly': True}),
            (-25, {'threshold':.99}, {'anomaly': True}),
            (100, {'threshold':.95}, {'anomaly': True}),
        ],
 #             {'weights': np.array((.5, .4, .1)), 'loc': (45, -10, -50), 'scale': (1, 5, 10)}, ]
        [
            (-50, {'threshold': .8}, {'anomaly': False}),
            (-10, {'threshold': .8}, {'anomaly': False}),
            (45,  {'threshold': .8}, {'anomaly': False}),
            (50,  {'threshold': .8}, {'anomaly': True}),
            (-100,{'threshold': .9}, {'anomaly': True}),
            (10,  {'threshold': .9}, {'anomaly': True}),
            (100, {'threshold': .99}, {'anomaly': True}),
        ],
    ]


class TestNormalMixtureModelSklearn(TestNormalMixtureModel):
    model_class = NormalMixtureModelSklearn


class TestCauchyMixtureModel(BaseTester):
    model_class = CauchyMixtureModel

    initial_parameters_list = loc_scale.copy()

    @staticmethod
    def compare_datasets_function(*args):
        return censored_ks_95th(*args)

    prediction_tests = [
        # [{'weights': np.array((1,)), 'loc': (0,), 'scale': (1,)},
        [(-1, {'threshold': .8}, {'anomaly': False}),
         (-2.1, {'threshold': .9}, {'anomaly': False}),
         (2, {'threshold': .9}, {'anomaly': False}),
         (8.1, {'threshold': .9}, {'anomaly': True}),
         (-10, {'threshold': .9}, {'anomaly': True}),
         (-10.5, {'threshold': .95}, {'anomaly': False}),
         (100, {'threshold': .95}, {'anomaly': True}),],
#             {'weights': np.array((1,)), 'loc': (45,), 'scale': (10,)},
        [(45, {'threshold': .8}, {'anomaly': False}),
         (55, {'threshold': .9}, {'anomaly': False}),
         (-50, {'threshold': .9}, {'anomaly': True}),
         (150, {'threshold': .9}, {'anomaly': True}),
         (1000.5, {'threshold': .95}, {'anomaly': True}),
         (1000, {'threshold': .999}, {'anomaly': False}),],
 #             {'weights': np.array((.5, .5)), 'loc': (-10, 30), 'scale': (10, 5)},
        [(-20, {'threshold': .8}, {'anomaly': False}),
         (-0.1, {'threshold': .8}, {'anomaly': False}),
         (35, {'threshold': .8}, {'anomaly': False}),
         (25.25, {'threshold': .8}, {'anomaly': False}),
         (-100, {'threshold': .9}, {'anomaly': True}),
         (150, {'threshold': .9}, {'anomaly': True}),
         (1000, {'threshold': .95}, {'anomaly': True}),
         (1000, {'threshold': .999}, {'anomaly': False}), ],

#             {'weights': np.array((.3, .7)), 'loc': (15, 45), 'scale': (10, 5)},
        [(5, {'threshold': .8}, {'anomaly': False}),
         (25, {'threshold': .8}, {'anomaly': False}),
         (35, {'threshold': .8}, {'anomaly': False}),
         (55, {'threshold': .8}, {'anomaly': False}),
         (-100, {'threshold': .9}, {'anomaly': True}),
         (150, {'threshold': .95}, {'anomaly': True}),
         (100, {'threshold': .9}, {'anomaly': True}),
         (100, {'threshold': .99}, {'anomaly': False}), ],
 #             {'weights': np.array((.5, .4, .1)), 'loc': (45, -10, -50), 'scale': (1, 5, 10)}, ]
        [(-50, {'threshold': .8}, {'anomaly': False}),
         (-10, {'threshold': .8}, {'anomaly': False}),
         (45, {'threshold': .8}, {'anomaly': False}),
         (50, {'threshold': .8}, {'anomaly': True}),
         (-150, {'threshold': .9}, {'anomaly': True}),
         (10, {'threshold': .9}, {'anomaly': False}),
         (100, {'threshold': .99}, {'anomaly': False}), ],
    ]

    generic_params_by_n_components = loc_scale_generic_params_by_n_components.copy()
    bad_params = loc_scale_bad_params.copy()

class TestBetaMixtureModel(BaseTester):
    model_class = BetaMixtureModel

    initial_parameters_list = [
        {'weights': np.array([1]), 'a': (1,), 'b': (1,)},
        {'weights': np.array([1]), 'a': (4,), 'b': (.5,)},
        {'weights': np.array([1]), 'a': (2,), 'b': (5,)},
        {'weights': np.array((.5, .5)), 'a': (1, 4), 'b': (1, .5)},
        {'weights': np.array((.7, .3)), 'a': (.5, 4), 'b': (3, .5)},
    ]
    generic_params_by_n_components =[
        {'a': (4,), 'b': (.5,)},
        {'a': (.1,5), 'b': (5,.1)},
        {'a': (.1,1,5), 'b': (2,1,1)},
        {'a': (.1, 1, 5,5), 'b': (2, 1, 1, 4)},
    ]
    bad_params = [
        {'a': (0,), 'b': (1,)},
        {'a': (1,), 'b': (0,)},
        {'a': (-1,), 'b': (1,)},
        {'a': (1,), 'b': (-1,)},
    ]
    prediction_tests = [
        [
            (0.5, {'threshold': 0.8}, {'anomaly': False}),
            (0.2, {'threshold': 0.9}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.99, {'threshold': 0.9}, {'anomaly': True}),
            (1.5, {'threshold': 0.9},  DataError('')),
            (-0.5, {'threshold': 0.9}, DataError('')),
        ],
        [
            (0.5, {'threshold': 0.8}, {'anomaly': True}),
            (0.2, {'threshold': 0.9}, {'anomaly': True}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.99, {'threshold': 0.9}, {'anomaly': False}),
            (0.9, {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            (0.3, {'threshold': 0.8}, {'anomaly': False}),
            (0.5, {'threshold': 0.95}, {'anomaly': False}),
            (0.2, {'threshold': 0.9}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.9, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            (0.5, {'threshold': 0.8}, {'anomaly': False}),
            (0.5, {'threshold': 0.95}, {'anomaly': False}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.99, {'threshold': 0.9}, {'anomaly': False}),
            (0.9, {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            (0.5, {'threshold': 0.8}, {'anomaly': True}),
            (0.5, {'threshold': 0.95}, {'anomaly': False}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
            (0.6, {'threshold': 0.8}, {'anomaly': True}),
            (0.999, {'threshold': 0.9}, {'anomaly': False}),
            (0.999999, {'threshold': 0.95}, {'anomaly': True}),
            (0.9999999, {'threshold': 0.98}, {'anomaly': True}),
            (0.9, {'threshold': 0.9}, {'anomaly': False}),
        ],
    ]


class TestBinomialMixtureModel(BaseTester):
    model_class = BinomialMixtureModel

    initial_parameters_list = p_params.copy()
    generate_kwargs_list = k_params.copy()
    generic_params_by_n_components = p_generic_params_by_n_components.copy()
    bad_params = p_bad_params.copy()

    @staticmethod
    def transform_data_for_comparison(X):
        return p_transform(X)

    prediction_tests =  [
        [
            ((1, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((3, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((17, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((1, 5), {'threshold': 0.9}, {'anomaly': False}),
            ((4, 50), {'threshold': 0.9}, {'anomaly': False}),
            ((10, 1000), {'threshold': 0.9}, {'anomaly': True}),
            ((-1, 10), {'threshold': 0.9}, DataError('')),
            ((0.5, 10), {'threshold': 0.9}, DataError('')),
            (5, {'threshold': 0.9}, DataError('')),
        ],
        [
            ((6, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((9, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((1, 5), {'threshold': 0.9}, {'anomaly': False}),
            ((1, 10), {'threshold': 0.9}, {'anomaly': True}),
            ((35, 50), {'threshold': 0.9}, {'anomaly': True}),
            ((100, 1000), {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            ((1, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((10, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((2, 5), {'threshold': 0.9}, {'anomaly': False}),
            ((1, 10), {'threshold': 0.9}, {'anomaly': False}),
            ((35, 50), {'threshold': 0.9}, {'anomaly': True}),
            ((50, 1000), {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            ((0, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((10, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((10, 10), {'threshold': 0.9}, {'anomaly': True}),
            ((1, 50), {'threshold': 0.9}, {'anomaly': False}),
            ((5, 1000), {'threshold': 0.95}, {'anomaly': False}),
        ],
        [
            ((0, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((10, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((30, 200), {'threshold': 0.9}, {'anomaly': True}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((10, 10), {'threshold': 0.9}, {'anomaly': False}),
            ((1, 50), {'threshold': 0.9}, {'anomaly': False}),
            ((40, 1000), {'threshold': 0.95}, {'anomaly': False}),
        ],
    ]



class TestGeometricMixtureModel(BaseTester):
    model_class = GeometricMixtureModel

    generic_params_by_n_components = p_generic_params_by_n_components.copy()
    bad_params = p_bad_params.copy()

    initial_parameters_list = p_params.copy()

    prediction_tests = [
        [
            (1, {'threshold': 0.81}, {'anomaly': False}),
            (15, {'threshold': 0.99}, {'anomaly': False}),
            (50, {'threshold': 0.98}, {'anomaly': True}),
            (0, {'threshold': 0.9}, {'anomaly': True}),
            (100, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (5, {'threshold': 0.8}, {'anomaly': True}),
            (5, {'threshold': 0.9}, {'anomaly': False}),
            (15, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (5, {'threshold': 0.9}, {'anomaly': False}),
            (5, {'threshold': 0.99}, {'anomaly': False}),
            (15, {'threshold': 0.99}, {'anomaly': False}),
            (50, {'threshold': 0.98}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (4, {'threshold': 0.9}, {'anomaly': False}),
            (5, {'threshold': 0.8}, {'anomaly': True}),
            (15, {'threshold': 0.99}, {'anomaly': False}),
            (100, {'threshold': 0.99}, {'anomaly': False}),
            (500, {'threshold': 0.98}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (4, {'threshold': 0.9}, {'anomaly': False}),
            (5, {'threshold': 0.8}, {'anomaly': False}),
            (15, {'threshold': 0.99}, {'anomaly': False}),
            (95, {'threshold': 0.98}, {'anomaly': True}),
        ],

    ]

class TestExponentialMixtureModel(BaseTester):
    model_class = ExponentialMixtureModel

    initial_parameters_list = [
        {'weights': np.array([1]), 'scale': (.01,)},
        {'weights': np.array([1]), 'scale': (.1,)},
        {'weights': np.array([1]), 'scale': (1,)},
        {'weights': np.array([1]), 'scale': (5,)},
        {'weights': np.array((.5, .5)), 'scale': (.01, 5)},
        {'weights': np.array([.3, .7]), 'scale': (1, 15)},
        {'weights': np.array([.2, .7, .1]), 'scale': (1, 15, 50)},
    ]

    generic_params_by_n_components = [
        {'scale': (1,)},
        {'scale': (.1,5)},
        {'scale': (.1,5,10)},
        {'scale': (.1,.5,2,20,)},
    ]

    bad_params = [
        {'scale': (0,)},
        {'scale': (-.1, )},
        {'scale': (1, 5, 1)},
    ]
    prediction_tests = [
        [
            (1, {'threshold': 0.99}, {'anomaly': True}),
            (0.01, {'threshold': 0.8}, {'anomaly': False}),
            (0.1, {'threshold': 0.9}, {'anomaly': True}),
            (1e-05, {'threshold': 0.98}, {'anomaly': True}),
            (0.05, {'threshold': 0.8}, {'anomaly': True}),
            (0.005, {'threshold': 0.8}, {'anomaly': False}),
            (-50, {'threshold': 0.8}, DataError('')),
            (-1, {'threshold': 0.8}, DataError('')),
        ],
        [
            (1, {'threshold': 0.99}, {'anomaly': True}),
            (0.01, {'threshold': 0.9}, {'anomaly': False}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
            (1e-05, {'threshold': 0.98}, {'anomaly': True}),
            (0.05, {'threshold': 0.9}, {'anomaly': False}),
            (0.005, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
            (2, {'threshold': 0.8}, {'anomaly': False}),
            (3, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.9}, {'anomaly': True}),
            (25, {'threshold': 0.98}, {'anomaly': True}),
            (8, {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': False}),
            (0.1, {'threshold': 0.99}, {'anomaly': False}),
            (25, {'threshold': 0.98}, {'anomaly': True}),
            (8, {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.99}, {'anomaly': False}),
            (25, {'threshold': 0.98}, {'anomaly': False}),
            (100, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (15, {'threshold': 0.9}, {'anomaly': False}),
            (50, {'threshold': 0.99}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (200, {'threshold': 0.9}, {'anomaly': True}),
        ],

    ]


class TestLogNormalMixtureModel(BaseTester):
    model_class = LogNormalMixtureModel

    initial_parameters_list = [
         {'weights': np.array([1]), 'mu':(0,), 'sigma': (.01,)},
         {'weights': np.array([1]), 'mu':(5,), 'sigma': (2,)},
         {'weights': np.array([1]), 'mu':(-10,), 'sigma': (2,)},
         {'weights': np.array((.5, .5)), 'mu': (-10,5), 'sigma': (2,2)},
         {'weights': np.array((.5, .5)), 'mu': (-10,10), 'sigma': (2,8)},
         {'weights': np.array((.5, .5)), 'mu': (1, 5), 'sigma': (2, 2)},
         {'weights': np.array([.2, .7, .1]), 'mu': (-2, 2, 8), 'sigma': (2,2,2)},
    ]
    generic_params_by_n_components = [
        {'mu': (1,), 'sigma': (1,)},
        {'mu': (0,1), 'sigma': (1,2)},
        {'mu': (0, 1, 5), 'sigma': (1, 2, 5)},
        {'mu': (0, 1, 5, 5), 'sigma': (1, 2, 5, 1)},
    ]

    bad_params = [
        {'mu':(1,), 'sigma': (0,)},
        {'mu': (1,), 'sigma': (-1,)},
        {'mu': (1,1), 'sigma': (1,1)}
    ]
    prediction_tests = [
        [
            (0.1, {'threshold': 0.8}, {'anomaly': True}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (2, {'threshold': 0.8}, {'anomaly': True}),
        ],
        [
            (0.1, {'threshold': 0.9}, {'anomaly': True}),
            (1, {'threshold': 0.9}, {'anomaly': True}),
            (2, {'threshold': 0.95}, {'anomaly': True}),
            (1000, {'threshold': 0.9}, {'anomaly': False}),
            (2000, {'threshold': 0.9}, {'anomaly': False}),
            (5000, {'threshold': 0.9}, {'anomaly': True}),
            (50000, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (1e-07, {'threshold': 0.9}, {'anomaly': True}),
            (1e-06, {'threshold': 0.9}, {'anomaly': True}),
            (1e-05, {'threshold': 0.9}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.001, {'threshold': 0.8}, {'anomaly': True}),
        ],
        [
            (1e-07, {'threshold': 0.9}, {'anomaly': True}),
            (1e-06, {'threshold': 0.9}, {'anomaly': True}),
            (1e-05, {'threshold': 0.9}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.001, {'threshold': 0.8}, {'anomaly': True}),
            (0.1, {'threshold': 0.9}, {'anomaly': True}),
            (1, {'threshold': 0.9}, {'anomaly': True}),
            (1000, {'threshold': 0.9}, {'anomaly': False}),
            (2000, {'threshold': 0.9}, {'anomaly': False}),
            (5000, {'threshold': 0.9}, {'anomaly': True}),
            (50000, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (1e-06, {'threshold': 0.9}, {'anomaly': True}),
            (0.0001, {'threshold': 0.9}, {'anomaly': False}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
            (1, {'threshold': 0.9}, {'anomaly': False}),
            (2, {'threshold': 0.9}, {'anomaly': False}),
            (1000, {'threshold': 0.9}, {'anomaly': False}),
            (2000, {'threshold': 0.9}, {'anomaly': False}),
            (1000000000, {'threshold': 0.9}, {'anomaly': False}),
            (100000000000, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (0.0001, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.8}, {'anomaly': True}),
            (1, {'threshold': 0.9}, {'anomaly': False}),
            (20, {'threshold': 0.9}, {'anomaly': False}),
            (100, {'threshold': 0.9}, {'anomaly': False}),
            (5000, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (1e-08, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.95}, {'anomaly': False}),
            (1, {'threshold': 0.9}, {'anomaly': False}),
            (20, {'threshold': 0.9}, {'anomaly': False}),
            (100, {'threshold': 0.9}, {'anomaly': False}),
            (100000, {'threshold': 0.9}, {'anomaly': True}),
        ],
    ]


def update_loc_scale_for_location_mixture(loc_scale_params):
    return [
        {**params, **{'scale': params['scale'][:1] * len(params['scale'])}}
        for params in loc_scale_params.copy()
    ]


class TestLocationMixtureModel(BaseTester):
    model_class = NormalLocationMixtureModel
    initial_parameters_list = update_loc_scale_for_location_mixture(loc_scale)
    generic_params_by_n_components = update_loc_scale_for_location_mixture(loc_scale_generic_params_by_n_components)
    # bad_params = update_loc_scale_for_location_mixture(loc_scale_bad_params)
    bad_params = [
        {'loc': np.array([0]), 'scale': np.array([-1])},
        {'loc': np.array([0]), 'scale': np.array([0])},
        {'loc': np.array([2, 2]), 'scale': np.array([1, 1])},
        {'loc': np.array([0, 20, 0]), 'scale': np.array([1, 1, 1])},
    ]

    prediction_tests = [
        [
            (-1, {'threshold': 0.8}, {'anomaly': False}),
            (-2.1, {'threshold': 0.98}, {'anomaly': False}),
            (2, {'threshold': 0.98}, {'anomaly': False}),
            (8.1, {'threshold': 0.999}, {'anomaly': True}),
            (-10, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (45, {'threshold': 0.8}, {'anomaly': False}),
            (55, {'threshold': 0.9}, {'anomaly': False}),
            (-50, {'threshold': 0.9}, {'anomaly': True}),
            (150, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (-20, {'threshold': 0.8}, {'anomaly': False}),
            (0.1, {'threshold': 0.8}, {'anomaly': False}),
            (40, {'threshold': 0.8}, {'anomaly': False}),
            (20, {'threshold': 0.8}, {'anomaly': False}),
            (-100, {'threshold': 0.99}, {'anomaly': True}),
            (100, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (-25, {'threshold': 0.99}, {'anomaly': True}),
            (5, {'threshold': 0.8}, {'anomaly': False}),
            (25, {'threshold': 0.8}, {'anomaly': False}),
            (30, {'threshold': 0.7}, {'anomaly': True}),
            (55, {'threshold': 0.9}, {'anomaly': False}),
            (75, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (-55, {'threshold': 0.9}, {'anomaly': True}),
            (-50, {'threshold': 0.8}, {'anomaly': False}),
            (-25, {'threshold': 0.9}, {'anomaly': True}),
            (-10, {'threshold': 0.8}, {'anomaly': False}),
            (10, {'threshold': 0.9}, {'anomaly': True}),
            (45, {'threshold': 0.8}, {'anomaly': False}),
            (50, {'threshold': 0.8}, {'anomaly': True}),
            ],
        ]

class TestPoissonMixtureModel(BaseTester):
    model_class = PoissonMixtureModel
    initial_parameters_list = poisson_params.copy()
    prediction_tests = [
        [
            (0.5, {'threshold': 0.8}, DataError('')),
            (-1, {'threshold': 0.8}, DataError('')),
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (1, {'threshold': 0.8}, {'anomaly': True}),
            (2, {'threshold': 0.8}, {'anomaly': True}),
        ],
        [
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (2, {'threshold': 0.8}, {'anomaly': False}),
            (5, {'threshold': 0.8}, {'anomaly': True}),
        ],
        [
            (2, {'threshold': 0.9}, {'anomaly': True}),
            (20, {'threshold': 0.8}, {'anomaly': False}),
            (15, {'threshold': 0.95}, {'anomaly': False}),
            (25, {'threshold': 0.95}, {'anomaly': False}),
            (40, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (0, {'threshold': 0.9}, {'anomaly': False}),
            (5, {'threshold': 0.8}, {'anomaly': False}),
            (11, {'threshold': 0.95}, {'anomaly': True}),
        ],
        [
            (0, {'threshold': 0.9}, {'anomaly': False}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (4, {'threshold': 0.9}, {'anomaly': True}),
            (10, {'threshold': 0.95}, {'anomaly': False}),
            (20, {'threshold': 0.95}, {'anomaly': True}),
        ],
        [
            (0, {'threshold': 0.9}, {'anomaly': False}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (4, {'threshold': 0.9}, {'anomaly': False}),
            (11, {'threshold': 0.95}, {'anomaly': True}),
            (22, {'threshold': 0.95}, {'anomaly': False}),
            (40, {'threshold': 0.95}, {'anomaly': True}),
        ],

    ]
    generic_params_by_n_components = poisson_generic_params_by_n_components.copy()
    bad_params = poisson_bad_params.copy()


class TestStudentsTMixtureModel(BaseTester):
    model_class = StudentsTMixtureModel
    initial_parameters_list = [
        {**params, **{'df': df}} for params, df in zip(loc_scale.copy(), [(2,), (5,), (20, 20), (10, 3), (5, 5, 5)])
    ]

    @staticmethod
    def compare_datasets_function(*args):
        return censored_ks_95th(*args)

    generic_params_by_n_components =  [
        {**params, **{'df': df}} for params, df in zip(loc_scale_generic_params_by_n_components.copy(),
                                                       [(10,), (5,5), (20, 20, 20), (10, 10,10,10)])
    ]
    bad_params = [
        {'loc': np.array([0]), 'scale': np.array([-1]), 'df':np.array([1])},
        {'loc': np.array([0]), 'scale': np.array([0]), 'df':np.array([1])},
        {'loc': np.array([0, 0]), 'scale': np.array([1, 1]), 'df':np.array([10, 10])},
        {'loc': np.array([0, 20, 0]), 'scale': np.array([1, 560, 1]), 'df':np.array([10, 10, 10])},
        {'loc': np.array([0]), 'scale': np.array([1]), 'df': np.array([0])},
        {'loc': np.array([0]), 'scale': np.array([1]), 'df': np.array([-1])},
    ]


    prediction_tests = [
        [
            (-1, {'threshold': 0.8}, {'anomaly': False}),
            (-2.1, {'threshold': 0.9}, {'anomaly': False}),
            (2, {'threshold': 0.9}, {'anomaly': False}),
            (-10.5, {'threshold': 0.95}, {'anomaly': True}),
            (100, {'threshold': 0.95}, {'anomaly': True}),
        ],
        [
            (45, {'threshold': 0.8}, {'anomaly': False}),
            (55, {'threshold': 0.9}, {'anomaly': False}),
            (-50, {'threshold': 0.9}, {'anomaly': True}),
            (150, {'threshold': 0.9}, {'anomaly': True}),
            (1000.5, {'threshold': 0.95}, {'anomaly': True}),
            (1000, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (-20, {'threshold': 0.8}, {'anomaly': False}),
            (-0.1, {'threshold': 0.8}, {'anomaly': False}),
            (35, {'threshold': 0.8}, {'anomaly': False}),
            (25.25, {'threshold': 0.8}, {'anomaly': False}),
            (-100, {'threshold': 0.9}, {'anomaly': True}),
            (150, {'threshold': 0.9}, {'anomaly': True}),
            (1000, {'threshold': 0.95}, {'anomaly': True}),
            (1000, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (5, {'threshold': 0.8}, {'anomaly': False}),
            (25, {'threshold': 0.8}, {'anomaly': False}),
            (35, {'threshold': 0.8}, {'anomaly': True}),
            (55, {'threshold': 0.8}, {'anomaly': True}),
            (-100, {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            (-50, {'threshold': 0.8}, {'anomaly': False}),
            (-10, {'threshold': 0.8}, {'anomaly': False}),
            (45, {'threshold': 0.8}, {'anomaly': False}),
            (50, {'threshold': 0.8}, {'anomaly': True}),
            (-150, {'threshold': 0.9}, {'anomaly': True}),
            (200, {'threshold': 0.99}, {'anomaly': True}),
            (10, {'threshold': 0.9}, {'anomaly': True}),
        ],

    ]


class TestZetaMixtureModel(BaseTester):
    model_class = ZetaMixtureModel
    initial_parameters_list = [
        {'weights': np.array([1]), 'a': (1.5,)},
        {'weights': np.array([1]), 'a': (1.9,)},
        {'weights': np.array([1]), 'a': (5,)},
        {'weights': np.array([.5, .5]), 'a': (1.5, 5)},
        {'weights': np.array([.3, .7]), 'a': (2, 10)},
    ]

    @staticmethod
    def compare_datasets_function(*args):
        return censored_ks_95th(*args)

    generic_params_by_n_components =  [
        {'a': (2,),},
        {'a': (2,4)},
        {'a': (1.5, 2, 4, )},
        {'a': (1.1, 2, 4, 7)},
    ]

    bad_params =  [
        {'a': (1,),},
        {'a': (.5,)},
        {'a': (-1, )},
    ]
    prediction_tests = [
        [
            (-10, {'threshold': 0.8}, DataError('')),
            (10000, {'threshold': 0.95}, {'anomaly': True}),
            (1, {'threshold': 0.95}, {'anomaly': False}),
            (500, {'threshold': 0.95}, {'anomaly': False}),
        ],
        [
            (50, {'threshold': 0.95}, {'anomaly': True}),
            (1, {'threshold': 0.95}, {'anomaly': False}),
            (3, {'threshold': 0.95}, {'anomaly': False}),
        ],
        [
            (1, {'threshold': 0.95}, {'anomaly': False}),
            (3, {'threshold': 0.95}, {'anomaly': True}),
        ],
        [
            (10000, {'threshold': 0.95}, {'anomaly': True}),
            (1, {'threshold': 0.95}, {'anomaly': False}),
        ],
        [
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (15, {'threshold': 0.9}, {'anomaly': True}),
            (5, {'threshold': 0.9, 'unweighted': True}, {'anomaly': False}),
            (5, {'threshold': 0.9, 'unweighted': False}, {'anomaly': True}),
        ],
    ]


class TestZeroInflatedBinomialMixtureModel(BaseTester):
    model_class = ZeroInflatedBinomialMixtureModel
    initial_parameters_list = [
        {'weights': np.array([.5, .5]), 'p': (.9,)},
         {'weights': np.array((.5, .5)), 'p': (.3,)},
         {'weights': np.array((.5, .3, .2)), 'p': (.6, .9)},
         {'weights': np.array((.2, .4, .4)), 'p': (.1, .9)},
         {'weights': np.array((.1, .5, .4)), 'p': (.1, .5)},
    ]
    bad_params = p_bad_params.copy()
    generic_params_by_n_components = p_generic_params_by_n_components.copy()
    generate_kwargs_list = k_params.copy()

    @staticmethod
    def transform_data_for_comparison(X):
        return p_transform(X)

    prediction_tests = [
        [
            ((0, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((1, 10), {'threshold': 0.8}, {'anomaly': True}),
            ((3, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((4, 5), {'threshold': 0.9}, {'anomaly': False}),
            ((45, 50), {'threshold': 0.9}, {'anomaly': False}),
            ((10, 10), {'threshold': 0.9}, {'anomaly': False}),
            ((17, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((100, 100), {'threshold': 0.9}, {'anomaly': True}),
            ((-1, 10), {'threshold': 0.9}, DataError('')),
            ((0.5, 10), {'threshold': 0.9},DataError('')),
            ((5,), {'threshold': 0.9}, DataError('')),
        ],
        [
            ((0, 10), {'threshold': 0.9}, {'anomaly': False}),
            ((1, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((1, 5), {'threshold': 0.95}, {'anomaly': False}),
            ((3, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((15, 50), {'threshold': 0.9}, {'anomaly': False}),
            ((7, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': True}),
            ((100, 1000), {'threshold': 0.9}, {'anomaly': True}),
        ],
        [
            ((1, 10), {'threshold': 0.8}, {'anomaly': True}),
            ((5, 20), {'threshold': 0.8}, {'anomaly': True}),
            ((12, 20), {'threshold': 0.9}, {'anomaly': False}),
            ((4, 5), {'threshold': 0.9}, {'anomaly': False}),
            ((19, 20), {'threshold': 0.95}, {'anomaly': False}),
        ],
        [
            ((0, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((1, 500), {'threshold': 0.9}, {'anomaly': True}),
            ((1, 10), {'threshold': 0.9}, {'anomaly': False}),
            ((25, 50), {'threshold': 0.9}, {'anomaly': True}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            ((0, 10), {'threshold': 0.8}, {'anomaly': False}),
            ((1, 500), {'threshold': 0.9}, {'anomaly': True}),
            ((1, 10), {'threshold': 0.9}, {'anomaly': False}),
            ((35, 100), {'threshold': 0.9}, {'anomaly': True}),
            ((25, 50), {'threshold': 0.9}, {'anomaly': False}),
            ((19, 20), {'threshold': 0.9}, {'anomaly': True}),
        ],
    ]


class TestZeroInflatedExponentialMixtureModel(BaseTester):
    model_class = ZeroInflatedExponentialMixtureModel
    initial_parameters_list = [
        {'weights': np.array([.5, .5]), 'scale': (.01,)},
        {'weights': np.array((.9, .1)), 'scale': (.05,)},
        {'weights': np.array((.9, .1)), 'scale': (1,)},
        {'weights': np.array((.4, .1, .5)), 'scale': (.1, 5)},
        {'weights': np.array((.33, .33, .34)), 'scale': (.1, 5)}
    ]
    generic_params_by_n_components = TestExponentialMixtureModel.generic_params_by_n_components.copy()
    bad_params = TestExponentialMixtureModel.bad_params.copy()
    prediction_tests = [
        [
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (1, {'threshold': 0.99}, {'anomaly': True}),
            (0.01, {'threshold': 0.8}, {'anomaly': False}),
            (0.1, {'threshold': 0.9}, {'anomaly': True}),
            (1e-05, {'threshold': 0.98}, {'anomaly': True}),
            (0.05, {'threshold': 0.8}, {'anomaly': True}),
            (0.005, {'threshold': 0.8}, {'anomaly': False}),
            (-50, {'threshold': 0.8}, DataError('')),
            (-1, {'threshold': 0.8}, DataError('')),
        ],
        [
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (1, {'threshold': 0.99}, {'anomaly': True}),
            (0.05, {'threshold': 0.9}, {'anomaly': False}),
            (0.001, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
        ],
        [
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (0.01, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.9}, {'anomaly': False}),
            (3, {'threshold': 0.95}, {'anomaly': False}),
            (3, {'threshold': 0.95, 'unweighted': False}, {'anomaly': True}),
            (1, {'threshold': 0.95, 'unweighted': False}, {'anomaly': False}),
            (8, {'threshold': 0.99}, {'anomaly': True}),
        ],
        [
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (0.001, {'threshold': 0.8}, {'anomaly': True}),
            (0.1, {'threshold': 0.8}, {'anomaly': False}),
            (0.1, {'threshold': 0.8, 'unweighted': False}, {'anomaly': True}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (5, {'threshold': 0.9}, {'anomaly': False}),
            (25, {'threshold': 0.98}, {'anomaly': True}),
        ],
        [
            (0, {'threshold': 0.8}, {'anomaly': False}),
            (0.001, {'threshold': 0.9}, {'anomaly': True}),
            (0.1, {'threshold': 0.8}, {'anomaly': False}),
            (1, {'threshold': 0.8}, {'anomaly': False}),
            (5, {'threshold': 0.9}, {'anomaly': False}),
            (25, {'threshold': 0.98}, {'anomaly': True}),
        ],
    ]


class TestZeroInflatedNormalMixtureModel(BaseTester):
    model_class = ZeroInflatedNormalMixtureModel
    initial_parameters_list = [update_zero_inflated(params) for params in loc_scale.copy()]
    bad_params = loc_scale_bad_params.copy()
    generic_params_by_n_components = loc_scale_generic_params_by_n_components.copy()
    prediction_tests = TestNormalMixtureModel.prediction_tests.copy()


class TestZeroInflatedPoissonMixtureModel(BaseTester):
    model_class = ZeroInflatedPoissonMixtureModel
    initial_parameters_list = [update_zero_inflated(params) for params in poisson_params.copy()]
    prediction_tests = TestPoissonMixtureModel.prediction_tests.copy()
    generic_params_by_n_components = poisson_generic_params_by_n_components.copy()
    bad_params = poisson_bad_params.copy()

###########################
# PLOTTING ################
###########################

def plot_generic(axis, data, label):
    axis.hist(data, density=True, bins=30, label=label)


def plot_symmetric_logscale(axis, data, label):
    nbins = 100
    pct_bz = (data < 0).mean()
    pct_az = 1 - pct_bz
    try:
        neg_min = np.log10(-(data[data < 0]).max())
    except ValueError:
        neg_min = -.5
    try:
        pos_min = np.log10((data[data > 0]).min())
    except ValueError:
        pos_min = -.5
    neg_bins = -np.logspace(neg_min, np.log10(abs(data.min())), int(pct_bz * nbins))[::-1]
    pos_bins = np.logspace(pos_min, np.log10(data.max()), int(pct_az * nbins))
    bins = np.hstack((neg_bins, 0, pos_bins))
    axis.hist(data, bins=bins, density=True, label=label)
    axis.set_xscale('symlog')



def plot_histogram_subplots(preset_data, generated_data, fig_name: str, caption='', plot_function=plot_generic):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, gridspec_kw={'bottom': .2}, figsize=(12, 8))
    plot_function(axis=axes[0], data=preset_data, label='Preset Data')
    axes[0].set_title('Target Data')
    plot_function(axis=axes[1], data=generated_data, label='Generated Data')
    axes[1].set_title('Data from Estimated Distribution')
    for ax in axes.flatten():
        ax.set_xlabel('X')
        ax.set_ylabel('P(x)')
    fig.text(0.05, 0.05, caption)
    savefig(fig, fig_name)
    plt.close(fig)
    return fig


def savefig(fig, fig_name):
    fig.savefig(f'logs/figs/{fig_name}.png')


###############################
# Comparison functions ########
###############################

def censored_ks(data1, data2, percentile=.05, **kwargs):
    """
        thinking that big tails induce False Positive KS test
    :param data1: first
    :param data2:
    :param percentile:
    :return:
    """
    _min, _max = np.quantile(data1, [percentile, 1-percentile])
    censored_data1 = data1[(data1 >= _min) & (data1 <= _max)]
    censored_data2 = data2[(data2 >= _min) & (data2 <= _max)]
    return scipy.stats.ks_2samp(censored_data1, censored_data2, **kwargs)

censored_ks_90th = partial(censored_ks, percentile=.1)
censored_ks_95th = partial(censored_ks, percentile=.05)
censored_ks_99th = partial(censored_ks, percentile=.01)

def chi2_test(data1, data2):
    quantiles = np.quantile(data1, [.1, .25, .5, .75, .9])
    cum_counts_1 = (data1[..., np.newaxis] <= quantiles).sum(axis=0)
    cum_counts_2 = (data2[..., np.newaxis] <= quantiles).sum(axis=0)
    quantile_counts_1 = cum_counts_1 - np.hstack((0, cum_counts_1[:-1]))
    quantile_counts_2 = cum_counts_2 - np.hstack((0, cum_counts_2[:-1]))
    quantile_counts_1 = np.append(quantile_counts_1, data1.shape[0] - quantile_counts_1.sum())
    quantile_counts_2 = np.append(quantile_counts_2, data2.shape[0] - quantile_counts_2.sum())
    return scipy.stats.chisquare(quantile_counts_1, quantile_counts_2)






# todo turn different sorts of ANOMALies?
#   relative likelihood? + CDF? otherwise log normal is like never anomalous

# todo other method to compare estimated models?
#   like compare against random or generic models?
#   or a unimodal model?


if __name__ == '__main__':
    TestZeroInflatedBinomialMixtureModel().run_tests()

