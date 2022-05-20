from typing import Dict, Union, List
from numpy import ndarray

COMPONENT_MIN = 30
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
eps = 1e-128
MIN_ITERATIONS = 25

uni_params = Dict[str, Union[List, ndarray, float]]
Params = Union[uni_params, Dict[str, uni_params]]


class DISTRIBUTIONS:
    NORMAL: str = 'NORMAL'
    LOGNORMAL: str = 'LOGNORMAL'
    ZINORMAL: str = 'ZEROINFLATEDNORMAL'
    POISSON: str = 'POISSON'
    ZIPOISSON: str = 'ZEROINFLATEDPOISSON'
    GEOMETRIC: str = 'GEOMETRIC'
    ZETA: str = 'ZETA'
    BINOMIAL: str = 'BINOMIAL'
    EXPONENTIAL: str = "EXPONENTIAL"
    ZIEXPONENTIAL: str = 'ZEROINFLATEDEXPONENTIAL'
    STUDENTST: str = 'STUDENTST'
    CAUCHY: str = 'CAUCHY'
    BETA: str = 'BETA'
    DISCRETE: list = [POISSON, ZIPOISSON,  GEOMETRIC, ZETA]



