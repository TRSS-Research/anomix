import numpy as np
import json
from typing import Union, Dict
from .models import model_dict
from ..utils.static import DATETIME_FORMAT
import datetime as dt


def load_model_from_file(model: Union[str, Dict]):
    if isinstance(model, str):
        if 'json' == model[-4:].lower():
            with open(model, 'r', encoding='utf8') as fo:
                model = fo.read()
        return load_model_from_config(json.loads(model))
    elif isinstance(model, dict):
        return load_model_from_config(model)
    raise NotImplementedError




def load_model_from_config(config: Dict):
    model = model_dict[config['type']]()
    params = config['params']
    for key in params.copy():
        param = np.array(params[key])
        if len(param.shape) == 0:
            param = param.reshape(-1)
        params[key] = param
    model.params = params
    model.weights = np.array(config['weights'])
    model.n_components = config['n_components']
    model.max_iterations = config['max_iterations']
    model.n_variables = config['n_variables']
    model.sample_size = config['sample_size']
    model.datetime_fit = dt.datetime.strptime(config['datetime_fit'],DATETIME_FORMAT
                                              ) if 'datetime_fit' in config else None
    model.seconds_to_fit = config.get('seconds_to_fit', None)
    model.isfit = True
    model.converged = config.get('converged', True)
    return model


