import pickle 
import yaml
import pdb
import torch
import numpy as np
from openpoints.utils import EasyConfig
from yaml.constructor import SafeConstructor


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def save_pickle(pickle_file, data):
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(data, pfile)


def easyconfig_constructor(loader, node):
    instance = EasyConfig()
    yield instance  
    state = loader.construct_mapping(node, deep=True)
    instance.update(state)


def yaml2easy_easyconfig_keys(config):
    new_config = {}
    if 'dictitems' in config:
        for key, value in config['dictitems'].items():
            if isinstance(value, dict):
                new_config[key] = yaml2easy_easyconfig_keys(value)
            else:
                new_config[key] = value
    else:
        for key, value in config.items():
            if isinstance(value, dict):
                new_config[key] = yaml2easy_easyconfig_keys(value)
            else:
                new_config[key] = value
    return EasyConfig(new_config)


SafeConstructor.add_constructor(
    'tag:yaml.org,2002:python/object/new:openpoints.utils.config.EasyConfig',
    easyconfig_constructor
)

def load_easyconfig_from_yaml(file_path):
    with open(file_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = yaml2easy_easyconfig_keys(cfg)
    return cfg

