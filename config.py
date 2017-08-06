import copy
import os
import itertools
import random
from general_utils import get_logger

import toml


class Config(object):
    """
    This class build configuration object from toml file.
    """
    def __init__(self, filename):
        """
        Args:
            filename: path to toml file
        """
        with open(filename) as conffile:
            config = toml.loads(conffile.read())

        # general config
        self.path_keys = list(config['path'].keys())
        self.output_path = str(config['path']["output_path"])
        self.model_output = str(config['path']["model_output"])
        self.learning_curves_output = str(config['path']["learning_curves_output"])
        self.log_path = str(config['path']["log_path"])

        # embeddings
        self.dim = int(config["data"]["dim"])
        self.dim_char = int(config["data"]["dim_char"])
        self.glove_filename = str(config["data"]["glove_filename"])
        # trimmed embeddings (created from glove_filename with build_data.py)
        self.trimmed_filename = str(config["data"]["trimmed_filename"])

        # dataset
        self.data_type = str(config["data"]["data_type"])
        self.dev_filename = str(config["data"]["dev_filename"])
        self.test_filename = str(config["data"]["test_filename"])
        self.train_filename = str(config["data"]["train_filename"])
        _max_iter = int(config["data"]["max_iter"])
        self.max_iter = _max_iter if _max_iter > 0 else None

        # vocab (created from dataset with build_data.py)
        self.words_filename = str(config["data"]["words_filename"])
        self.tags_filename = str(config["data"]["tags_filename"])
        self.chars_filename = str(config["data"]["chars_filename"])

        # training
        self.search_params_keys = [k for k, v in config["hyperparameters"].items()
                                     if isinstance(v, list)]
        self.search_params_vals = [config["hyperparameters"][k] for k in self.search_params_keys]
        self.num_random_search = config["hyperparameters"]["num_random_search"]
        self.train_embeddings = config["hyperparameters"]["train_embeddings"]
        self.nepochs = config["hyperparameters"]["nepochs"]
        self.dropout = config["hyperparameters"]["dropout"]
        self.batch_size = config["hyperparameters"]["batch_size"]
        self.lr_method = config["hyperparameters"]["lr_method"]
        self.lr = config["hyperparameters"]["lr"]
        self.lr_decay = config["hyperparameters"]["lr_decay"]
        self.clip= config["hyperparameters"]["clip"]
        self.nepoch_no_imprv = config["hyperparameters"]["nepoch_no_imprv"]
        self.reload = config["hyperparameters"]["reload"]

        # model hyperparameters
        self.hidden_size = config["hyperparameters"]["hidden_size"]
        self.char_hidden_size = config["hyperparameters"]["char_hidden_size"]

        # NOTE: if both chars and crf, only 1.6x slower on GPU
        # if crf, training is 1.7x slower on CPU
        self.crf = config["hyperparameters"]["crf"]
        # if char embedding, training is 3.5x slower on CPU
        self.chars = config["hyperparameters"]["chars"]

        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # create instance of logger
        self.logger = get_logger(self.log_path)

    def save(self, filepath):
        result = {
            'path': {},
            'data': {},
            'hyperparameters': {}
        }

        result['path'] = {
            'output_path': self.output_path,
            'model_output': self.model_output,
            'learning_curves_output': self.learning_curves_output,
            'log_path': self.log_path
        }

        result['data'] = {
            'dim': self.dim,
            'dim_char': self.dim_char,
            'glove_filename': self.glove_filename,
            'trimmed_filename': self.trimmed_filename,
            'data_type': self.data_type,
            'dev_filename': self.dev_filename,
            'test_filename': self.test_filename,
            'train_filename': self.train_filename,
            'max_iter': self.max_iter if self.max_iter is not None else -1,
            'words_filename': self.words_filename,
            'tags_filename': self.tags_filename,
            'chars_filename': self.chars_filename,
        }

        result['hyperparameters'] = {
            'num_random_search': self.num_random_search,
            'train_embeddings': self.train_embeddings,
            'nepochs': self.nepochs,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'lr_method': self.lr_method,
            'lr': self.lr,
            'lr_decay': self.lr_decay,
            'clip': self.clip,
            'nepoch_no_imprv': self.nepoch_no_imprv,
            'reload': self.reload,
            'hidden_size': self.hidden_size,
            'char_hidden_size': self.char_hidden_size,
            'crf': self.crf,
            'chars': self.chars,
        }

        with open(filepath, 'w', encoding='utf-8') as fd:
            toml.dump(result, fd)



def random_search(config):
    candidates = list(itertools.product(*config.search_params_vals))
    if len(candidates) <= 0:
        yield config

    random.shuffle(candidates)
    for i, params in zip(range(config.num_random_search), iter(candidates)):
        variant = copy.copy(config)
        for k, v in zip(config.search_params_keys, params):
            assert not isinstance(v, list)
            setattr(variant, k, v)

        for k in config.path_keys:
            path = getattr(variant, k)
            d = os.path.join(os.path.dirname(path), str(i))
            if not os.path.exists(d):
                os.makedirs(d)
            setattr(variant, k, os.path.join(d, os.path.basename(path)))
        yield variant
