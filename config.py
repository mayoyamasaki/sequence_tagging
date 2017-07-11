import configparser
import os
from general_utils import get_logger


class Config():
    """
    This class build configuration object from ini file.
    """
    def __init__(self, filename):
        """
        Args:
            filename: path to ini file
        """
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read(filename)

        # general config
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
        self.dev_filename = str(config["data"]["dev_filename"])
        self.test_filename = str(config["data"]["test_filename"])
        self.train_filename = str(config["data"]["train_filename"])
        self.max_iter = None if str(config["data"]["max_iter"]) == "None"\
                             else int(config["data"]["max_iter"])

        # vocab (created from dataset with build_data.py)
        self.words_filename = str(config["data"]["words_filename"])
        self.tags_filename = str(config["data"]["tags_filename"])
        self.chars_filename = str(config["data"]["chars_filename"])

        # training
        self.train_embeddings = config["hyperparameters"].getboolean("train_embeddings")
        self.nepochs = int(config["hyperparameters"]["nepochs"])
        self.dropout = float(config["hyperparameters"]["dropout"])
        self.batch_size = int(config["hyperparameters"]["batch_size"])
        self.lr = float(config["hyperparameters"]["lr"])
        self.lr_decay = float(config["hyperparameters"]["lr_decay"])
        self.nepoch_no_imprv =int(config["hyperparameters"]["nepoch_no_imprv"])

        # model hyperparameters
        self.hidden_size = int(config["hyperparameters"]["hidden_size"])
        self.char_hidden_size = int(config["hyperparameters"]["char_hidden_size"])

        # NOTE: if both chars and crf, only 1.6x slower on GPU
        # if crf, training is 1.7x slower on CPU
        self.crf = config["hyperparameters"].getboolean("crf")
        # if char embedding, training is 3.5x slower on CPU
        self.chars = config["hyperparameters"].getboolean("chars")

        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # create instance of logger
        self.logger = get_logger(self.log_path)
