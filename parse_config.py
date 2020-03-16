import datetime
import logging
from utils import read_json, write_json
from logger import setup_logging
from pathlib import Path
from functools import partial


class ConfigParser:
    def __init__(self, config_name):
        self.config_name = config_name
        self.config = read_json(self.config_name)

        # set save_dir where trained models and log will be saved
        save_dir = Path(self.config['trainer']['save_dir'])

        experiment_name = self.config['name']
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self.save_dir = save_dir / 'models' / experiment_name / run_id
        self.log_dir = save_dir / 'logs' / experiment_name / run_id

        # make directory for saving checkpoints and log
        exist_ok = False
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = self[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = self[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, name), *args, **module_args)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}'.format(verbosity,
                                                                                      self.log_levels.keys())
        assert verbosity in self.log_levels.keys(), msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(verbosity)
        return logger

