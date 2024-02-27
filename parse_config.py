import logging
from pathlib import Path
from functools import  partial
from datetime import datetime
from utils import read_json, write_json, setup_logging


class ConfigParser:
    def __init__(self, config, resume=None, run_id=None):
        """        
        A class parsing configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.

        Args:
            config (dict): Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
            resume (str, optional): String, path to the checkpoint being loaded.
            run_id (str, optional): Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """

        self.config = config
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        out_dir = Path(self.config['trainer']['out_dir'])

        experiment_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        
        run_name = f'{experiment_name}_{run_id}'
        self.out_dir = out_dir / run_name

        self.out_dir.mkdir(0o755, parents=True, exist_ok=True)  # make directory for saving checkpoints and log.

        write_json(self.config, self.out_dir / 'config.json')

        setup_logging(self.out_dir) # configure logging module

    @classmethod
    def from_args(cls, args):
        """
        Initialize ConfigParser class from cli arguments.

        Args:
            args (argparse.ArgumentParser): Command line arguments.
        """
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = Path(args.config)
        else:
            assert args.config is not None, "Configuration file needs to be specified. Add, e.g., '-c config.json'"
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)


        return cls(config, resume)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to 
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity="INFO"):
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, verbosity))
        return logger


