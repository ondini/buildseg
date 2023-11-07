import torch
from abc import abstractmethod
from numpy import inf
from utils import TensorboardWriter

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.optimizer = optimizer

        self.epochs = config['trainer']['num_epochs']
        self.save_period = config['trainer']['save_period']
        self.monitor = config['trainer'].get('save_metric')

        # configuration to monitor model performance and save best
        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = config['trainer'].get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.out_dir, config['trainer']['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod 
    def _train_epoch(self, epoch):
        """
        Abstract method for epoch training logic.

        Args:
            epoch (int): Current epoch number.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Abstract method for epoch validation logic.

        Args:
            epoch (int): Current epoch number.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic running the single epoch iterations.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info(f"=> Starting {epoch}. epoch.")

            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(val_log)
            log.update(**{'train_'+k : v for k, v in train_log.items()})

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info(f'  => {str(key):10s}: {value}')
    
            # check if the model performance improved, according to mnt_metric, if yes save it
            improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
        
            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info(f"Validation performance didn\'t improve for {self.early_stop} epochs. Stopping.")
                break

            if improved or epoch % self.save_period == 0:
                self._save_checkpoint(epoch, improved)

    def _save_checkpoint(self, epoch, improved=False):
        """
        Saving checkpoints

        Args:
            epoch (int): Current epoch number.
            improved (bool): If model performance improved.
        """
        arch = type(self.model).__name__
        state = {
            'architecture': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        prefix = 'best_' if improved else ''
        filename = str(self.checkpoint_dir / (prefix + f'ckpt_ep{epoch}.pth'))
        torch.save(state, filename)
        self.logger.info(f"  => Checkpoint saved as: {filename}!")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        Args:
            resume_path (str): Checkpoint path to be resumed.

        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['name'] != self.config['name']:
            self.logger.warning("WARNING: Model name given in the config file is different from the one in checkpoint. \
                                 This may lead to an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("WARNING: Optimizer type given in config file is different from the on in checkpoint. \
                                Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"  => Checkpoint loaded. Resuming training from epoch {self.start_epoch}")
