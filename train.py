import argparse
import torch

from parse_config import ConfigParser
from scoring import GetMetricFunction, Metrics

import models as models_module
import dataset as data_module   
import scoring as module_scoring
import trainer as trainer_module

def train(config):
    logger = config.get_logger('train')

    train_loader = config.init_obj('train_data_loader', data_module)
    valid_loader = config.init_obj('val_data_loader', data_module)

    # build model architecture, then print to console
    model = config.init_obj('model', models_module)
    logger.info(f'Starting training script with: { type(model).__name__}')

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = model.to(device)
    if len(config['gpus']) > 1:
        model = torch.nn.DataParallel(model, device_ids=config['gpus'])
        logger.info(f"Runnig in parallel on: {config['gpus']}")

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) # check if none.

    # get function handles of loss and metrics

    loss_fn = getattr(module_scoring, config['loss'])()
    metrics = {met:GetMetricFunction(met) for met in config['metrics']}
    metrics = Metrics(metrics)

    trainer = config.init_obj('trainer', trainer_module, \
                            model, loss_fn, metrics, optimizer, config, device, \
                            train_loader, valid_loader, lr_scheduler)

    trainer.train()

    logger.info('Training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("FVApp training script")
    parser.add_argument('-c', '--config', default='/home/kafkaon1/Dev/FVAPP/configs/config_kp.json', type=str, \
                      help='config file path (str, default: config.json)')
    parser.add_argument('-r', '--resume', default=None, #default='/home/kafkaon1/Dev/out/train/SolAR_MCRNN_0130_162327/best_ckpt_ep14.pth',#'/home/kafkaon1/Dev/out/train/SolAR_MCRNN_1113_064543/ckpt_ep198.pth', type=str, \
                      help='path to the resumed checkpoint (str, default: None)')

    args = parser.parse_args()

    config = ConfigParser.from_args(args)
    train(config)

