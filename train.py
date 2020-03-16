import models as module_model
import metrics as module_metric
import losses as module_loss
import data_loaders as module_data_loader
import utils.optim as module_optimizer
import utils.lr_scheduler as module_lr_scheduler
import argparse
import torch
import numpy as np
from data_loaders import _create_transforms
from trainer import SegmentationTrainer
from parse_config import ConfigParser


# fix random seed for reproducibility
SEED = 4444
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):

    # TODO
    joint_transforms, image_transforms, target_transforms, val_transform = _create_transforms(config)
    train_data_loader = config.init_obj('train_data_loader', module_data_loader, joint_transforms=joint_transforms,
                                        image_transforms=image_transforms, target_transforms=target_transforms)
    valid_data_loader = config.init_obj('val_data_loader', module_data_loader, image_transforms=val_transform,
                                        target_transforms=target_transforms)

    criterion = config.init_obj('dice_loss', module_loss)
    metrics = [getattr(module_metric, x) for x in config['metrics']]
    model = config.init_obj('model', module_model)
    optimizer = config.init_obj('optimizer', module_optimizer, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, optimizer)

    if config['trainer']['name'] == 'SegmentationTrainer':
        trainer = SegmentationTrainer(model, criterion, metrics, optimizer, config, lr_scheduler)
    else:
        raise NotImplementedError("Unsupported trainer")

    trainer.setup_loader(train_data_loader, valid_data_loader, None)

    trainer.train()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Semantic Segmentation')
    args_parser.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path, default: config.json')

    args = args_parser.parse_args()
    config = ConfigParser(args.config)
    main(config)