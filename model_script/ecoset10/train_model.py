import os
import warnings
import random
import argparse
import pathlib

import numpy as np
import torch
import pytorch_lightning as pl
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from model_arch.rtnet import LitRTNet
from model_arch.rtnet2 import LitRTNetABNN
from model_arch.alexnet import LitAlexNet
from model_arch.resnet18 import LitResNet18
from model_arch.model_transform import get_transform
from images.EcoSet_Percept.ecoset_loader import EcoSetLoader

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = "ignore"

MODEL_REGISTRY = {
    'rtnet': LitRTNet,
    'rtnet2': LitRTNetABNN,
    'alexnet': LitAlexNet,
    'resnet18': LitResNet18,
}


def set_random_seed(inst):
    # generate a list of seeds for 36 instances,
    # this will be always the same
    np.random.seed(42)
    random_seeds = np.random.choice(range(1,10000), size=100, replace=False)
    inst_seed = random_seeds[inst]

    # set seeds for training
    np.random.seed(inst_seed)
    random.seed(inst_seed)
    torch.manual_seed(inst_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inst_seed)
    pl.seed_everything(inst_seed)


def load_dataset(model_name):
    train, val, _ = EcoSetLoader.load(
        npz_path = 'images/EcoSet_Percept/data/ecoset_subset.npz',
        transform = get_transform(model_name),
    )

    train_loader = DataLoader(dataset=train,
                              batch_size = 512,
                              shuffle=True,
                              )

    val_loader = DataLoader(dataset=val,
                            batch_size = 500,
                            shuffle=False,
                            )

    return train_loader, val_loader


def main():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        pl.seed_everything(42)

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--model', type=str, help='Model architecture')
    parser.add_argument('--instance', type=int, required=True, help='Instance number')
    args = parser.parse_args()
    train_loader, test_loader = load_dataset(args.model)
    set_random_seed(args.instance)

    # Initialize W&B Logger
    wandb_logger = WandbLogger(project=f'{args.model} on ecoset10',
                               name = f"instance={args.instance}",
                               )
    ModelClass = MODEL_REGISTRY.get(args.model)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Initialize the model
    model = ModelClass(num_classes=10, lr=1e-4, weight_decay=1e-3,
                       instance=args.instance, 
                       save_dir=pathlib.Path(f'checkpoints/{args.model}')
                       )

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=150,
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=20,
    )
    wandb_logger.watch(model, log='all', log_graph=False)

    # Start training
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
