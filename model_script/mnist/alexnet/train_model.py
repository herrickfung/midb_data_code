import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
import argparse

import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics import Accuracy
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pathlib
import wandb

from torch import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['PYTHONWARNINGS'] = "ignore"


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


def load_dataset():
    data_path = pathlib.Path(__file__).parent.parent.absolute()

    AlexTransform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root=f'{data_path}/data',
                                   train=True,
                                   transform=AlexTransform,
                                   download=True,
                                   )

    test_dataset = datasets.MNIST(root=f'{data_path}/data',
                                  train=False,
                                  transform=AlexTransform,
                                  )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = 128,
                              shuffle=True
                              )

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size = 128,
                              shuffle=False
                              )

    return train_loader, test_loader


class LitAlexNet(pl.LightningModule):
    def __init__(self, num_classes=16, lr=1e-4, weight_decay=1e-3):
        super(LitAlexNet, self).__init__()
        self.save_hyperparameters()

        # Define the AlexNet architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

def main():
    parser = argparse.ArgumentParser(description='Train LitAlexnet model.')
    parser.add_argument('--instance', type=int, required=True, help='Instance number')
    args = parser.parse_args()
    set_random_seed(args.instance)

    wandb_logger = WandbLogger(project='Alexnet on 8-Choice MNIST',
                               name = f"instance={args.instance}",
                               )  # Add the project name

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model/',
        filename=f'model-{args.instance}-ckpt',
        save_top_k=3,
        mode='min'
    )

    # Initialize the model
    model = LitAlexNet(num_classes=10, lr=1e-3)

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=15,
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=20,
    )
    wandb_logger.watch(model, log='all', log_graph=False)

    # Start training
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    # these control for randomness among the training datasets
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        pl.seed_everything(42)
    train_loader, test_loader = load_dataset()
    main()
