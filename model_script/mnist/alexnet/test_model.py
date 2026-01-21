# Standard library imports
import os
import random
import warnings

# Third-party libraries
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
from torch import nn, optim
from torchmetrics import Accuracy
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
from scipy.io import loadmat
from tqdm import tqdm
import pathlib

# PyTorch Lightning callbacks
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = "ignore"
# Seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_index(filepath):
    low = np.load(filepath, allow_pickle=True)['low']
    high = np.load(filepath, allow_pickle=True)['high']
    index = np.concatenate([low, high]) - 1
    return index


def get_test_loader(noise, seed=42):
    # Parse command-line arguments
    data_path = pathlib.Path(__file__).parent.parent.absolute()
    test_dataset = datasets.MNIST(root=f'{data_path}/data', train=False, download=True)
    index = load_index('noise_index.npz')
    test_dataset.data = test_dataset.data[index]
    test_dataset.targets = test_dataset.targets[index]

    # do transforms add torch.rand noise
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand_like(x) * noise),
    ])
    test_dataset.transform = transform
    test_loader = DataLoader(test_dataset, batch_size=480, shuffle=False, pin_memory=True)
    return test_loader, index


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


def get_model(inst_no):
    model_path = f'model/model-{inst_no}-ckpt.ckpt'
    model = LitAlexNet(num_classes=10)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def test(inst_no, noise, model, test_loader, index):
    model.eval()
    results = []

    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            true_label = label.cpu().numpy()
            stimulus_label = [test_loader.dataset.classes[label] for label in true_label]
            predicted_label = [test_loader.dataset.classes[pred] for pred in predicted]
            correct = (predicted == label).cpu().numpy()

            # calculate confidence, softmax and top2diff
            softmax_conf = torch.max(F.softmax(outputs, dim=1), 1)[0].cpu().numpy()
            top_two_vals, top_two_indices = torch.topk(outputs, 2, dim=1)
            top_choice = top_two_vals[:, 0]
            second_choice = top_two_vals[:, 1]
            top2diff_conf = (top_choice - second_choice).cpu().numpy()

            results = [
                {
                    'inst': inst_no,
                    'minst_index': index[x],
                    'noise': noise,
                    'stim': stimulus_label[x][0],
                    'resp': predicted_label[x][0],
                    'acc': int(correct[x]),
                    'softmax_conf': softmax_conf[x],
                    'top2diff_conf': top2diff_conf[x],
                } for x in range(len(image))
            ]

    results = pd.DataFrame(results)
    return results


def main():
    # setup to test N instances in N level of noise
    instances = np.arange(0, 60, 1)
    noises = np.arange(1, 3.5, 0.01)
    all_results = pd.DataFrame()

    for inst in tqdm(instances, desc='Instances'):
        for noise in tqdm(noises, desc='Noises', leave=False):
            test_loader, index = get_test_loader(noise)
            model = get_model(inst)
            results = test(inst, noise, model, test_loader, index)
            all_results = pd.concat([all_results, results], ignore_index=True)
            all_results.to_csv('raw_cnn_data.csv', index=False)


if __name__ == "__main__":
    main()
