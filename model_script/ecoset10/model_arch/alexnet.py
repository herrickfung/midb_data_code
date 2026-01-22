import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics import Accuracy
import numpy as np
import pathlib
import pytorch_lightning as pl


class LitAlexNet(pl.LightningModule):
    def __init__(self, 
        num_classes: int,
        lr: float,
        weight_decay: float,
        instance: int,
        save_dir: str = None
    ):
        super(LitAlexNet, self).__init__()
        self.save_hyperparameters()
        self.instance = instance
        self.save_dir = save_dir

        # Define the AlexNet architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
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
        self.val_loss_list = None
        self.val_acc_list = None

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

    def on_validation_start(self):
        self.val_loss_list = torch.tensor([], device=self.device)
        self.val_acc_list = torch.tensor([], device=self.device)

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
        self.log('val_loss_step', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_step', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_loss_list = torch.cat((self.val_loss_list, loss.detach().unsqueeze(0)))
        self.val_acc_list = torch.cat((self.val_acc_list, acc.detach().unsqueeze(0)))

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_list.mean()
        avg_val_acc = self.val_acc_list.mean()
        self.log_dict({
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc
        })

        # save model weights and val metrics
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'inst_{self.instance}/epoch_{self.current_epoch}.pth'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc
        }, save_path)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'reduce_on_plateau': True,
            }
        }