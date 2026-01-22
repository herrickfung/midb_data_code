import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision import models
import numpy as np
import pathlib
import pytorch_lightning as pl


class LitResNet18(pl.LightningModule):
    def __init__(self, 
                 num_classes: int,
                 lr: float,
                 weight_decay: float,
                 instance: int,
                 save_dir: str = None
                 ):

        super(LitResNet18, self).__init__()
        self.save_hyperparameters()
        self.instance = instance
        self.save_dir = save_dir

        # Define the ResNet-18 architecture
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.resnet18.fc.in_features, num_classes)
        )
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')
        self.val_loss_list = None
        self.val_acc_list = None

    def forward(self, x):
        return self.resnet18(x)

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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'reduce_on_plateau': True,
            }
        }
