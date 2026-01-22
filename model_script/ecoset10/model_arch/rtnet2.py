"""
ABNN implementation based on "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models".
"""

import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

# ------------------------
# BNLD layers
# ------------------------
class BNLDConv(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.noise_std = nn.Parameter(torch.zeros(num_features))  # learnable per channel
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3], keepdim=True)
            var = x.var([0, 2, 3], keepdim=True, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # per-channel noise
        noise = torch.randn_like(x_norm) * self.noise_std.view(1, -1, 1, 1)
        return self.gamma.view(1, -1, 1, 1) * (x_norm + noise) + self.beta.view(1, -1, 1, 1)


class BNLDLinear(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.noise_std = nn.Parameter(torch.zeros(num_features))  # learnable per feature
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True)
            var = x.var(0, keepdim=True, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze(0)
        else:
            mean = self.running_mean.view(1, -1)
            var = self.running_var.view(1, -1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        noise = torch.randn_like(x_norm) * self.noise_std.view(1, -1)
        return self.gamma * (x_norm + noise) + self.beta


# ------------------------
# ABNN Loss
# ------------------------
class ABNNLoss(nn.Module):
    def __init__(self, num_classes, model_parameters, weight_decay=1e-4):
        super().__init__()
        self.model_parameters = list(model_parameters) if model_parameters is not None else []
        self.weight_decay = weight_decay
        self.eta = nn.Parameter(torch.ones(num_classes))

    def set_model_parameters(self, model_parameters):
        self.model_parameters = list(model_parameters)

    def forward(self, outputs, labels):
        eta = self.eta.to(labels.device)
        nll_loss = F.cross_entropy(outputs, labels)
        log_prior_loss = self.negative_log_prior()
        custom_ce_loss = self.custom_cross_entropy_loss(outputs, labels, eta)
        return nll_loss + log_prior_loss + custom_ce_loss

    def negative_log_prior(self):
        if not self.model_parameters:
            return torch.tensor(0, device=self.eta.device)
        total_params = sum(p.numel() for p in self.model_parameters)
        l2_reg = sum(p.pow(2).sum() for p in self.model_parameters)
        return self.weight_decay * (l2_reg / float(total_params))

    def custom_cross_entropy_loss(self, outputs, labels, eta):
        log_probs = F.log_softmax(outputs, dim=1)
        weighted_log_probs = eta[labels] * log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        return -torch.mean(weighted_log_probs)


# ------------------------
# AlexNet & ABNN variants
# ------------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 4, 0), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x)); x = F.dropout(x, 0.5)
        return self.fc3(x)


class AlexNet_BNLD(AlexNet):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.conv1.add_module("bnld_conv1", BNLDConv(96))
        self.conv2.add_module("bnld_conv2", BNLDConv(256))
        self.conv3.add_module("bnld_conv3", BNLDConv(384))
        self.conv4.add_module("bnld_conv4", BNLDConv(384))
        self.conv5.add_module("bnld_conv5", BNLDConv(256))
        self.fc1.add_module("bnld_fc1", BNLDLinear(4096))
        self.fc2.add_module("bnld_fc2", BNLDLinear(4096))
        self.fc3.add_module("bnld_fc3", BNLDLinear(num_classes))


def alexnet_to_abnn(pretrained_model, num_classes=10):
    abnn = AlexNet_BNLD(num_classes=num_classes)
    state_dict = pretrained_model.state_dict()
    abnn_dict = abnn.state_dict()
    for k in abnn_dict.keys():
        if k in state_dict and abnn_dict[k].shape == state_dict[k].shape:
            abnn_dict[k] = state_dict[k]
    abnn.load_state_dict(abnn_dict)
    return abnn


# ------------------------
# LightningModule
# ------------------------
class LitRTNetABNN(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3, weight_decay=1e-6,
                 instance=0, save_dir=None, 
                 pretrain_epochs=20, noise_reg = 1e-2,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.net = AlexNet(num_classes=num_classes)
        self.is_abnn = False
        self.abnn_loss = None
        self.instance = instance
        self.save_dir = save_dir
        self.val_loss_list = None
        self.val_acc_list = None
        self.automatic_optimization = False
        self.current_epoch_count = 0
        self.best_val_acc = 0.0
        self.lr_counter = 0
        self._optimizers = None

    def switch_to_abnn(self):
        self.net = alexnet_to_abnn(self.net, self.hparams.num_classes).to(self.device)
        self.abnn_loss = ABNNLoss(self.hparams.num_classes, self.net.parameters(), self.hparams.weight_decay)
        self.is_abnn = True

        # Keep layers trainable initially
        for param in self.net.parameters():
            param.requires_grad = True

        # Initialize BNLD noise to zero
        for m in self.net.modules():
            if isinstance(m, (BNLDConv, BNLDLinear)):
                m.noise_std.data.zero_()
                m.noise_std.data.clamp_(0, 5)  # clamp to non-negative

        self._optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # ABNN switch
        if not self.is_abnn and self.current_epoch_count >= self.hparams.pretrain_epochs:
            self.switch_to_abnn()

        logits = self.net(inputs)
        if self.is_abnn:
            loss = self.abnn_loss(logits, labels)
            noise_reward = sum((m.noise_std ** 2).sum() for m in self.net.modules()
                               if isinstance(m, (BNLDConv, BNLDLinear)))
            loss -= self.hparams.noise_reg * noise_reward
        else:
            loss = F.cross_entropy(logits, labels)

        opt = self._optimizer
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        self.current_epoch_count += 1

    def on_validation_start(self):
        self.val_loss_list = torch.tensor([], device=self.device)
        self.val_acc_list = torch.tensor([], device=self.device)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.net(inputs)
        loss = self.abnn_loss(logits, labels) if self.is_abnn else F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.val_loss_list = torch.cat((self.val_loss_list, loss.unsqueeze(0)))
        self.val_acc_list = torch.cat((self.val_acc_list, acc.unsqueeze(0)))

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_list.mean()
        avg_val_acc = self.val_acc_list.mean()
        self.log_dict({'val_loss': avg_val_loss, 'val_acc': avg_val_acc})

        # # Optional: save checkpoint if good performance
        # if self.save_dir and avg_val_acc > 0.75:
        #     self.save_dir.mkdir(parents=True, exist_ok=True)
        #     save_path = self.save_dir / f'inst_{self.instance}/epoch_{self.current_epoch}.pth'
        #     save_path.parent.mkdir(parents=True, exist_ok=True)
        #     torch.save({'model_state_dict': self.net.state_dict(),
        #                 'val_loss': avg_val_loss,
        #                 'val_acc': avg_val_acc}, save_path)

    def configure_optimizers(self):
        self._optimizer = torch.optim.Adam(self.net.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return self._optimizer

    def predict(self, x, n_samples=10):
        if not self.is_abnn:
            logits = self.net(x)
            return torch.argmax(F.softmax(logits, dim=1), dim=1)
        yhats = [F.softmax(self.net(x), dim=1) for _ in range(n_samples)]
        return torch.argmax(torch.mean(torch.stack(yhats), dim=0), dim=1)
