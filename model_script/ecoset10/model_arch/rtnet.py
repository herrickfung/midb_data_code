import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics import Accuracy
import pytorch_lightning as pl

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine



class KLAnnealingELBO(Trace_ELBO):
    def __init__(self, kl_weight_fn=lambda: 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight_fn = kl_weight_fn

    def _differentiable_loss_particle(self, model, guide, *args, **kwargs):
        # Run guide and model
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

        # log p(x|z)
        log_likelihood = model_trace.log_prob_sum()
        # log q(z) - log p(z)
        kl = guide_trace.log_prob_sum() - model_trace.log_prob_sum()

        beta = self.kl_weight_fn()
        # Only scale KL
        loss = -(log_likelihood - beta * kl)
        return loss

# AlexNet
class alexnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
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

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        return out

def get_init(inst):
    init_lookup = {}
    loc_values = torch.arange(0, 1.2, 0.1).tolist()
    # # for first set of training, most of the untrainable models are the low scale values in first batch
    # # these models doesn't train 5,10,15,20,21,25,26,27,30,31,32,35,36,37,40,41,42,45,46,47,48,50,51,52,55,56,57,58,59
    # # these models doesn't train 4, use third line scale values
    # scale_values = [1, 2, 3, 4, 5]  
    # scale_values = [6, 7, 8, 9, 10]
    scale_values = [4.5, 4.5, 4.5, 4.5, 4.5]
    idx = 0
    for loc in loc_values:
        for scale in scale_values:
            init_lookup[idx] = {'loc': loc, 'scale': scale}
            idx += 1
    return init_lookup.get(inst)


def make_model(net, inst):
    log_softmax = nn.LogSoftmax(dim=1)
    def model(x_data, y_data):
        init = get_init(inst)
        scale_factor = 0.05

        convLayer1_w = Normal(loc=torch.full_like(net.conv1[0].weight, init['loc']), scale=torch.full_like(net.conv1[0].weight, init['scale'] * scale_factor))
        convLayer1_b = Normal(loc=torch.full_like(net.conv1[0].bias, init['loc']), scale=torch.full_like(net.conv1[0].bias, init['scale'] * scale_factor))
        convLayer2_w = Normal(loc=torch.full_like(net.conv2[0].weight, init['loc']), scale=torch.full_like(net.conv2[0].weight, init['scale'] * scale_factor))
        convLayer2_b = Normal(loc=torch.full_like(net.conv2[0].bias, init['loc']), scale=torch.full_like(net.conv2[0].bias, init['scale'] * scale_factor))
        convLayer3_w = Normal(loc=torch.full_like(net.conv3[0].weight, init['loc']), scale=torch.full_like(net.conv3[0].weight, init['scale'] * scale_factor))
        convLayer3_b = Normal(loc=torch.full_like(net.conv3[0].bias, init['loc']), scale=torch.full_like(net.conv3[0].bias, init['scale'] * scale_factor))
        convLayer4_w = Normal(loc=torch.full_like(net.conv4[0].weight, init['loc']), scale=torch.full_like(net.conv4[0].weight, init['scale'] * scale_factor))
        convLayer4_b = Normal(loc=torch.full_like(net.conv4[0].bias, init['loc']), scale=torch.full_like(net.conv4[0].bias, init['scale'] * scale_factor))
        convLayer5_w = Normal(loc=torch.full_like(net.conv5[0].weight, init['loc']), scale=torch.full_like(net.conv5[0].weight, init['scale'] * scale_factor))
        convLayer5_b = Normal(loc=torch.full_like(net.conv5[0].bias, init['loc']), scale=torch.full_like(net.conv5[0].bias, init['scale'] * scale_factor))

        fc1Layer_w = Normal(loc=torch.full_like(net.fc1.weight, init['loc']), scale=torch.full_like(net.fc1.weight, init['scale'] * scale_factor))
        fc1Layer_b = Normal(loc=torch.full_like(net.fc1.bias, init['loc']), scale=torch.full_like(net.fc1.bias, init['scale'] * scale_factor))
        fc2Layer_w = Normal(loc=torch.full_like(net.fc2.weight, init['loc']), scale=torch.full_like(net.fc2.weight, init['scale'] * scale_factor))
        fc2Layer_b = Normal(loc=torch.full_like(net.fc2.bias, init['loc']), scale=torch.full_like(net.fc2.bias, init['scale'] * scale_factor))
        fc3Layer_w = Normal(loc=torch.full_like(net.fc3.weight, init['loc']), scale=torch.full_like(net.fc3.weight, init['scale'] * scale_factor))
        fc3Layer_b = Normal(loc=torch.full_like(net.fc3.bias, init['loc']), scale=torch.full_like(net.fc3.bias, init['scale'] * scale_factor))

        priors = {
            'conv1[0].weight': convLayer1_w,
            'conv1[0].bias': convLayer1_b,
            'conv2[0].weight': convLayer2_w,
            'conv2[0].bias': convLayer2_b,
            'conv3[0].weight': convLayer3_w,
            'conv3[0].bias': convLayer3_b,
            'conv4[0].weight': convLayer4_w,
            'conv4[0].bias': convLayer4_b,
            'conv5[0].weight': convLayer5_w,
            'conv5[0].bias': convLayer5_b,

            'fc1.weight': fc1Layer_w,
            'fc1.bias': fc1Layer_b,
            'fc2.weight': fc2Layer_w,
            'fc2.bias': fc2Layer_b,
            'fc3.weight': fc3Layer_w,
            'fc3.bias': fc3Layer_b
            }

        lifted_module = pyro.random_module("module", net, priors)()
        lhat = log_softmax(lifted_module(x_data))
        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
    return model


def make_guide(net):
    softplus = nn.Softplus()
    def guide(x_data, y_data):
        scale_factor = 1

        # First layer weight distribution priors
        convLayer1w_mu    = torch.randn_like(net.conv1[0].weight)
        convLayer1w_sigma = torch.randn_like(net.conv1[0].weight)
        convLayer1w_mu_param    = pyro.param("convLayer1w_mu", convLayer1w_mu)
        convLayer1w_sigma_param = softplus(pyro.param("convLayer1w_sigma", convLayer1w_sigma))
        convLayer1_w = Normal(loc=convLayer1w_mu_param, scale=convLayer1w_sigma_param * scale_factor)
        # First layer bias distribution priors
        convLayer1b_mu    = torch.randn_like(net.conv1[0].bias)
        convLayer1b_sigma = torch.randn_like(net.conv1[0].bias)
        convLayer1b_mu_param    = pyro.param("convLayer1b_mu", convLayer1b_mu)
        convLayer1b_sigma_param = softplus(pyro.param("convLayer1b_sigma", convLayer1b_sigma))
        convLayer1_b = Normal(loc=convLayer1b_mu_param, scale=convLayer1b_sigma_param * scale_factor)
        # Second layer weight distribution priors
        convLayer2w_mu    = torch.randn_like(net.conv2[0].weight)
        convLayer2w_sigma = torch.randn_like(net.conv2[0].weight)
        convLayer2w_mu_param    = pyro.param("convLayer2w_mu", convLayer2w_mu)
        convLayer2w_sigma_param = softplus(pyro.param("convLayer2w_sigma", convLayer2w_sigma))
        convLayer2_w = Normal(loc=convLayer2w_mu_param, scale=convLayer2w_sigma_param * scale_factor)
        # Second layer bias distribution priors
        convLayer2b_mu    = torch.randn_like(net.conv2[0].bias)
        convLayer2b_sigma = torch.randn_like(net.conv2[0].bias)
        convLayer2b_mu_param    = pyro.param("convLayer2b_mu", convLayer2b_mu)
        convLayer2b_sigma_param = softplus(pyro.param("convLayer2b_sigma", convLayer2b_sigma))
        convLayer2_b = Normal(loc=convLayer2b_mu_param, scale=convLayer2b_sigma_param * scale_factor)
        # Third layer weight distribution priors
        convLayer3w_mu    = torch.randn_like(net.conv3[0].weight)
        convLayer3w_sigma = torch.randn_like(net.conv3[0].weight)
        convLayer3w_mu_param    = pyro.param("convLayer3w_mu", convLayer3w_mu)
        convLayer3w_sigma_param = softplus(pyro.param("convLayer3w_sigma", convLayer3w_sigma))
        convLayer3_w = Normal(loc=convLayer3w_mu_param, scale=convLayer3w_sigma_param * scale_factor)
        # Third layer bias distribution priors
        convLayer3b_mu    = torch.randn_like(net.conv3[0].bias)
        convLayer3b_sigma = torch.randn_like(net.conv3[0].bias)
        convLayer3b_mu_param    = pyro.param("convLayer3b_mu", convLayer3b_mu)
        convLayer3b_sigma_param = softplus(pyro.param("convLayer3b_sigma", convLayer3b_sigma))
        convLayer3_b = Normal(loc=convLayer3b_mu_param, scale=convLayer3b_sigma_param * scale_factor)
        # Fourth layer weight distribution priors
        convLayer4w_mu    = torch.randn_like(net.conv4[0].weight)
        convLayer4w_sigma = torch.randn_like(net.conv4[0].weight)
        convLayer4w_mu_param    = pyro.param("convLayer4w_mu", convLayer4w_mu)
        convLayer4w_sigma_param = softplus(pyro.param("convLayer4w_sigma", convLayer4w_sigma))
        convLayer4_w = Normal(loc=convLayer4w_mu_param, scale=convLayer4w_sigma_param * scale_factor)
        # Fourth layer bias distribution priors
        convLayer4b_mu    = torch.randn_like(net.conv4[0].bias)
        convLayer4b_sigma = torch.randn_like(net.conv4[0].bias)
        convLayer4b_mu_param    = pyro.param("convLayer4b_mu", convLayer4b_mu)
        convLayer4b_sigma_param = softplus(pyro.param("convLayer4b_sigma", convLayer4b_sigma))
        convLayer4_b = Normal(loc=convLayer4b_mu_param, scale=convLayer4b_sigma_param * scale_factor)
        # Fifth layer weight distribution priors
        convLayer5w_mu    = torch.randn_like(net.conv5[0].weight)
        convLayer5w_sigma = torch.randn_like(net.conv5[0].weight)
        convLayer5w_mu_param    = pyro.param("convLayer5w_mu", convLayer5w_mu)
        convLayer5w_sigma_param = softplus(pyro.param("convLayer5w_sigma", convLayer5w_sigma))
        convLayer5_w = Normal(loc=convLayer5w_mu_param, scale=convLayer5w_sigma_param * scale_factor)
        # Fifth layer bias distribution priors
        convLayer5b_mu    = torch.randn_like(net.conv5[0].bias)
        convLayer5b_sigma = torch.randn_like(net.conv5[0].bias)
        convLayer5b_mu_param    = pyro.param("convLayer5b_mu", convLayer5b_mu)
        convLayer5b_sigma_param = softplus(pyro.param("convLayer5b_sigma", convLayer5b_sigma))
        convLayer5_b = Normal(loc=convLayer5b_mu_param, scale=convLayer5b_sigma_param * scale_factor)

        # First fully connected layer weight distribution priors
        fc1w_mu = torch.randn_like(net.fc1.weight)
        fc1w_sigma = torch.randn_like(net.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1Layer_w = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param * scale_factor).independent(1)

        # First fully connected layer bias distribution priors
        fc1b_mu = torch.randn_like(net.fc1.bias)
        fc1b_sigma = torch.randn_like(net.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1Layer_b = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param * scale_factor)

        # Second fully connected layer weight distribution priors
        fc2w_mu = torch.randn_like(net.fc2.weight)
        fc2w_sigma = torch.randn_like(net.fc2.weight)
        fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
        fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
        fc2Layer_w = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param * scale_factor).independent(1)

        # Second fully connected layer bias distribution priors
        fc2b_mu = torch.randn_like(net.fc2.bias)
        fc2b_sigma = torch.randn_like(net.fc2.bias)
        fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
        fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
        fc2Layer_b = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param * scale_factor)

        # Third fully connected layer weight distribution priors
        fc3w_mu = torch.randn_like(net.fc3.weight)
        fc3w_sigma = torch.randn_like(net.fc3.weight)
        fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
        fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
        fc3Layer_w = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param * scale_factor).independent(1)

        # Third fully connected layer bias distribution priors
        fc3b_mu = torch.randn_like(net.fc3.bias)
        fc3b_sigma = torch.randn_like(net.fc3.bias)
        fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
        fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
        fc3Layer_b = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param * scale_factor)

        priors = {
            'conv1[0].weight': convLayer1_w,
            'conv1[0].bias': convLayer1_b,
            'conv2[0].weight': convLayer2_w,
            'conv2[0].bias': convLayer2_b,
            'conv3[0].weight': convLayer3_w,
            'conv3[0].bias': convLayer3_b,
            'conv4[0].weight': convLayer4_w,
            'conv4[0].bias': convLayer4_b,
            'conv5[0].weight': convLayer5_w,
            'conv5[0].bias': convLayer5_b,

            'fc1.weight': fc1Layer_w,
            'fc1.bias': fc1Layer_b,
            'fc2.weight': fc2Layer_w,
            'fc2.bias': fc2Layer_b,
            'fc3.weight': fc3Layer_w,
            'fc3.bias': fc3Layer_b
            }

        lifted_module = pyro.random_module("module", net, priors)
        return lifted_module()
    return guide


class LitRTNet(pl.LightningModule):
    def __init__(self, 
                 num_classes: int, 
                 lr: float, 
                 weight_decay: float,
                 instance: int, 
                 save_dir: str = None
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.net = alexnet(num_classes=num_classes)
        self.model=make_model(self.net, instance)
        self.guide=make_guide(self.net)

        self.kl_weight = 0
        self.svi = SVI(
            model = self.model,
            guide = self.guide,
            optim = Adam({"lr": self.hparams.lr}),
            loss = KLAnnealingELBO(kl_weight_fn = lambda: self.kl_weight)
        )

        self.instance = instance
        self.save_dir = save_dir
        self.val_loss_list = None
        self.val_acc_list = None
        self.automatic_optimization = False

    def compute_evidence(self, x, device):
        sampled_models = [self.guide(None, None).to(device) for _ in range(1)]
        yhats = [F.log_softmax(model(x), dim=1) for model in sampled_models][0]
        return yhats

    def forward(self,x, threshold=8):
        # forward function use in testing the data, produce rt
        device = x.device
        batch_size = x.shape[0]
        rts = torch.zeros(batch_size, device=device)
        evidence_array = torch.zeros(batch_size, 10, device=device)
        evidence_count = torch.zeros(batch_size, 10, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)  # tracks which images still accumulating

        while active.any():
            conf_ev = self.compute_evidence(x[active], device=device)
            evidence_array[active] += conf_ev
            evidence_count[active] += torch.exp(conf_ev)  # total evidence convert evidence to count
            max_count = torch.max(evidence_count[active], dim=-1).values # get max evidence
            active_indices = torch.where(active)[0]
            still_active = max_count < threshold
            active[active_indices] = still_active
            rts[active] += 1
        return evidence_array, rts

    def predict(self, x, n_samples=10):
        # forward function use during training for validation
        sampled_models = [self.guide(None, None) for _ in range(n_samples)]
        yhats = [F.softmax(model(x), dim=1) for model in sampled_models]
        mean_probs = torch.mean(torch.stack(yhats), dim=0)
        return torch.argmax(mean_probs, dim=1)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        beta = float(1 / (1 + np.exp(-0.2 * (self.current_epoch - 15))))
        self.kl_weight = beta
        loss = self.svi.step(inputs, labels)
        loss /= len(inputs)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_start(self):
        self.val_loss_list = torch.tensor([], device=self.device)
        self.val_acc_list = torch.tensor([], device=self.device)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.svi.evaluate_loss(inputs, labels)
        loss_tensor = torch.tensor(loss, device=self.device)
        preds = self.predict(inputs, n_samples=10)
        acc = (preds == labels).float().mean()

        self.log('val_loss_step', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_step', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_loss_list = torch.cat((self.val_loss_list, loss_tensor.unsqueeze(0)))
        self.val_acc_list = torch.cat((self.val_acc_list, acc.detach().unsqueeze(0)))

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_list.mean()
        avg_val_acc = self.val_acc_list.mean()
        self.log_dict({
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc
        })

        # save model with > 0.8 accuracy and last epoch
        # if avg_val_acc > 0.8 or self.current_epoch == self.trainer.max_epochs - 1:
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f'inst_{self.instance}/epoch_{self.current_epoch}.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            param_store = pyro.get_param_store()
            params_dict = {name: value.detach().cpu() for name, value in param_store.named_parameters()}
            torch.save({
                'model_state_dict': self.net.state_dict(),
                'pyro_params': params_dict,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc
            }, save_path)

    def configure_optimizers(self):
        return []
