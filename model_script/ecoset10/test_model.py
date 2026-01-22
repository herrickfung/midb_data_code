# Standard library imports
import argparse
import os
import random
import warnings

# Third-party libraries
import json
import pickle
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

from model_arch.rtnet import LitRTNet
from model_arch.rtnet2 import LitRTNetABNN
from model_arch.alexnet import LitAlexNet
from model_arch.resnet18 import LitResNet18
from model_arch.model_transform import get_transform

import pyro

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = "ignore"
# Seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


MODEL_REGISTRY = {
    'rtnet': LitRTNet,
    'rtnet2': LitRTNetABNN,
    'alexnet': LitAlexNet,
    'resnet18': LitResNet18,
}


class FlatImageLabelDataset(Dataset):
    def __init__(self, flist, image_dir, transform=None):
        """
        flist: list of tuples (filename, label)
        image_dir: folder where all images are saved (flat, no label subfolders)
        """
        self.flist = flist
        self.image_dir = image_dir
        self.transform = transform
        self.classes = sorted(list(set([lbl for _, lbl in flist])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        img_name, label = self.flist[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx


def load_flist(filepath):
    with open(filepath, 'r') as f:
        flist = json.load(f)
    return flist


def get_test_loader(blur):
    # Parse command-line arguments
    current_dir = pathlib.Path(__file__).parent.absolute()
    img_dir = f'{current_dir}/images/model_stim/blur_{blur:.3f}'
    labels_path = f'{current_dir}/images/model_stim/labels.json'

    with open(labels_path, "r") as f:
        flist_dict = json.load(f)
    flist = [(f"img_{int(idx):03d}.png", lbl) for idx, lbl in flist_dict.items()]

    testset = FlatImageLabelDataset(flist, img_dir, transform=get_transform('default'))
    test_loader = DataLoader(testset, batch_size=400, shuffle=False, pin_memory=True)
    return test_loader


def get_model(model_name, inst_no, epoch):
    model_path = f'checkpoints/{model_name}/inst_{inst_no}/epoch_{epoch}.pth'
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(10, 1e-4, 1e-3, inst_no)
    checkpoint = torch.load(model_path, weights_only=True)

    if model_name == 'rtnet':
        model.net.load_state_dict(checkpoint['model_state_dict'])
        pyro.clear_param_store()
        for name, value in checkpoint['pyro_params'].items():
            pyro.param(name, value)
        val_loss, val_acc = checkpoint['val_loss'], checkpoint['val_acc']
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_loss, val_acc = checkpoint['val_loss'], checkpoint['val_acc']
    return model, val_loss, val_acc


def test(model_name, inst_no, epoch, blur, model, 
         test_loader, val_loss, val_acc, 
         rep=1, device='cuda'
         ):
    model.to(device)
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs, rt_array = outputs
        else:
            rt_array = [None] * outputs.shape[0]
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == labels)
        softmax = F.softmax(outputs, dim=1)
        top2vals, _ = torch.topk(outputs, 2, dim = 1)
        pe_conf = top2vals[:, 0]
        top2diff_conf = top2vals[:, 0] - top2vals[:, 1]

    # move to cpu
    labels_np = labels.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    correct_np = correct.cpu().numpy().astype(int)
    conf_ev_array = outputs.cpu().numpy()
    softmax_array = softmax.cpu().numpy()
    softmax_conf_np = np.max(softmax_array, axis=1)
    pe_conf_np = pe_conf.cpu().numpy()
    top2diff_conf_np = top2diff_conf.cpu().numpy()
    rt_array_np = rt_array.cpu().numpy()

    classes = test_loader.dataset.classes
    stim_labels = [classes[label] for label in labels_np]
    pred_labels = [classes[label] for label in predicted_np]

    results = [
        {
            'inst': inst_no,
            'epoch': epoch,
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'image_index': x,
            'blur': blur,
            'stim': stim_labels[x],
            'rep': rep,
            'resp': pred_labels[x],
            'acc': correct_np[x],
            'rt': rt_array_np[x],
            'conf_softmax': softmax_conf_np[x],
            'conf_pe': pe_conf_np[x],
            'conf_top2diff': top2diff_conf_np[x],
            'softmax': softmax_array[x],
            'conf_ev': conf_ev_array[x],
        } for x in range(len(labels_np))
    ]

    print(f'Model {model_name} | Inst {inst_no} | Epoch {epoch} | Blur {blur:.3f} | Test Acc: {np.mean(correct_np)*100:.2f}%')
    return results


def main():
    parser = argparse.ArgumentParser(description='Test model.')
    parser.add_argument('--model', type=str, help='Model architecture')
    parser.add_argument('--instance', type=int, required=True, help='Instance number')
    args = parser.parse_args()

    # # setup to test N instances in N level of blurs
    # epochs = np.arange(0, 30, 1)
    # blurs = np.arange(0.5, 2.025, 0.025)

    # RTNet
    reps = 10
    epochs = [149]
    blurs = [0.9, 1.525]
    all_results = []

    # for rep in range(reps):
    save_dir = pathlib.Path(f'results/{args.model}')
    save_dir.mkdir(exist_ok=True, parents=True)
    for blur in tqdm(blurs, desc='Testing blur level ...'):
        test_loader = get_test_loader(blur)
        for epoch in epochs:
            model, val_loss, val_acc = get_model(args.model, args.instance, epoch)
            for rep in range(reps):
                result = test(
                            args.model,
                            args.instance, 
                            epoch, 
                            blur, 
                            model,
                            test_loader,
                            val_loss, 
                            val_acc,
                            rep,
                            )
                all_results.extend(result)

        # save every blur
        with open(save_dir / f'inst_{args.instance}.pkl', 'wb') as f:
            pickle.dump(all_results, f)


if __name__ == "__main__":
    main()
