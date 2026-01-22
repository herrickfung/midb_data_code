import numpy as np
import pandas as pd
import pathlib
import torch
import json

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image

from EcoSet_Percept.ecoset_loader import EcoSetLoader

TARGET_CLASSES = [
    'car',
    'house',
    'phone',
    'bed',
    'cat',
    'flower',
    'train',
    'elephant',
    'knife',
    'bridge'
]


class AddGaussianBlur(object):
    def __init__(self, kernel, std):
        self.kernel = kernel
        self.std = std

        if self.kernel == 0:
            self.kernel = int(6 * self.std + 1)# OpenCV2 default for kernel size adjustment
            if self.kernel % 2 == 0:
                self.kernel += 1  # kernel size must be odd

    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size=self.kernel, sigma=self.std)(tensor)
        return tensor


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    human_path = current_path / 'human_stim'
    model_path = current_path / 'model_stim'
    human_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    return human_path, model_path


def get_data():
    ecoset_dir = "/storage/coda1/p-drahnev6/0/shared/herrick/ecoset"
    info_dir = pathlib.Path(__file__).parent.absolute() / 'EcoSet_Percept' / 'data'
    npz_dir = 'EcoSet_Percept/data/ecoset_subset.npz'
    loader = EcoSetLoader(ecoset_dir, info_dir, target_classes=TARGET_CLASSES)
    _, _, test = loader.load(npz_path=npz_dir)
    return test


def train_test_split(data):
    labels = [label for _, label in data]
    labels = np.array(labels)

    n_train = 10
    train_indices, test_indices = [], []

    for c in np.unique(labels):
        class_idx = np.where(labels == c)
        train_indices.extend(class_idx[0][:n_train])
        test_indices.extend(class_idx[0][n_train:])
    train_subset = Subset(data, train_indices)
    test_subset = Subset(data, test_indices)    
    return train_subset, test_subset


def human_transform(train_data, test_data, path):
    train_path = path / 'train' 
    test_path = path / 'test'
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    noise_array = [6, 10]  # clear and 2 levels of noise

    base_transform = transforms.Compose([
        transforms.Resize((227, 227), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])

    # train dataset
    for i, (img, label) in enumerate(train_data):
        class_pos = i % 10
        if class_pos < 2:
            noise = 0
        elif class_pos < 4:
            noise = 1
        elif class_pos < 7:
            noise = noise_array[0]
        else:
            noise = noise_array[1]

        trans = transforms.Compose([
            base_transform,
            AddGaussianBlur(kernel=0, std=noise),
            transforms.ToPILImage(),      
            transforms.Resize((400, 400), interpolation=transforms.InterpolationMode.LANCZOS),
        ])
        trans_img = trans(img)  
        # trans_img = transforms.ToPILImage()(trans_img)
        trans_img.save(train_path / f'img_{i:03d}_train_{label}_{noise}.png')

    # test dataset
    for i, (img, label) in enumerate(test_data):
        class_pos = i % 40
        if class_pos < 20:
            noise = noise_array[0]
        else:
            noise = noise_array[1]

        trans = transforms.Compose([
            base_transform,
            AddGaussianBlur(kernel=0, std=noise),
            transforms.ToPILImage(),      
            transforms.Resize((400, 400), interpolation=transforms.InterpolationMode.LANCZOS),
        ])
        trans_img = trans(img)  
        # trans_img = transforms.ToPILImage()(trans_img)
        trans_img.save(test_path / f'img_{i:03d}_test_{label}_{noise}.png')



def model_transform(data, path):
    model_noise_array = np.arange(0.5, 2.025, 0.025)

    for noise in tqdm(model_noise_array, desc="Generating model images ..."):
        model_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            AddGaussianBlur(kernel=0, std=noise),
            transforms.ToPILImage(),
        ])
        label_dict = {}
        for i, (img, label) in enumerate(data):
            img_model = model_transform(img)
            noise_dir = path / f'blur_{noise:.3f}'
            noise_dir.mkdir(parents=True, exist_ok=True)
            img_model.save(noise_dir / f'img_{i:03d}.png')
            label_dict[i] = int(label)
    
    with open(path / "labels.json", 'w') as f:
        json.dump(label_dict, f)
    

def main():
    set_seed()
    human_path, model_path = manage_path()
    test_data = get_data()
    train, test = train_test_split(test_data)
    # human_transform(train, test, human_path)
    model_transform(test, model_path)
    # output_full(test_data, human_path)


if __name__ == "__main__":
    main()