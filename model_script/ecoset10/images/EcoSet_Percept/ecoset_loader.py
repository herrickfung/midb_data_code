from concurrent.futures import ThreadPoolExecutor
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from tqdm import tqdm


class NumpyDataset(Dataset):
    """Dataset holding images and labels as numpy arrays, transforms applied dynamically."""
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs  # numpy array (H,W,C) or PIL.Image
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # convert to PIL for transforms
        if self.transform:
            img = self.transform(img)
        return img, label


class EcoSetLoader:
    def __init__(self, 
                 ecoset_dir: str, 
                 info_dir: str, 
                 target_classes: list, 
                 seed: int=42):
        self.ecoset_dir = ecoset_dir
        self.info_dir = pathlib.Path(f"{info_dir}/ecoset_info.csv")
        self.target_classes = target_classes
        self.seed = seed
        self.info = self._read_info()

    def _read_info(self):
        info = pd.read_csv(self.info_dir)
        info.category_index = info.category_index - 1
        return info

    def _class_filter(self):
        return self.info[self.info.category_name.isin(self.target_classes)].index.tolist()

    def _balance_indices(self, labels, n_per_class=None):
        np.random.seed(self.seed)
        unique_classes = np.unique(labels)
        if n_per_class is None:
            n_per_class = min([np.sum(labels == cls) for cls in unique_classes])
        selected_idx = []
        for cls in unique_classes:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) > n_per_class:
                cls_idx = np.random.choice(cls_idx, n_per_class, replace=False)
            selected_idx.extend(cls_idx)
        return selected_idx

    def save_subset(self, n_test_img: int, save_path: str):
        """Load EcoSet, filter, balance, and save all images and labels as numpy arrays."""
        retain_idx = self._class_filter()
        class_to_new = {old: new for new, old in enumerate(retain_idx)}

        data_cache = {}
        for split in ['train', 'val', 'test']:
            ds = datasets.ImageFolder(root=pathlib.Path(f"{self.ecoset_dir}/{split}"), transform=None)
            indices = [i for i, label in enumerate(ds.targets) if label in retain_idx]
            labels = np.array([class_to_new[ds.targets[i]] for i in indices])

            if split == 'train':
                sel_idx = self._balance_indices(labels)
            elif split == 'test':
                sel_idx = self._balance_indices(labels, n_per_class=n_test_img)
            else:
                sel_idx = np.arange(len(labels))

            def process_image(i):
                img, _ = ds[i]
                img = img.resize((227, 227))
                return np.array(img)

            with ThreadPoolExecutor(max_workers=12) as executor:
                imgs = list(tqdm(executor.map(process_image, np.array(indices)[sel_idx]),
                                 total=len(sel_idx), desc=f"{split}"))
            imgs = np.stack(imgs)
            labels = labels[sel_idx]
            data_cache[split] = (imgs, labels)

        np.savez(save_path,
                    train_imgs=data_cache['train'][0],
                    train_labels=data_cache['train'][1],
                    val_imgs=data_cache['val'][0],
                    val_labels=data_cache['val'][1],
                    test_imgs=data_cache['test'][0],
                    test_labels=data_cache['test'][1],
                )
        print(f"Saved subset to {save_path}")

    @staticmethod
    def load(npz_path, transform=None):
        """Load numpy arrays and wrap in Dataset with optional transform."""
        data = np.load(npz_path)
        train = NumpyDataset(data['train_imgs'], data['train_labels'], transform=transform)
        val = NumpyDataset(data['val_imgs'], data['val_labels'], transform=transform)
        test = NumpyDataset(data['test_imgs'], data['test_labels'], transform=transform)
        return train, val, test