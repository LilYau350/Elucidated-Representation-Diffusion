import os
import math
import random
import torch
from PIL import Image, PngImagePlugin, ImageFile
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.datasets as datasets
import torch.distributed as dist
import h5py
from tools.dist_util import is_main_process, dist_barrier


# Helper functions for cropping
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L146
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # Downsample if necessary
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# Latent HDF5 Dataset
class Latent(Dataset):
    def __init__(self, h5_file, dataset_type="train"):
        super().__init__()
        self.h5_file = h5_file
        self.dataset_type = dataset_type

        # Open the file to determine the length
        with h5py.File(self.h5_file, 'r') as f:
            self.num_samples = len(f[f'{self.dataset_type}_latents'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            latent = f[f'{self.dataset_type}_latents'][idx]
            label = f[f'{self.dataset_type}_labels'][idx]
        latent = torch.tensor(latent.copy(), dtype=torch.float32)
        label = torch.tensor(label.copy(), dtype=torch.long)
        return latent, label


class LatentWithPixelDataset(Dataset):
    def __init__(self, h5_file, dataset_type="train"):
        super().__init__()
        self.h5_file = h5_file
        self.dataset_type = dataset_type

        # Open the file to determine the length
        with h5py.File(self.h5_file, 'r') as f:
            self.num_samples = len(f[f'{self.dataset_type}_latents'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            latent = f[f'{self.dataset_type}_latents'][idx]
            pixel = f[f'{self.dataset_type}_pixels'][idx]
            label = f[f'{self.dataset_type}_labels'][idx]

        latent = torch.tensor(latent.copy(), dtype=torch.float32)
        pixel = torch.tensor(pixel.copy(), dtype=torch.float32)
        label = torch.tensor(label.copy(), dtype=torch.long)

        return latent, pixel, label
    

# CIFAR10 Dataset
def load_cifar10(data_dir, image_size, random_crop, random_flip,):
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img if img.size == (image_size, image_size) else (
            random_crop_arr(img, image_size) if random_crop else center_crop_arr(img, image_size))
        ),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if is_main_process():
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        
    dist_barrier()
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)    
    
    return train_dataset, test_dataset

# CelebA Dataset Loader
def load_celebA(data_dir, image_size, random_crop, random_flip,):
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img if img.size == (image_size, image_size) else (
            random_crop_arr(img, image_size) if random_crop else center_crop_arr(img, image_size))
        ),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    
    return train_dataset, val_dataset

# ImageNet Dataset Loader
def load_imagenet(data_dir, image_size, random_crop, random_flip,):
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img if img.size == (image_size, image_size) else (
            random_crop_arr(img, image_size) if random_crop else center_crop_arr(img, image_size))
        ),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    return train_dataset, val_dataset

# HDF5Latent Loader
def load_latent(data_dir):
    h5_file = os.path.join(data_dir)
    train_dataset = Latent(h5_file=h5_file, dataset_type='train', )
    val_dataset = Latent(h5_file=h5_file, dataset_type='val')
    
    return train_dataset, val_dataset

def load_latent_pixel(data_dir):
    h5_file = os.path.join(data_dir)
    train_dataset = LatentWithPixelDataset(h5_file, dataset_type="train")
    val_dataset = LatentWithPixelDataset(h5_file,  dataset_type="val")
    return train_dataset, val_dataset

# LSUN Dataset Loader
def load_lsun(data_dir, image_size, random_crop, random_flip,):
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img if img.size == (image_size, image_size) else (
            random_crop_arr(img, image_size) if random_crop else center_crop_arr(img, image_size))
        ),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    return train_dataset, val_dataset

# Unified Dataset Loader
def load_dataset(data_dir, dataset_name, batch_size=128, image_size=None, random_crop=False, random_flip=True, num_workers=4, shuffle=True):
    if dataset_name == 'CIFAR-10':
        train_dataset, test_dataset = load_cifar10(data_dir, image_size, random_crop, random_flip,)   
          
    elif dataset_name == 'CelebA':
        train_dataset, test_dataset = load_celebA(data_dir, image_size, random_crop, random_flip,)
    
    elif dataset_name == 'ImageNet':
        train_dataset, test_dataset = load_imagenet(data_dir, image_size, random_crop, random_flip,)
        
    elif dataset_name == 'Latent':
        train_dataset, test_dataset = load_latent(data_dir)

    elif dataset_name == 'Latent_Pixel':
        train_dataset, test_dataset = load_latent_pixel(data_dir)
        
    elif dataset_name == 'LSUN':
        train_dataset, test_dataset = load_lsun(data_dir, image_size, random_crop, random_flip,)
        
    else:
        raise ValueError("Unsupported dataset")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader


