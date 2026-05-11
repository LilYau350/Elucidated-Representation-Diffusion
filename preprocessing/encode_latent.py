import argparse
from diffusers import AutoencoderKL
import torch
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import h5py
from PIL import Image, PngImagePlugin, ImageFile
import math
import random

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2 ** 20)  # 1024MB
PngImagePlugin.MAX_TEXT_MEMORY = 128 * (2 ** 20)  # 128MB

'''
ImageNet.h5
├── train_latents  # Shape: (num_train_samples, latent_dim)
├── train_pixels  # Shape: (num_train_samples, pixel_dim)
├── train_labels   # Shape: (num_train_samples,)
├── val_latents    # Shape: (num_val_samples, latent_dim)
├── val_pixels    # Shape: (num_val_samples, pixel_dim)
└── val_labels     # Shape: (num_val_samples,)
'''

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the reducing_gap
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
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
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the reducing_gap
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
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
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

# Load the AutoencoderKL model
def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").cuda()
    vae.eval()  # Set model to evaluation mode
    return vae

def load_imagenet(input, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load datasets with transformations
    train_dataset = datasets.ImageFolder(root=f"{input}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{input}/val", transform=transform)
    
    # Create DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def compress_batch(images, device, vae):
    images = images.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        latent_dist = vae.encode(images).latent_dist
        latents = torch.cat([latent_dist.mean, latent_dist.std], dim=1)
    return latents

def save_compressed_latents(data_loader, f, dataset_name, device, vae, save_pixels):
    latents_dataset = None  
    labels_dataset = None
    pixels_dataset = None 
    
    for batch_idx, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Compressing {dataset_name}"):
        latents = compress_batch(images, device, vae)
        
        if latents_dataset is None:
            num_latents = len(data_loader.dataset)
            latents_shape = latents.shape[1:]  # (C, H, W)
            pixels_shape = images.shape[1:]          # (C, H, W)
            
            latents_dataset = f.create_dataset(
                f'{dataset_name}_latents', (num_latents, *latents_shape), dtype='float32'
            )

            if save_pixels:
                pixels_dataset = f.create_dataset(
                    f"{dataset_name}_pixels", (num_latents, *pixels_shape), dtype="uint8",  
                )
                
            labels_dataset = f.create_dataset(
                f'{dataset_name}_labels', (num_latents,), dtype='uint16'  
            )
        
        start_idx = batch_idx * data_loader.batch_size
        end_idx = start_idx + latents.size(0)
        
        latents_dataset[start_idx:end_idx] = latents.cpu().numpy()
        if save_pixels:
            # Convert pixels from [-1, 1] to [0, 255] and store as uint8
            pixel = images.cpu().numpy()  # Assuming images are already normalized to [-1,1]
            pixel = np.clip((pixel + 1) * 127.5, 0, 255).astype(np.uint8)  # Convert to [0, 255] and cast to uint8
            pixels_dataset[start_idx:end_idx] = pixel
            
        # Save labels to HDF5         
        labels_dataset[start_idx:end_idx] = labels.cpu().numpy()
        
        # Ensure data is flushed to disk  
        f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder Image Compression/Decompression")
    parser.add_argument("--input", type=str, default="/data/ImageNet/ILSVRC2012", help="Input folder path or latent file")
    parser.add_argument("--output", type=str, default="/data/ImageNet", help="Output folder path")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for processing")
    parser.add_argument("--save_pixels", type=bool, default=False, help="Also save downsampled pixels into HDF5.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model once at the start
    vae = initialize_vae(args)

    # Load train and val datasets
    train_loader, val_loader = load_imagenet(args.input, args.image_size, args.batch_size)

    # Save compressed latents and labels to HDF5
    h5_file = os.path.join(args.output, "ImageNet.h5")
    with h5py.File(h5_file, 'w') as f:
        save_compressed_latents(train_loader, f, "train", device, vae, args.save_pixels)
        save_compressed_latents(val_loader, f, "val", device, vae, args.save_pixels)


