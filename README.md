## 🛠️ Requirements and Installation
You can refer to the following instructions to configure the environment required for the experiments. 

You can create a new conda environment:

```
conda env create -f environment.yml
conda activate diffusion
```

or install the necessary package by:

```
pip install -r requirements.txt
```
## 🗂️ Data Preparation
#### CelebA Dataset
We follow [ScoreSDE](https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/datasets.py#L121) and [FairGen](https://github.com/ermongroup/fairgen/blob/c5159789eb26699de26a4c306e6862ae3eb3cf39/src/preprocess_celeba.py#L41) for data processing.

1. **Download CelebA**: Download the dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory. Select `Align&Cropped Images` and download `Img/img_align_celeba.zip`, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt`. Unzip `Img/img_align_celeba.zip` into `data/`.

2. **Preprocess CelebA**:
   ``` bash
   python ./preprocessing/preprocess_celeba.py --data_dir=/path/to/data/ --out_dir=./CelebA --partition=train
   ```
   Run this script for `--partition=[train, val, test]` to cache all necessary data. The preprocessed files will be saved in `CelebA/`.

#### ImageNet Dataset
For ImageNet, download the dataset from the [official website](https://image-net.org/download-images). We provide both online and offline preprocessing:
##### ImageNet-64
- **Online Processing**: The functionality for online processing is integrated into [data_loader.py](/datasets/data_loader.py).
- **Offline Preprocessing for ImageNet-64**: Use the script at [image_resizer_imagenet.py](/preprocessing/image_resizer_imagenet.py).
``` bash
python ./preprocessing/image_resizer_imagenet.py -i /path/to/imagenet -o /path/to/output --size 64 -r
```

We refer to the methods described in [this paper](https://arxiv.org/abs/1707.08819) and use code from [PatrykChrabaszcz/resize](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/image_resizer_imagent.py) and [openai/guided diffusion](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/image_datasets.py#L126). We use BOX and BICUBIC methods to ensure high-quality resizing.

##### ImageNet-256
For ImageNet-256, we crop images to 256x256 and compress them using AutoencoderKL from [Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py). We provide a preprocessing script at [encode.py](./preprocessing/encode.py). 
``` bash
python ./preprocessing/encode_latent.py --input /path/to/imagenet --output /path/to/output --batch_size 32 --image_size 256 
```
During compression, we use a scale factor of 0.18215 to stabilize diffusion model training, and similarly, divide by 0.18215 during decompression. This follows practices from [LDM](https://github.com/CompVis/latent-diffusion) and [DiT](https://github.com/huggingface/diffusers/issues/437#issuecomment-1356945792).

The compressed latent codes are treated as images, except for their file extension.

## 📑 Reference Statistics
To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics are calculated using batches of samples stored in `.npz` (numpy) files.

#### Download Batches
OpenAI provides pre-computed sample batches for reference datasets and several baselines, all stored in `.npz` format. For links to download all the sample and reference batches, refer to [evaluations/README.md](./evaluations/README.md).

#### Compute Your Own Statistics
We also provide a script at [cal_ref_stats.py](./preprocessing/cal_ref_stats.py) to calculate `sigma` and `mu` for custom datasets, which are used for FID and Inception Score calculations. You can use this script to compute reference statistics on your own dataset. 
``` bash
python ./preprocessing/cal_ref_stats.py --base_path=/path/to/dataset --dataset_type=train --dataset_name=my_dataset --batch_size=32 --image_size=64
```
Additionally, we provide pre-calculated `.npz` files for CIFAR-10 and CelebA, which include `mu` and `sigma` values.

## 🚀 Train Model
To train our diffusion models, we provide a command script located at [run.sh](./run.sh). You can use this script to reproduce our results. The script includes various experiment commands with the necessary parameters to quickly reproduce the results presented in our paper.

## 💡 Acknowledgements
This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). We use implementations for sampling and FID evaluation from [NVlabs/edm](https://github.com/NVlabs/edm).
