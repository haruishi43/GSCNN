# Gated-SCNN: Gated Shape CNNs for Semantic Segmentation

Official codes from ICCV2019 is in [github.com/nv-tlabs/GSCNN](https://github.com/nv-tlabs/GSCNN).
I found that I couldn't train the original code due to some bugs and missing parts, so I updated it here.

WideResNet portions are based on https://github.com/NVIDIA/semantic-segmentation.

## Setup

Install `encoding`:
```bash
git clone https://github.com/zhanghang1989/PyTorch-Encoding && cd PyTorch-Encoding
python setup.py install
cd ..
```

Setup GSCNN:
```bash
git clone https://github.com/haruishi43/GSCNN && cd GSCNN
pip intall -r requirements.txt
```

__Other dependencies__:

- [`accimage`](https://github.com/pytorch/accimage)

## Download pretrained models

Download WideResNet38 trained on ImageNet for training: [pretrained_models/wider_resnet38.pth.tar](https://drive.google.com/file/d/1OfKQPQXbXGbWAQJj2R82x6qyz6f-1U6t/view?usp=sharing). This is provided by [NVIDIA/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation).


Download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1wlhAXg-PfoUM-rFy2cksk43Ng3PpsK2c/view), and save it in 'checkpoints/'

## Download inferred images

Download (if needed) the inferred images from the [Google Drive Folder](https://drive.google.com/file/d/105WYnpSagdlf5-ZlSKWkRVeq-MyKLYOV/view)

# Training

Code:

```bash
# 2x RTX 3090
CUDA_VISIBLE_DEVICES=0,1, python scripts/train.py --lr 0.005 --bs_mult 3 --bs_mult 2
```

# Evaluation (Cityscapes)

The code takes around 7 hours

```bash
CUDA_VISIBLE_DEVICES=0, python scripts/train.py --evaluate --snapshot checkpoints/best_cityscapes_checkpoint.pth
# or
CUDA_VISIBLE_DEVICES=0, python scripts/test.py --bs_mult_val 1 --snapshot <path>
```
