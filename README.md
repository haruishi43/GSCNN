# Gated-SCNN: Gated Shape CNNs for Semantic Segmentation

Based on based on https://github.com/NVIDIA/semantic-segmentation.

# Setup

Install `encoding`:
```bash
git clone https://github.com/zhanghang1989/PyTorch-Encoding && cd PyTorch-Encoding
python setup.py install
cd ..
```

Setup GSCNN:
```bash
git clone https://github.com/nv-tlabs/GSCNN && cd GSCNN
pip intall -r requirements.txt
```

## Acceleration

- [`accimage`](https://github.com/pytorch/accimage)

## Download pretrained models

Download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1wlhAXg-PfoUM-rFy2cksk43Ng3PpsK2c/view), and save it in 'checkpoints/'

## Download inferred images

Download (if needed) the inferred images from the [Google Drive Folder](https://drive.google.com/file/d/105WYnpSagdlf5-ZlSKWkRVeq-MyKLYOV/view)


# Training

Code:

```bash
# 2x RTX 3090
CUDA_VISIBLE_DEVICES=0,1, python sseg/train.py --lr 0.005 --bs_mult 3 --bs_mult 2
```

# Evaluation (Cityscapes)

The code takes around 7 hours

```bash
CUDA_VISIBLE_DEVICES=0, python sseg/train.py --evaluate --snapshot checkpoints/best_cityscapes_checkpoint.pth
```
