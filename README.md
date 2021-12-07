# Gated-SCNN: Gated Shape CNNs for Semantic Segmentation

Based on based on https://github.com/NVIDIA/semantic-segmentation.

# Setup

```bash
git clone https://github.com/nv-tlabs/GSCNN
cd GSCNN
```

#### Download pretrained models

Download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1wlhAXg-PfoUM-rFy2cksk43Ng3PpsK2c/view), and save it in 'checkpoints/'

#### Download inferred images

Download (if needed) the inferred images from the [Google Drive Folder](https://drive.google.com/file/d/105WYnpSagdlf5-ZlSKWkRVeq-MyKLYOV/view)

#### Evaluation (Cityscapes)

```bash
python train.py --evaluate --snapshot checkpoints/best_cityscapes_checkpoint.pth
```
