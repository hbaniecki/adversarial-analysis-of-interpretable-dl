Modified and extended code based on https://github.com/M-Nauta/PIPNet from Nauta et al. *PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification*. CVPR 2023

See also Nauta et al. *Interpreting and correcting medical image classification with PIP-Net*. ECAI 2023

Added `unsafe_**` files.

### Datasets

#### 1. ISIC 2019

https://challenge.isic-archive.com/data/#2019

Tschandl et al. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific
Data, 2018

Hernandez-Perez et al. *BCN20000: Dermoscopic lesions in the wild*. Scientific Data, 2024

#### 2. MURA

https://www.kaggle.com/api/v1/datasets/download/cjinny/mura-v11

Rajpurkar et al. *MURA: Large dataset for abnormality detection in musculoskeletal radiographs*. MIDL 2018


### Environment setup

```
conda create -n pipnet python=3.9
conda activate pipnet
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python==4.10.0.84 timm==0.4.12 Augmentor matplotlib numpy==1.26.4 ipykernel datasets scipy scikit-learn
```

### Run experiments

Train models

```bash
# see unsafe_sbatch_train.sh
python unsafe_run_sbatch_train.py
```

Backdoor attack

```bash
# see unsafe_sbatch_attack.sh
python unsafe_run_sbatch_attack.py
``` 

Analysis

```bash
python unsafe_visualize_topk.py

python unsafe_table.py
```