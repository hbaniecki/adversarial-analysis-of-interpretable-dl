Modified and extended code based on https://github.com/Henrymachiyu/ProtoViT from Ma et al. *Interpretable image classification with adaptive prototype-based vision transformers*. NeurIPS 2024

See also Chen et al. *This looks like that: Deep learning for interpretable image recognition*. NeurIPS 2019

Added `unsafe_**` files.

### Datasets

#### CUB-200 (aka Birds)

Follow the instructions in https://github.com/Henrymachiyu/ProtoViT?tab=readme-ov-file#cub-200-2011-dataset

Wah et al. *Caltech-UCSD Birds-200-2011*. California Institute of Technology, 2011

#### Out-of-distribution Birds

This is our custom-made set of 2044 images from 13 bird species (classes) coming from the training set of ImageNet-1k.
Crucially, some ImageNet classes overlap with CUB, e.g. 13 -- junco, 16 -- bulbul, 94 -- hummingbird, 144 -- pelican, 146 -- albatross, and thus we collected 200 images from each of the following classes that we did not find in CUB: 7 -- cock, 8 -- hen, 17 -- jay, 18 -- magpie, 19 -- chickadee, 21 -- kite, 22 -- bald eagle, 23 -- vulture, 24 -- great grey owl, 127 -- white stork, 128 -- black stork, 134 -- crane, 145 -- king penguin.
We then manually filtered the resulting 2600 images to remove images of low quality, with objects (birds) occupying a very small portion of an image, or labeling errors.

Files can be openly accessed at https://drive.google.com/file/d/131qDSgjl3bFEJ75ovz9dMvcqsU97U861

#### Stanford Cars

https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars

See also https://github.com/pytorch/vision/issues/7545#issuecomment-2282674373

Krause et al. *3D Object representations for fine-grained categorization*. ICCVW 2013

### Environment setup

```
conda create -n protovit python=3.9
conda activate protovit
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python==4.10.0.84 timm==0.4.12 Augmentor matplotlib numpy==1.26.4 ipykernel datasets scipy
```

### Run experiments

Train, fine-tune, and manipulate models

```bash
# see unsafe_sbatch.sh
python unsafe_run_sbatch.py
```

Analysis

```bash
# see unsafe_sbatch_analysis.sh
sbatch unsafe_sbatch_analysis.sh
```