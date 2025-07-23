#!/bin/bash

#SBATCH --job-name=protovit
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --partition=long
#SBATCH --time=04-00:00:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate protovit

date

# python unsafe_train_birds.py --num_workers=4
# python unsafe_train_birds.py --num_workers=4 --backbone_architecture $1 --random_seed $2

# python unsafe_finetune_birds.py --num_workers=4
# python unsafe_finetune_birds.py --num_workers=4 --backbone_architecture $1 --prototype_distribution $2 --random_seed $3

# python unsafe_attack_birds.py --num_workers=4
# python unsafe_attack_birds.py --num_workers=4 --backbone_architecture $1 --prototype_distribution $2 --random_seed $3

date
