#!/bin/bash

#SBATCH --job-name=pipnet
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --partition=short,long
#SBATCH --time=14:00:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate protovit

date

python unsafe_train.py --default unsafe_arguments/isic2019_default.txt --num_workers 4 --net $1 --seed $2

# python unsafe_train.py --default unsafe_arguments/mura_default.txt --num_workers 4 --net $1 --seed $2

# python unsafe_visualize_topk.py

date