#!/bin/bash

#SBATCH --job-name=pipnet
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --partition=short,long
#SBATCH --time=12:00:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate protovit

date

python unsafe_attack.py --default unsafe_arguments/isic2019_attack.txt --num_workers 4 --net $1 --seed $2  --mode $3

# python unsafe_attack.py --default unsafe_arguments/mura_attack.txt --num_workers 4 --net $1 --seed $2  --mode $3

# python unsafe_table.py

date