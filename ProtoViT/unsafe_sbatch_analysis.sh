#!/bin/bash

#SBATCH --job-name=protovit
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --partition=short
#SBATCH --time=12:00:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate protovit

date

# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_deit_small_patch16_224_cars_0 --data in_distribution_test
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_deit_small_patch16_224_cars_0 --data out_of_distribution_birds
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_deit_small_patch16_224_cars_0 --data cars_test
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_deit_small_patch16_224_out_of_distribution_birds_0 --data in_distribution_test
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_deit_small_patch16_224_out_of_distribution_birds_0 --data out_of_distribution_birds
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_deit_small_patch16_224_out_of_distribution_birds_0 --data cars_test
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_cait_xxs24_224_cars_0 --data in_distribution_test
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_cait_xxs24_224_cars_0 --data out_of_distribution_birds
# python unsafe_global_analysis.py --num_workers=4 --run_name attack_birds_cait_xxs24_224_cars_0 --data cars_test

# python unsafe_local_analysis.py --run_name attack_birds_deit_small_patch16_224_cars_0
# python unsafe_local_analysis.py --run_name attack_birds_deit_small_patch16_224_out_of_distribution_birds_0
# python unsafe_local_analysis.py --run_name attack_birds_cait_xxs24_224_cars_0

date