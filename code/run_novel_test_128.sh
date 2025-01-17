#!/bin/bash
#SBATCH --job-name=test_128
#SBATCH -p gpu-a100
#SBATCH --time=24:00:00

#SBATCH -o /scratch/08310/rs821505/train_outputs/test_novel_128.o%j
#SBATCH -e /scratch/08310/rs821505/train_outputs/test_novel_128.e%j

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

### init virtual environment if needed
module add gcc

### the command to run
python inference.py --quick_test REVIDE_128 --base_on_prior_frame
