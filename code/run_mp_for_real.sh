#!/bin/bash
#SBATCH --job-name=dehaze
#SBATCH -p gpu-a100
#SBATCH --time=24:00:00

#SBATCH -o /scratch/08310/rs821505/train_outputs/dehaze_64.o%j
#SBATCH -e /scratch/08310/rs821505/train_outputs/dehaze_64.e%j

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=40


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
export MASTER_PORT=12340

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "Parent IP="$MASTER_ADDR
echo "Parent Port="$MASTER_PORT

### init virtual environment if needed
module add gcc

# Give some details on pytorch distributed stuff
export TORCH_DISTRIBUTED_DEBUG=INFO

### the command to run
srun python main.py --slurm_env_var --template Dehaze_revidereduced --prev_timestamp 20221210_19.51 --auto_pre_train
