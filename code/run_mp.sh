#!/bin/bash
#SBATCH --job-name=revide_par
#SBATCH -p gpu-a100
#SBATCH --time=05:00:00

#SBATCH -o /scratch/08310/rs821505/train_outputs/run_mp.o%j
#SBATCH -e /scratch/08310/rs821505/train_outputs/run_mp.e%j

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1


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

### the command to run
srun python main.py --add_slurm_env_vars --template Pre_Dehaze_revidereduced