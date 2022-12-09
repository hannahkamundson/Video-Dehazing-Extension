from argparse import Namespace
import os

def add_slurm_env_vars(args: Namespace) -> Namespace:
    """
    Add to the namespace by pulling in SLURM environment variables. This
    will fill out all info for the distributed manager.

    Args:
        args (Namespace): Namespace before it has slurm env vars
    Returns:
        Namespace: Namespace with the new params
    """
    args.global_rank = int(os.environ['SLURM_PROCID'])
    args.local_rank = int(os.environ['SLURM_LOCALID'])
    args.is_distributed = True
    args.gpu_per_node = int(os.environ['SLURM_GPUS_PER_NODE'])
    args.number_of_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])

    return args