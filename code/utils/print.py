import os

# I am sure there is a better way to do this but I don't want to spend too much time learning Python's logging system
def print_pretty(*args, **kwargs):
    global_rank = os.environ['SLURM_PROCID']
    node_id=os.environ['SLURM_NODEID']
    print(f'{node_id}r{global_rank}', *args, **kwargs)