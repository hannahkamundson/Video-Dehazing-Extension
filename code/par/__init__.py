import torch.distributed as dist
import torch
from argparse import Namespace

# For more details on how we decided to do different parallelization techinques in Torch:
# https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
# https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#distributeddataparallel-as-a-batch-job-in-the-servers
# http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html
class DistributedManager:
    def __init__(self,
                 is_distributed: bool,
                 local_rank: int,
                 global_rank: int,
                 gpus_per_node: int,
                 total_nodes: int,
                 ):
        """
        This class handles parallelization of torch. It does multi-node, multi-GPU parallelization and keeps 
        our info together so we don't have to think too hard about what we are passing around until we need it.
        Args:
            local_rank (int): The rank of the GPU amongst other GPUs in the node.
            global_rank (int): The rank of the GPU amongst all other GPUs in and outside of the node
            total_gpus (int): The total number of GPUs. We are only allowing one task per GPU
            total_nodes (int): The total number of nodes
        """
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.gpus_per_node = gpus_per_node
        self.total_gpus = gpus_per_node * total_nodes
        self.total_nodes = total_nodes
        
        # If it is distributed, ensure we have everything defined that is needed
        if is_distributed:
            assert self.local_rank is not None, "You must define a local rank (a rank of the GPU amongst other GPUs in the node)."
            assert self.global_rank is not None, "You must define a global rank (a rank compared to all other GPUs)."
            assert self.total_gpus is not None, "You must define the total number of GPUs."
            assert self.total_nodes is not None, "You must define the total number of nodes."
            
            self._initialize_torch()
        
    def _initialize_torch(self):
        """
        Initialize the process groups (like the number of processes, the protocol of communication,
        etc)
        """
        
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=self.total_gpus,
                                rank=self.global_rank)
        
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(self.local_rank)
        
    def is_parent_gpu(self) -> bool:
        return self.global_rank == 0
    
def create(args: Namespace) -> DistributedManager:
    return DistributedManager(is_distributed=args.is_distributed,
                              local_rank=args.local_rank,
                              global_rank=args.global_rank,
                              gpus_per_node=args.gpus_per_node,
                              total_nodes=args.number_of_nodes)