import os
from importlib import import_module
from logger.data_dirs import DataDirectory
from logger.logger import Logger
from utils.data_utils import get_device
from argparse import Namespace
from par import DistributedManager

import torch
import torch.nn as nn
from utils.print import print_pretty


# Model
class Model(nn.Module):
    def __init__(self, 
                 is_cpu: bool,
                 number_gpus: int,
                 save_middle_models: bool,
                 model_type: str,
                 resume_previous_run: bool,
                 auto_pre_train: bool,
                 pre_train_path: str,
                 test_only: bool,
                 ckp: Logger, 
                 dirs: DataDirectory,
                 args: Namespace,
                 distributed_manager: DistributedManager
                 ):
        super(Model, self).__init__()
        print_pretty('Making model...')
        self.cpu = is_cpu
        self.device = get_device(is_cpu)
        self.n_GPUs = number_gpus
        self.save_middle_models = save_middle_models
        self.dirs = dirs
        self.is_distributed = distributed_manager.is_distributed
        self.write_files = distributed_manager.is_parent_gpu() or not distributed_manager.is_distributed

        module = import_module('model.' + model_type.lower())
        self.model = module.make_model(args, dirs).to(self.device)
        
        # If we want to parallelize this
        if distributed_manager.is_distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            self.model.cuda(distributed_manager.local_rank)
            self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                             device_ids=[distributed_manager.local_rank])
            
        # If we aren't using a CPU and we want more than one GPU
        # Keeping this so it is backwards compatible but it is faster to use DistributedDataParallel
        elif self._is_old_parallel_execution():
            print_pretty("Model: Warning, you are using DataParallel, but it is faster to use DistributedDataParallel")
            self.model = nn.DataParallel(self.model, range(number_gpus))

        # Optionally load any pretrained/priorly trained models
        self.load(
            auto_load=auto_pre_train,
            pre_train=pre_train_path,
            resume=resume_previous_run,
            test_only=test_only,
            cpu=is_cpu,
        )
        
        if self.write_files:
            print_pretty("Model: checkpoint log file", self.get_model(), file=ckp.log_file)
        
    def _is_old_parallel_execution(self) -> bool:
        return not self.cpu and self.n_GPUs > 1

    def forward(self, *args):
        """
        Forward the model to the actual model we are wrapping
        """
        return self.model(*args)

    def get_model(self):
        # If we are using one of our parallelization tehcniques, we need to return the internal module
        if self._is_old_parallel_execution() or self.is_distributed:
            return self.model.module
        # Otherwise, just return the model
        else:
            return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, epoch, is_best=False):
        target = self.get_model()
        self.dirs.save_torch_model('model_latest.pt', target.state_dict())
        if is_best:
            self.dirs.save_torch_model('model_best.pt', target.state_dict())
        if self.save_middle_models:
            if epoch % 1 == 0:
                self.dirs.save_torch_model('model_{}.pt'.format(epoch), target.state_dict())

    def save_now_model(self, flag):
        if self.write_files:
            target = self.get_model()
            self.dirs.save_torch_model(file_name='model_{}.pt'.format(flag),
                                contents=target.state_dict())
        
    def save_model_with_name(self, name: str):
        target = self.get_model()
        self.dirs.save_torch_model(file_name=name,
                             contents=target.state_dict())

    def load(self, auto_load: bool, test_only: bool, pre_train='.', resume=False, cpu=False): 
        """
        Load a previous model if needed.

        Args:
            auto_load (bool): Should we be auto loading a pre dehaze based on the timestamp?
            test_only (bool): Are we only doing testing?
            pre_train (str, optional): Do we want to load a pre trained model from a specific path?
            resume (bool, optional): Are we resuming a previous run? Defaults to False.
            cpu (bool, optional): Is this running on a cpu? Defaults to False.
        """
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        # If we should load the pre trained model from the default directory
        if auto_load:
            print_pretty('Auto loading model from {}'.format(self.dirs.pre_dehaze_model_path()))
            self.get_model().load_state_dict(
                self.dirs.load_torch_from_pre_dehaze(**kwargs), strict=False
            )
        # If we should load the pre trained model from a specific spot
        elif pre_train != '.':
            print_pretty('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs), strict=False
            )
        elif resume:
            print_pretty('Loading model from {}'.format(os.path.join('model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(self.dirs.get_path(os.path.join('model', 'model_latest.pt')), **kwargs),
                strict=False
            )
        elif test_only:
            self.get_model().load_state_dict(
                torch.load(self.dirs.get_path(os.path.join('model', 'model_best.pt')), **kwargs),
                strict=False
            )
        else:
            pass
