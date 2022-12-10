from importlib import import_module
from torch.utils.data import DataLoader, distributed
from argparse import Namespace
from par import DistributedManager
from utils.print import print_pretty
from time import time
import multiprocessing as mp

class Data:
    def __init__(self, 
                 train_dataset_name: str, 
                 test_dataset_name: str,
                 test_only: bool,
                 batch_size: int,
                 number_of_threads: int,
                 is_cpu: bool,
                 namespace: Namespace,
                 distributed_manager: DistributedManager
                 ):
        """
        Args:
            test_only: Do we only want to test the data (and not train it)?
            is_cpu: Are we running this on CPUs?
        """
        print_pretty(f'Data Manager: Creating for train and test: {train_dataset_name} & {test_dataset_name}')
        self.data_train: str = train_dataset_name
        self.data_test: str = test_dataset_name
        self.args: Namespace = namespace

        # If we are only doing tests
        if test_only:
            print_pretty("Data Manager: Skipping train dataset load because user requested test only")
            # Don't load the training loader
            self.loader_train = None
        # Otherwise, if we are doing training
        else:
            # Load training dataset
            self.loader_train = self._create_loader(is_train=True,
                dataset_name=self.data_train,
                batch_size=batch_size,
                is_cpu=is_cpu,
                number_of_threads=number_of_threads,
                namespace=self.args,
                distributed_manager=distributed_manager)
        
            
        # Load testing dataset
        self.loader_test = self._create_loader(is_train=False,
            dataset_name=self.data_test,
            batch_size=1,
            is_cpu=is_cpu,
            number_of_threads=number_of_threads,
            namespace=self.args,
            distributed_manager=distributed_manager
            )

    def _create_loader(self,
                       is_train: bool,
                       dataset_name: str,
                       is_cpu: bool,
                       number_of_threads: int,
                       batch_size: int,
                       namespace: Namespace,
                       distributed_manager: DistributedManager
                       ) -> DataLoader:
        data_type = "training" if is_train else "testing"
        print_pretty(f'Data Manager: Loading the {data_type} dataset with name {dataset_name} batch size: {batch_size}, number of workers: {number_of_threads} and pin memory: {not is_cpu}')

        module = import_module('data.' + dataset_name.lower())
        dataset = getattr(module, dataset_name.upper())(namespace, name=dataset_name, train=is_train, distributed_manager=distributed_manager)
        
        
        # If it is distributed and we are training, we need to load the data in a distributed fashion
        if is_train and distributed_manager.is_distributed:
            print_pretty("Data Manager: Creating distributed data loader for train\n")
            print_pretty(f"Total Numberof Gpus in sampler:{distributed_manager.total_gpus}\n")
            # Ensures that each process gets differnt data from the batch.
            sampler = distributed.DistributedSampler(dataset, 
                                                     num_replicas=distributed_manager.total_gpus, 
                                                     rank=distributed_manager.global_rank,
                                                     drop_last=False)
            
            batch_size_per_gpu = int(batch_size/distributed_manager.total_gpus)
            print_pretty(f"Data Manager: Original batch size was {batch_size} and this GPU will load {batch_size_per_gpu}")
            print_pretty(f"World_size: {distributed_manager.total_gpus}\n")
            # dload_test(dataset) 
            return DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=64,
                shuffle=False,
                pin_memory=True,
                num_workers=36
            )
            
            
        # Otherwise, just load the data like normal
        else:
            print_pretty(f"Data Manager: Creating non distributed data loader for {'train' if is_train else 'test'}")
            return DataLoader(
                dataset=dataset,
                batch_size=64,
                shuffle=is_train,
                pin_memory=not is_cpu,
                num_workers=36
            )





'''
def dload_test(dataset):
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(dataset,shuffle=False,num_workers=num_workers,batch_size=64,pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print_pretty("Finish with:{} second, num_workers={}\n".format(end - start, num_workers))
'''
