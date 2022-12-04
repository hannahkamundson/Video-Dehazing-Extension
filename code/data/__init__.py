from importlib import import_module
from torch.utils.data import DataLoader
from argparse import Namespace

class Data:
    def __init__(self, 
        train_dataset_name: str, 
        test_dataset_name: str, 
        test_only: bool, 
        batch_size: int,
        number_of_threads: int,
        is_cpu: bool,
        namespace: Namespace):
        """
        Args:
            test_only: Do we only want to test the data (and not train it)?
            is_cpu: Are we running this on CPUs?
        """
        print(f'Data Manager: Creating for train and test: {train_dataset_name} & {test_dataset_name}')
        self.data_train: str = train_dataset_name
        self.data_test: str = test_dataset_name
        self.args=namespace

        # If we are only doing tests
        if test_only:
            print("Data Manager: Skipping train dataset load because user requested test only")
            # Don't load the training loader
            self.loader_train = None
        # Otherwise, if we are doing training
        else:
            # Load training dataset
            self.loader_train = self.create_loader(is_train=True,
                dataset_name=self.data_train,
                batch_size=batch_size,
                is_cpu=is_cpu,
                number_of_threads=number_of_threads,
                args=self.args)
        
            
        # Load testing dataset
        self.loader_test = self.create_loader(is_train=False,
            dataset_name=self.data_test,
            batch_size=1,
            is_cpu=is_cpu,
            number_of_threads=number_of_threads,
            args=self.args
            )

    def create_loader(is_train: bool,
        dataset_name: str,
        is_cpu: bool,
        number_of_threads: int,
        batch_size: int,
        args: Namespace,
        ) -> DataLoader:
        data_type = "training" if is_train else "testing"
        print(f'Data Manager: Loading the {data_type} dataset with batch size: {batch_size}, number of workers: {number_of_threads} and pin memory: {not is_cpu}')

        module = import_module('data.' + dataset_name.lower())
        dataset = getattr(module, dataset_name.upper())(args, name=dataset_name, train=is_train)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            pin_memory=not is_cpu,
            num_workers=number_of_threads
        )
