from importlib import import_module
from torch.utils.data import DataLoader


class Data:
    def __init__(self, 
        train_dataset_name: str, 
        test_dataset_name: str, 
        test_only: bool, 
        batch_size: int,
        number_of_threads: int,
        is_cpu: bool):
        """
        Args:
            test_only: Do we only want to test the data (and not train it)?
            is_cpu: Are we running this on CPUs?
        """
        self.data_train = train_dataset_name
        self.data_test = test_dataset_name

        # If we are only doing tests
        if test_only:
            # Don't load the training loader
            self.loader_train = None
        # Otherwise, if we are doing training
        else:
            # Load training dataset 
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train.upper())(self.args, name=self.data_train, train=True)
            self.loader_train = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=not is_cpu,
                num_workers=number_of_threads
            )
        
            
        # load testing dataset
        m_test = import_module('data.' + self.data_test.lower())
        testset = getattr(m_test, self.data_test.upper())(self.args, name=self.data_test, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not is_cpu,
            num_workers=number_of_threads
        )
