import os
from data import imagedata
from par import DistributedManager


class RESIDE(imagedata.IMAGEDATA):
    def __init__(self, namespace, distributed_manager: DistributedManager, name='RESIDE', train=True):
        super(RESIDE, self).__init__(name=name, 
            train=train, 
            batch_size=namespace.batch_size,
            load_all_on_ram=namespace.process,
            train_directory=namespace.dir_data,
            test_directory=namespace.dir_data_test,
            test_every_x_batch=namespace.test_every,
            patch_size=namespace.patch_size,
            no_data_augmentation=namespace.no_augment,
            size_must_mode=namespace.size_must_mode,
            max_rgb_value =namespace.rgb_range,
            distributed_manager=distributed_manager,
            clear_folder='clear', 
            hazy_folder='hazy')
