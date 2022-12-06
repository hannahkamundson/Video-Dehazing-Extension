import os
from data import imagedata


class RESIDE(imagedata.IMAGEDATA):
    def __init__(self, namespace, name='RESIDE', train=True):
        super(RESIDE, self).__init__(args=namespace, 
            name=name, 
            train=train, 
            train_directory=namespace.dir_data,
            test_directory=namespace.dir_data_test,
            clear_folder='clear', 
            hazy_folder='hazy')
