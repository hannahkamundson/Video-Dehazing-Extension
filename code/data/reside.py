import os
from data import imagedata


class RESIDE(imagedata.IMAGEDATA):
    def __init__(self, namespace, name='RESIDE', train=True):
        super(RESIDE, self).__init__(args=namespace, name=name, train=train, clear_folder='clear', hazy_folder='hazy')
