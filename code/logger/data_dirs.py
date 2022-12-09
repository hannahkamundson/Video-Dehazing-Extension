import datetime
import os
from argparse import Namespace
import torch
import imageio

PRE_DEHAZE_FOLDER_NAME = 'Pre_Dehaze'
DEHAZE_FOLDER_NAME = 'Dehaze'

class DataDirectory:
    """
    This class holds info about our data directory process.
    If you want to load something from torch, do it through this class
    """
    def __init__(self, args: Namespace, should_write: bool):
        """_summary_

        Args:
            should_write (bool): Is it okay for this data directory to write out? For example, only one GPU should write the folders
                if it is multi node multi GPU
        """
        self.should_write = should_write
        self.base_directory = self._create_base_directory(args)
        print(f"Dirs: Creating data directory {self.base_directory}")
        # I don't fully understand what this is doing, but I think it is saying if we need
        # to load the path, make sure the loading path exists. Otherwise, set it to not load
        # It isn't clear to me where this is used later on but I haven't put a ton of energy 
        # into searching.
        if not args.load == '.' and not os.path.exists(self.base_directory):
            args.load = '.'      

    def _create_base_directory(self, args: Namespace) -> str:
        """
        Create the base directory we will be using to store everything.
        """
        # If we aren't supposed to be loading anything
        if args.load == '.':
            if args.save == '.':
                args.save = 'UNSPECIFIED'
            return self._create_folder_structure(experiment_dir=args.experiment_dir, 
                                                dataset_name=args.save, 
                                                template=args.template,
                                                date_time= args.prev_timestamp,
                                                auto_pre_train=args.auto_pre_train)
        # If we are loading a previous start and continuing from there
        else:
            return args.experiment_dir + args.load + datetime.datetime.now().strftime('%Y%m%d_%H.%M')
        
    def _create_folder_structure(self, experiment_dir: str, dataset_name: str, template: str, date_time: str, auto_pre_train: bool) -> str:
        """
        Return the folder structure to save stuff in.
        
        I want the path to look like this
        | dataset (eg revide or revide_reduced)
            | datetime
                | Pre_Dehaze
                | Dehaze 
        """
        trainer_type = PRE_DEHAZE_FOLDER_NAME if template.startswith('Pre_Dehaze') else DEHAZE_FOLDER_NAME
        
        if date_time is not None:
            # If the date_time was passed in, make sure it exists
            if auto_pre_train and not os.path.exists(os.path.join(experiment_dir, dataset_name, date_time)):
                raise ValueError(f'The timestamp you specified does not exist but you are trying to auto load the pre trained model which requires it to exist {date_time}')
            
            # Make sure it doesn't already have the trainer type
            if os.path.exists(os.path.join(experiment_dir, dataset_name, date_time, trainer_type)):
                raise ValueError(f"The timestamp you specified already has the given template. Please provide a new" \
                                f"timestamp or copy over the other so we can keep track of what we are doing. {date_time} {template}")
            
        # If we are creating a new run, create the date time. Otherwise, use a previous one
        # For example, maybe we are running a Dehaze after a PreDehaze and want to store it in the same place                                 
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H.%M') if date_time is None else date_time

        return os.path.join(experiment_dir, dataset_name, timestamp, trainer_type)

    def create_directory_if_not_exists(self, folder: str):
        """
        If a path doesn't exist, make it

        Args:
            folder (str): The folder you want to add to the base path
        """
        if self.should_write:
            dir_with_folder: str = os.path.join(self.base_directory, folder)
            self._make_path(dir_with_folder)
            
    def make_base_directory(self):
        """
        Make the base directory if it doesn't already exist.
        This should probably only be called once
        """
        self._make_path(self.base_directory)
    
    def save_torch(self, file_name: str, contents, folder_path = None):
        """
        Save something from torch
        """
        if folder_path is None:
            torch.save(contents, os.path.join(self.base_directory, file_name))
        else:
            torch.save(contents, os.path.join(self.base_directory, folder_path, file_name))
            
    def save_torch_model(self, file_name: str, contents):
        self.save_torch(file_name, contents, folder_path='model')
    
    def get_path(self, folder_name: str) -> str:
        """
        Get the absolute path in terms of the base directory

        Args:
            folder_name (str): The folder/file/etc that is extended from the base 
            path

        Returns:
            str:  The absolute path
        """
        return os.path.abspath(os.path.join(self.base_directory, folder_name))
    
    def path_exists(self, name: str) -> bool:
        """
        Does the path exist?

        Args:
            name (str): the extension to the base directory
        Returns:
            bool: Does it exist?
        """
        return os.path.exists(os.path.join(self.base_directory, name))
    
    def get_absolute_base_path(self) -> str:
        """
        Get the absolute path of the base directory

        Returns:
            _type_: Base directory path where everything is stored
        """
        return os.path.abspath(self.base_directory)
    
    def imageio_write(self, folder, file_name, contents):
        """
        Write file to a folder with a specific name
        """
        imageio.imwrite(contents, os.path.join(self.base_directory, folder, file_name))
        
    def load_torch(self, file_name: str):
        """
        Load torch in terms of the base directory.

        Args:
            file_name (str): The file you want to load

        Returns:
            _type_: Whatever it is that Torch returns
        """
        return torch.load(os.path.join(self.base_directory, file_name))
    
    def pre_dehaze_model_path(self) -> str:
        timestamp_dir = os.path.dirname(self.base_directory)
        pre_dehaze = os.path.join(timestamp_dir, PRE_DEHAZE_FOLDER_NAME)
        return os.path.join(pre_dehaze, 'model', 'model_best.pt')
    
    def load_torch_from_pre_dehaze(self, **kwargs):
        model_file = self.pre_dehaze_model_path()
        
        return torch.load(model_file, **kwargs)
    
    def _make_path(self, path: str):
        if self.should_write:
            if not os.path.exists(path):
                print(f"Data Dirs: making {path}")
                os.makedirs(path)
        else:
            print(f"Data Dirs: skipping {path}")
        
        # Make everything wait here so we aren't moving forward before we should
        # torch.distributed.barrier()