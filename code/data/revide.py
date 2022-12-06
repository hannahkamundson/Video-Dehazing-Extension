from data import imagedata
import glob
import os

class REVIDE(imagedata.IMAGEDATA):
    def __init__(self, namespace, name='REVIDE', train=True):
        super(REVIDE, self).__init__(args=namespace, 
            name=name, 
            train=train, 
            train_directory=namespace.dir_data,
            test_directory=namespace.dir_data_test,
            clear_folder='gt', 
            hazy_folder='hazy')

    def _scan(self) -> list[str]:
        """
        This scans in a specific way for the revide data, which is nested in video folders.
        For example:

        | video 1
            | frame 1 image of video 1
            | frame 2 image of video 1
        | video 2
            | frame 1 image of video 2
            | frame 2 image of video 2

        Returns:
            a tuple that has (list of paths for clear/ground truth images, list of paths for hazy/input images)
        """
        clear_image_paths: list[str] = []

        for folder_path in glob.glob(os.path.join(self.dir_gt, '*')):
            clear_image_paths.extend(glob.glob(os.path.join(folder_path, '*')))

        hazy_image_paths: list[str] = []
        for folder_path in glob.glob(os.path.join(self.dir_input, '*')):
            hazy_image_paths.extend(glob.glob(os.path.join(folder_path, '*')))

        assert len(clear_image_paths) == len(hazy_image_paths), f'The number of clear images must match the number of hazy images: clear {len(clear_image_paths)} hazy {len(hazy_image_paths)}'

        return sorted(clear_image_paths), sorted(hazy_image_paths)