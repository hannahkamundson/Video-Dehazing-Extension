from data import imagedata
import glob
import os
import imageio
from par import DistributedManager

class REVIDE(imagedata.IMAGEDATA):
    def __init__(self, namespace, distributed_manager: DistributedManager, name='REVIDE', train=True):
        super(REVIDE, self).__init__(name=name, 
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

    def _load_file(self, idx):
        """
        Load the files.
        I'm overriding this just to make sure we are grabbing the same frame from the same video
        """
        idx = self._get_index(idx)
        # Get the paths for the ground truth/clear image and the hazy/input image
        f_gt = self.images_gt[idx]
        f_input = self.images_input[idx]

        assert os.path.basename(f_gt) == os.path.basename(f_input), f'The frames need to be the same: clear {f_gt} hazy {f_input}'

        # Since the file loading is more complicated, ensure they are coming from the same video
        ground_truth_video = os.path.basename(os.path.dirname(f_gt))
        input_video = os.path.basename(os.path.dirname(f_input))
        assert ground_truth_video == input_video, f'The videos need to be the same: clear {f_gt} hazy {f_input}'

        # read the images
        gt = imageio.imread(f_gt)[:, :, :3]
        input = imageio.imread(f_input)[:, :, :3]
        filename, _ = os.path.splitext(os.path.basename(f_gt))
        return input, gt, filename