import os
import glob
import imageio
import utils.data_utils as utils
import torch.utils.data as data


class IMAGEDATA(data.Dataset):
    def __init__(self, 
        name: str, 
        patch_size: int,
        load_all_on_ram: bool,
        test_every_x_batch: int,
        batch_size: int,
        train_directory: str,
        test_directory: str,
        no_data_augmentation: bool,
        size_must_mode: int,
        max_rgb_value: int,
        train=True, 
        clear_folder: str='GT', 
        hazy_folder: str='INPUT'):
        
        self.name = name
        self.train = train
        self.clear_folder = clear_folder
        self.hazy_folder = hazy_folder
        self.patch_size = patch_size
        self.size_must_mode = size_must_mode
        self.max_rgb_value = max_rgb_value
        # Do you want to load all the data at once on RAM?
        self.load_all_on_ram = load_all_on_ram
        # Do we want to use data augmentation?
        self.no_data_augmentation = no_data_augmentation

        # Set the clear and hazy file systems
        # Choose a directory based on whether we are looking at the training dataset or the testing dataset
        if train:
            self._set_filesystem(train_directory)
        else:
            self._set_filesystem(test_directory)

        self.images_gt, self.images_input = self._scan()

        self.num_image = len(self.images_gt)
        print("Number of images to load:", self.num_image)

        if train:
            self.repeat = max(test_every_x_batch // max((self.num_image // batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if self.load_all_on_ram:
            self.data_gt, self.data_input = self._load(self.images_gt, self.images_input)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        # apath is just the path to the dataset. It should contain a "clear "images" folder and a 
        # "hazy images" folder, though they can be named different things depending on the child 
        # class' implementation
        self.apath = dir_data

        # Set the clear path by getting the full path to the dataset + adding the clear folder to the end
        self.dir_gt = os.path.join(self.apath, self.clear_folder)
        self.dir_input = os.path.join(self.apath, self.hazy_folder)
        print(f'DataSet clear/ground truth path:', self.dir_gt)
        print(f'DataSet hazy/input path:', self.dir_input)

    def _scan(self) -> tuple[list[str], list[str]]:
        """
        Get the images for the ground truth and the input.
        In our case, ground truth usually means clear images and input means hazy images.

        Returns:
            a tuple that has (list of paths for clear/ground truth images, list of paths for hazy/input images)
        """
        names_gt = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        names_input = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        assert len(names_gt) == len(names_input), "len(names_gt) must equal len(names_input)"
        return names_gt, names_input

    def _load(self, names_gt, names_input):
        print('Loading image dataset...')
        data_input = [imageio.imread(filename)[:, :, :3] for filename in names_input]
        data_gt = [imageio.imread(filename)[:, :, :3] for filename in names_gt]
        return data_gt, data_input

    def __getitem__(self, idx):
        """
        This is the main method that needs to be implemented for dataset, which supports fetching a data sample for 
        a given key.
        """
        if self.load_all_on_ram:
            input, gt, filename = self._load_file_from_loaded_data(idx)
        else:
            input, gt, filename = self._load_file(idx)

        input, gt = self.get_patch(input, gt, self.size_must_mode)
        input_tensor, gt_tensor = utils.np2Tensor(input, gt, rgb_range=self.max_rgb_value)

        return input_tensor, gt_tensor, filename

    def __len__(self):
        if self.train:
            return self.num_image * self.repeat
        else:
            return self.num_image

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_gt)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_gt = self.images_gt[idx]
        f_input = self.images_input[idx]
        gt = imageio.imread(f_gt)[:, :, :3]
        input = imageio.imread(f_input)[:, :, :3]
        filename, _ = os.path.splitext(os.path.basename(f_gt))
        return input, gt, filename

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)
        gt = self.data_gt[idx]
        input = self.data_input[idx]
        filename = os.path.splitext(os.path.split(self.images_gt[idx])[-1])[0]
        return input, gt, filename

    def get_patch(self, input, gt, size_must_mode=1):
        if self.train:
            input, gt = utils.get_patch(input, gt, patch_size=self.patch_size)
            h, w, _ = input.shape
            if h != self.patch_size or w != self.patch_size:
                input = utils.bicubic_resize(input, size=(self.patch_size, self.patch_size))
                gt = utils.bicubic_resize(gt, size=(self.patch_size, self.patch_size))
                h, w, _ = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.no_data_augmentation:
                input, gt = utils.data_augment(input, gt)
        else:
            h, w, _ = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
        return input, gt
