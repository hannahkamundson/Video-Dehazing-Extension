import os
import torch
import glob
import numpy as np
import imageio.v2 as imageio
import cv2
import math
import time
import argparse
from model.dehaze_sgid_pff import DEHAZE_SGID_PFF
from utils.print import print_pretty
import datetime


class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.data_path = args.data_path
        self.gt_path = args.gt_path
        self.model_path = args.model_path
        self.result_path = args.result_path
        self.size_must_mode = args.size_must_mode
        self.device = args.device
        # Do we want to base our mdoel off the previous frame?
        self.base_on_prior_frame = args.base_on_prior_frame

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        infer_flag = args.infer_flag if args.infer_flag != '.' else time_str
        self.result_path = os.path.join(self.result_path, 'infer_{}'.format(infer_flag))
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
            print('mkdir: {}'.format(self.result_path))
        self.result_img_path = os.path.join(self.result_path, 'inference_image')
        if not os.path.exists(self.result_img_path):
            os.mkdir(self.result_img_path)
            print('mkdir: {}'.format(self.result_img_path))
        self.logger = Traverse_Logger(self.result_path, 'inference_log.txt')

        self.logger.write_log('Inference - {}'.format(infer_flag))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('gt_path: {}'.format(self.gt_path))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = DEHAZE_SGID_PFF(img_channels=3, 
                                   t_channels=1, 
                                   n_resblock=3, 
                                   n_feat=32, 
                                   device=self.device,
                                   auto_load_pretrained=False,
                                   dirs=None)
        self.net.load_state_dict(torch.load(self.model_path), strict=False)
        self.net = self.net.to(self.device)
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def infer(self):
        with torch.no_grad():
            images_psnr = []
            images_ssim = []
            
            # Get the hazy images
            input_images: list[str] = []

            for folder_path in glob.glob(os.path.join(self.data_path, '*')):
                input_images.extend(glob.glob(os.path.join(folder_path, '*')))
                
            input_images = sorted(input_images)
            
            # Get ground truth images
            gt_images: list[str] = []

            for folder_path in glob.glob(os.path.join(self.gt_path, '*')):
                gt_images.extend(glob.glob(os.path.join(folder_path, '*')))
                
            gt_images = sorted(gt_images)
            
            # We need to store the prior image and the prior video name
            prior_image = None
            prior_video_name = None
            videos_seen_so_far = {}

            for in_im, gt_im in sorted(zip(input_images, gt_images), key=lambda x: x[1]):
                start_time = time.time()
                
                # Make sure we are calling the same path
                assert os.path.basename(gt_im) == os.path.basename(in_im), f'The frames need to be the same: clear {gt_im} hazy {in_im}'

                # Since the file loading is more complicated with video, ensure they are coming from the same video
                ground_truth_video = os.path.basename(os.path.dirname(gt_im))
                input_video = os.path.basename(os.path.dirname(in_im))
                assert ground_truth_video == input_video, f'The videos need to be the same: clear {gt_im} hazy {in_im}'
                
                # If we are switching to a new video
                if self.base_on_prior_frame and input_video != prior_video_name:
                    # Assert that we aren't switching between videos. If a video is being used, all the frames need to come in afterwards
                    assert input_video not in videos_seen_so_far, "The frames in videos need to be in order"
                    videos_seen_so_far.add(input_video)
                
                # Start dealing with the file name
                filename = os.path.basename(in_im).split('.')[0]
                inputs = imageio.imread(in_im)
                gt = imageio.imread(gt_im)

                h, w, _ = inputs.shape
                new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                inputs = inputs[:new_h, :new_w, :]
                gt = gt[:new_h, :new_w, :]
                
                in_tensor = self.numpy2tensor(inputs).to(self.device)
                preprocess_time = time.time()
                # If we want to use the old technique or if we are using the new technique and have come to a new video, we 
                # need to create a new reference image
                if not self.base_on_prior_frame or input_video != prior_video_name:
                    _, output, _, _, _ = self.net(in_tensor, None)
                # Otherwise, pass in the prior frame
                else:
                    # Read the prior frame in and get it into tensor
                    prior_img = imageio.imread(prior_image)
                    prior_img = prior_img[:new_h, :new_w, :]
                    prior_tensor = self.numpy2tensor(prior_img).to(self.device)
                    _, output, _, _, _ = self.net(in_tensor, prior_tensor)
                    
                forward_time = time.time()
                output_img = self.tensor2numpy(output)

                psnr, ssim = self.calc_PSNR_SSIM(output_img, gt)
                images_psnr.append(psnr)
                images_ssim.append(ssim)

                if self.save_image:
                    imageio.imwrite(os.path.join(self.result_img_path, input_video, '{}.jpg'.format(filename)), output_img)
                postprocess_time = time.time()

                self.logger.write_log(
                    '> {} PSNR={:.3f}, SSIM={:.4f} pre_time:{:.3f}s, forward_time:{:.3f}s, post_time:{:.3f}s, total_time:{:.3f}s'
                        .format(filename, psnr, ssim,
                                preprocess_time - start_time,
                                forward_time - preprocess_time,
                                postprocess_time - forward_time,
                                postprocess_time - start_time))
                
                prior_video_name = ground_truth_video
                prior_image = os.path.join(self.result_img_path, input_video, '{}.jpg'.format(filename))

            self.logger.write_log("# Total AVG-PSNR={:.3f}, AVG-SSIM={:.4f}".format(
                sum(images_psnr) / len(images_psnr),
                sum(images_ssim) / len(images_ssim)
            ))

    def numpy2tensor(self, input, rgb_range=1.):
        img = np.array(input).astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
        tensor = tensor.unsqueeze(0)
        return tensor

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def calc_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.PSNR(cropped_GT, cropped_output)
        ssim = self.SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image-Dehaze-Inference')
    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--data_path', type=str, default='../dataset/test', help='the path of test data')
    parser.add_argument('--gt_path', type=str, default='../dataset/gt', help='the path of gt data')
    parser.add_argument('--model_path', type=str, default='../pretrained_models/model.pt', help='the path of model')
    parser.add_argument('--result_path', type=str, default='../infer_results', help='the path of result')
    parser.add_argument('--infer_flag', type=str, default='infer', help='the flag of inference')
    parser.add_argument('--size_must_mode', type=int, default=4, help='the size of input must mode')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick_test', type=str, default='.', help='SOTS_indoor/SOTS_outdoor')
    parser.add_argument('--base_on_prior_frame', action='store_true', default=False, help='Do you want to use the previous frame instead of a reference image?')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    if args.quick_test == 'SOTS_indoor':
        args.data_path = '../dataset/SOTS/indoor/hazy'
        args.gt_path = '../dataset/SOTS/indoor/clear'
        args.model_path = '../pretrained_models/SOTS_indoor.pt'
        args.infer_flag = 'SOTS_indoor'
    elif args.quick_test == 'SOTS_outdoor':
        args.data_path = '../dataset/SOTS/outdoor/hazy'
        args.gt_path = '../dataset/SOTS/outdoor/clear'
        args.model_path = '../pretrained_models/SOTS_outdoor.pt'
        args.infer_flag = 'SOTS_outdoor'
    elif args.quick_test == 'REVIDE_64':
        args.data_path = '/scratch/08310/rs821505/Video-Dehazing-Extension/dataset/REVIDE_REDUCED/Test/hazy'
        args.gt_path = '/scratch/08310/rs821505/Video-Dehazing-Extension/dataset/REVIDE_REDUCED/Test/gt'
        args.model_path = '/scratch/08310/rs821505/Video-Dehazing-Extension/experiment/REVIDE_REDUCED/20221210_19.51/Dehaze/model/model_best.pt'
        args.infer_flag = 'SOTS_outdoor'
        args.save_image = True
        args.infer_flag = datetime.datetime.now().strftime('%Y%m%d_%H.%M')
        args.result_path = '../infer_results/REVIDE_64'
    elif args.quick_test == 'REVIDE_128':
        args.data_path = '/scratch/08310/rs821505/Video-Dehazing-Extension/dataset/REVIDE_REDUCED128/Test/hazy'
        args.gt_path = '/scratch/08310/rs821505/Video-Dehazing-Extension/dataset/REVIDE_REDUCED128/Test/gt'
        args.model_path = '/scratch/08310/rs821505/Video-Dehazing-Extension/experiment/REVIDE_REDUCED/20221210_21.58/Dehaze/model/model_best.pt'
        args.infer_flag = datetime.datetime.now().strftime('%Y%m%d_%H.%M')
        args.save_image = True
        args.result_path = '../infer_results/REVIDE_128'

    Infer = Inference(args)
    Infer.infer()
