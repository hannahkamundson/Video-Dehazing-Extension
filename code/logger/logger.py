import torch
import numpy as np
import os
from logger.data_dirs import DataDirectory
from par import DistributedManager
import matplotlib
from utils.print import print_pretty

matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Logger:
    def __init__(self, 
                 args, 
                 init_loss_log,
                 dirs: DataDirectory,
                 distributed_manager: DistributedManager
                 ):
        """
        Args:
            args (_type_): _description_
            init_loss_log (_type_): _description_
            dirs (DataDirectory): The data directory that has access to where we are
                storing/saving/loading things from
        """
        self.dirs: DataDirectory = dirs
        self.args = args
        self.psnr_log = torch.Tensor()
        self.loss_log = {}
        for key in init_loss_log.keys():
            self.loss_log[key] = torch.Tensor()
        
        self.dir = dirs.base_directory
        self.write_files = not distributed_manager.is_distributed or distributed_manager.is_parent_gpu()
        
        # I don't fully understand what this is doing, but I think it is saying if we need
        # to load the path, make sure the loading path exists. Otherwise, set it to not load
        # It isn't clear to me where this is used later on but I haven't put a ton of energy 
        # into searching.
        if not args.load == '.' and os.path.exists(dirs.base_directory):
            # Load stuff if we are continuing on from a previous run
            self.loss_log = dirs.load_torch('loss_log.pt')
            self.psnr_log = dirs.load_torch('psnr_log.pt')
            print_pretty('Continue from epoch {}...'.format(len(self.psnr_log)))

        # If the path doesn't exist, make it
        dirs.make_base_directory()
            
        # If the model path doesn't exist, make it
        dirs.create_directory_if_not_exists('model')
            
        # If the results path doesn't exist, make it
        result_folder = os.path.join('result', args.data_test)
        dirs.create_directory_if_not_exists(result_folder)
        print_pretty("Creating dir for saving images...", dirs.get_path(result_folder))
        
        print_pretty('Save Path : {}'.format(dirs.get_absolute_base_path()))

        if self.write_files:
            open_type = 'a' if dirs.path_exists('log.txt') else 'w'
            self.log_file = open(dirs.get_path('log.txt'), open_type)
            with open(dirs.get_path('config.txt'), open_type) as f:
                f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

    def write_log(self, log):
        if self.write_files:
            print_pretty(log)
            self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        if self.write_files:
            trainer.model.save(epoch, is_best)
            # Save Torch stuff
            self.dirs.save_torch('loss_log.pt', self.loss_log)
            self.dirs.save_torch('psnr_log.pt', self.psnr_log)
            self.dirs.save_torch('optimizer.pt', trainer.optimizer.state_dict())
            # self.plot_loss_log(epoch)
            # self.plot_psnr_log(epoch)

    def save_images(self,epoch, filename, save_list):
        #dirname = os.path.join('result', self.args.data_test)
        #self.dirs.create_directory_if_not_exists(dirname)
        #filename = '{}/{}'.format(dirname, filename)
        #if self.args.task == '.':
        #    postfix = ['combine']
        #else:
         #   postfix = ['combine']
        #for img, post in zip(save_list, postfix):
        outs = ['input_','output_','target_']
        for j, img in enumerate(save_list):
            self.dirs.imageio_write(epoch,list(),outs[j]+filename, img)

    def start_log(self, train=True):
        if train:
            for key in self.loss_log.keys():
                self.loss_log[key] = torch.cat((self.loss_log[key], torch.zeros(1)))
        else:
            self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))

    def report_log(self, item, train=True):
        if train:
            for key in item.keys():
                self.loss_log[key][-1] += item[key]
        else:
            self.psnr_log[-1] += item

    def end_log(self, n_div, train=True):
        if train:
            for key in self.loss_log.keys():
                self.loss_log[key][-1].div_(n_div)
        else:
            self.psnr_log[-1].div_(n_div)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for key in self.loss_log.keys():
            log.append('[{}: {:.4f}]'.format(key, self.loss_log[key][-1] / n_samples))
        return ''.join(log)

    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        print_pretty(f'Logger: loss log: {self.loss_log}')
        for key in self.loss_log.keys():
            label = '{} Loss'.format(key)
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.loss_log[key].numpy())
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(self.dirs.get_path('loss_{}.pdf'.format(key)))
            plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(self.dirs.get_path('psnr.pdf'))
        plt.close(fig)

    def done(self):
        if self.write_files:
            self.log_file.close()
