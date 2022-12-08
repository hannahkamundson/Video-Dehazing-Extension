import os
from importlib import import_module
from logger.data_dirs import DataDirectory

import torch
import torch.distributed as dist
import torch.nn as nn


# Model
class Model(nn.Module):
    def __init__(self, args, ckp, dirs: DataDirectory):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models
        self.dirs = dirs

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            auto_load=args.auto_pre_train,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu,
        )
        print(self.get_model(), file=ckp.log_file)

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        if not self.cpu and self.n_GPUs > 1:
            return self.model.module
        else:
            return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, epoch, is_best=False):
        target = self.get_model()
        self.dirs.save_torch_model('model_latest.pt', target.state_dict())
        if is_best:
            self.dirs.save_torch_model('model_best.pt', target.state_dict())
        if self.save_middle_models:
            if epoch % 1 == 0:
                self.dirs.save_torch_model('model_{}.pt'.format(epoch), target.state_dict())

    def save_now_model(self, flag):
        target = self.get_model()
        self.dirs.save_torch_model(file_name='model_{}.pt'.format(flag),
                             contents=target.state_dict())
        
    def save_model_with_name(self, name: str):
        target = self.get_model()
        self.dirs.save_torch_model(file_name=name,
                             contents=target.state_dict())

    def load(self, auto_load, pre_train='.', resume=False, cpu=False):  #
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        # If we should load the pre trained model from the default directory
        if auto_load:
            print('Auto loading model from {}'.format(self.dirs.pre_dehaze_model_path()))
            self.get_model().load_state_dict(
                self.dirs.load_torch_from_pre_dehaze(**kwargs), strict=False
            )
        # If we should load the pre trained model from a specific spot
        elif pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs), strict=False
            )
        elif resume:
            print('Loading model from {}'.format(os.path.join('model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(self.dirs.get_path(os.path.join('model', 'model_latest.pt')), **kwargs),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(self.dirs.get_path(os.path.join('model', 'model_best.pt')), **kwargs),
                strict=False
            )
        else:
            pass
