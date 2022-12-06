import os
from importlib import import_module

import torch
import torch.distributed as dist
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
#         if not args.cpu and args.n_GPUs > 1:
#             self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        
        # DDP setting
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1
        ngpus_per_node = torch.cuda.device_count()

        if args.distributed:
            if args.local_rank != -1: # for torch.distributed.launch
                args.rank = args.local_rank
                args.gpu = args.local_rank
            elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
                args.rank = int(os.environ['SLURM_PROCID'])
                args.gpu = args.rank % torch.cuda.device_count()
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

        # suppress printing if not on master gpu
        if args.rank!=0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

        ### model ###
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                self.model.cuda(args.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#                 model_without_ddp = self.model.module
            else:
#                 self.model.cuda()
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
#                 model_without_ddp = self.model.module
        else:
            raise NotImplementedError("Only DistributedDataParallel is supported.")

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
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

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        if self.save_middle_models:
            if epoch % 1 == 0:
                torch.save(
                    target.state_dict(),
                    os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
                )

    def save_now_model(self, apath, flag):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_{}.pt'.format(flag))
        )

    def load(self, apath, pre_train='.', resume=False, cpu=False):  #
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs), strict=False
            )
        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_best.pt'), **kwargs),
                strict=False
            )
        else:
            pass
