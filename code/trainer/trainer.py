import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.data_utils import get_device
from par import DistributedManager


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp, distributed_manager: DistributedManager):
        self.args = args
        self.device = get_device(self.args.cpu)
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp
        self.distributed_manager = distributed_manager

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lr_scheduler.StepLR(self.optimizer, **kwargs)

    def train(self):
        pass

    def validate(self):
        # Only validate if it is the parent GPU
        if self.distributed_manager.is_parent_gpu():
            self.do_validate()
    
    def do_validate(self):
        pass
    
    def get_last_started_epoch(self) -> int:
        return self.scheduler.last_epoch

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.get_last_started_epoch()  + 1
            return epoch >= self.args.epochs
        
    def pre_train(self):
        if self.distributed_manager.is_distributed:
            self.loader_train.sampler.set_epoch(self.get_last_started_epoch() + 1)
        
    def step_next(self):
        """
        Step to the next stage after an epoch
        """
        print("Stepping to next epoch")
        if self.args.test_only:
            return
        # If it will terminate on the next one, save the final model
        # As in if this was the final epoch
        elif self.scheduler.last_epoch + 2 >= self.args.epochs:
            self.model.save_model_with_name('model_final.pt')
        
        epoch = self.scheduler.last_epoch + 1
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=False)
            
        # This is where self.scheduler.step() was moved
        self.scheduler.step()
