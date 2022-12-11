import decimal
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer
import torch.optim as optim
from loss import gradient_loss
from utils.data_utils import get_device_type
from par import DistributedManager
from utils.print import print_pretty


class Trainer_Pre_Dehaze(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp, distributed_manager: DistributedManager):
        super(Trainer_Pre_Dehaze, self).__init__(args, loader, my_model, my_loss, ckp, distributed_manager)
        print_pretty("Using Trainer_Pre_Dehaze")
        device = get_device_type(args.cpu)
        self.grad_loss = gradient_loss.Gradient_Loss(device=device)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().parameters()}],
                               **kwargs)
        return optimizer

    def train(self):
        print_pretty("PreDehaze: Now training")

        # This is where self.scheduler.step() was
        
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()
        # self.scheduler.get_lr()[0]
        self.ckp.write_log(f'Epoch {epoch} with Lr {lr}')
        self.model.train()
        self.ckp.start_log()

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input.to(self.device)
            gt = gt.to(self.device)

            output, _, _, mid_loss = self.model(input)

            self.optimizer.zero_grad()
            
            tempp = False
            if tempp:
                
                loss_log = self.loss.get_init_loss_log()
                print_pretty(torch.min(output),torch.max(output),torch.min(gt),torch.max(gt)) 
                out_lin = nn.Sigmoid()(output)
                print_pretty('New scale: ', torch.min(out_lin),torch.max(out_lin))
                loss = nn.BCELoss().to(self.device)(out_lin,gt)
                loss_log['Total'] = loss.item()
                #loss_log['grad'] = 0
                
                #if mid_loss:
                    #loss_log['others']=0
            else:


                loss, loss_log = self.loss(output, gt)

                grad_loss = self.grad_loss(output, gt)
                loss = loss + 0.1 * grad_loss
                loss_log['grad'] = grad_loss.item()
                loss_log['Total'] = loss.item()

                if mid_loss:  # mid loss is the loss during the model
                    effective_mid_loss = self.args.mid_loss_weight * mid_loss
                    loss = loss + effective_mid_loss
                    loss_log['others'] = effective_mid_loss.item()
                    loss_log['Total'] = loss.item()

            loss.backward()
            # This is where self.optimizer.step() was
            self.optimizer.step()

            self.ckp.report_log(loss_log)

            if (batch + 1) % self.args.print_every == 0 and self.distributed_manager.is_parent_gpu():
                self.ckp.write_log('[{}/{}]\tLoss : {}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.display_loss(batch)
                ))

            if (batch + 1) % self.args.max_iter_save == 0:
                self.model.save_now_model(flag='{}_{}'.format(epoch - 1, (batch + 1) // self.args.max_iter_save))

        self.ckp.end_log(len(self.loader_train))


    def do_validate(self):
         print_pretty("PreDehaze: Now testing")
         epoch = self.scheduler.last_epoch + 1
         self.ckp.write_log('\nEvaluation:')
         self.model.eval()
         self.ckp.start_log(train=False)
         with torch.no_grad():
             tqdm_test = tqdm(self.loader_test)
             for idx, (input, gt, filename) in enumerate(tqdm_test):

                 #filename = filename[0]
                 filename = 'img_'+str(idx)+'.png'
                 input = input.to(self.device)
                 gt = gt.to(self.device)

                 output, trans, air, _ = self.model(input)

                 PSNR = data_utils.calc_psnr(gt, output, rgb_range=self.args.rgb_range, )
                 self.ckp.report_log(PSNR, train=False)

                 if self.args.save_images:
                     gt, input, output, trans, air = data_utils.postprocess(gt, input, output, trans, air,
                                                                            rgb_range=self.args.rgb_range)
                     combine1 = np.concatenate((input, output, gt), axis=1)
                     combine2 = np.concatenate((trans, air, air), axis=1)
                     combine = np.concatenate((combine1, combine2), axis=0)
                     save_list = [input,output,gt]
                     self.ckp.save_images(filename, save_list)

             self.ckp.end_log(len(self.loader_test), train=False)
             best = self.ckp.psnr_log.max(0)
             self.ckp.write_log('[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                 self.args.data_test, self.ckp.psnr_log[-1],
                 best[0], best[1] + 1))
             #if not self.args.test_only:
                 #self.ckp.save(epoch, is_best=(best[1] + 1 == epoch))
         #if not self.args.test_only:
             #self.ckp.save(self, epoch, is_best=False)
         print_pretty("PreDehaze: Now testing")
