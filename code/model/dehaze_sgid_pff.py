import torch.nn as nn
import torch
from model.pre_dehaze_t import PRE_DEHAZE_T
from model.dehaze_t import DEHAZE_T
from logger.data_dirs import DataDirectory
import os
from utils.data_utils import get_device_type
from utils.print import print_pretty


def make_model(args, dirs: DataDirectory):
    pretrain_pre_dehaze_pt = args.pretrain_models_dir + 'pretrain_pre_dehaze_net.pt' if not args.test_only else '.'
    return DEHAZE_SGID_PFF(img_channels=args.n_colors, 
                           t_channels=args.t_channels, 
                           n_resblock=args.n_resblock,
                           n_feat=args.n_feat, 
                           pretrain_pre_dehaze_pt=pretrain_pre_dehaze_pt, 
                           device=get_device_type(args.cpu),
                           auto_load_pretrained=args.auto_pre_train,
                           dirs=dirs)


class DEHAZE_SGID_PFF(nn.Module):

    def __init__(self, 
                 auto_load_pretrained,
                 dirs: DataDirectory,
                 img_channels=3, 
                 t_channels=1, 
                 n_resblock=3, 
                 n_feat=32,
                 pretrain_pre_dehaze_pt='.', 
                 device='cuda'):
        super(DEHAZE_SGID_PFF, self).__init__()
        print_pretty("Creating Dehaze-SGID-PFF Net")
        self.device = device

        self.pre_dehaze = PRE_DEHAZE_T(img_channels=img_channels, 
                                       t_channels=t_channels, 
                                       n_resblock=n_resblock,
                                       n_feat=n_feat, 
                                       device=device)
        self.dehaze = DEHAZE_T(img_channels=img_channels, 
                               t_channels=t_channels, 
                               n_resblock=n_resblock,
                               n_feat=n_feat, 
                               device=device)

        if auto_load_pretrained:
            self.pre_dehaze.load_state_dict(dirs.load_torch_from_pre_dehaze())
            print_pretty('Auto loading pre dehaze model from {}'.format(os.path.abspath(dirs.pre_dehaze_model_path())))
        elif pretrain_pre_dehaze_pt != '.':
            self.pre_dehaze.load_state_dict(torch.load(pretrain_pre_dehaze_pt))
            print_pretty('Loading pre dehaze model from {}'.format(pretrain_pre_dehaze_pt))

    def forward(self, x, prior_image):
        # If there isn't a prior image, we need to use the pre dehaze to get a reference image
        if prior_image is None:
            pre_est_J, _, _, _ = self.pre_dehaze(x)
        # Otherwise, we need to use the prior image as our estimate
        else:
            pre_est_J = prior_image

        output, trans, air, mid_loss = self.dehaze(x, pre_est_J)

        return pre_est_J, output, trans, air, mid_loss
