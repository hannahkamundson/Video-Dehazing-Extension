import argparse
from option import template

parser = argparse.ArgumentParser(description='Image_Dehaze', add_help=True)

parser.add_argument('--template', default='ImageDehaze_SGID_PFF',
                    help='You can set various templates in options.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Multi node multi GPU processing
parser.add_argument('--slurm_env_var', 
                    action='store_true',
                    default=False,
                    help='Should we pull multi node/multi GPU info from SLRUM env vars?')
parser.add_argument('--is_distributed', 
                    action='store_true',
                    default=False,
                    help='Do you want this to be multi node multi GPU distributed?')
parser.add_argument('--global_rank', 
                    type=int, 
                    help='The global rank of the GPU in comparison to all other GPUs')
parser.add_argument('--local_rank', 
                    type=int, 
                    help='The local rank of the GPU in comparison to all GPUs on the same node')
parser.add_argument('--gpus_per_node', 
                    type=int, 
                    help='The number of GPUs per node')
parser.add_argument('--number_nodes', 
                    type=int, 
                    help='The number of nodes')

# Data specifications
parser.add_argument('--dir_data', type=str, default='.',
                    help='dataset directory')
parser.add_argument('--dir_data_test', type=str, default='.',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='.',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='.',
                    help='test dataset name')
parser.add_argument('--process', action='store_true',
                    help='if True, load all dataset at once at RAM')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--size_must_mode', type=int, default=1,
                    help='the size of the network input must mode this number')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='.',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--prev_timestamp', type=str,
                    help='List the timestamp in the experiment directory that you want to use')
parser.add_argument('--auto_pre_train', action='store_true',
                    help='Auto load the pre trained model with the same timestamp')

# Training specifications
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--other_loss', type=str, default='others',
                    help='loss function configuration')
parser.add_argument('--mid_loss_weight', type=float, default=1.,
                    help='the weight of mid loss in trainer')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Log specifications
parser.add_argument('--experiment_dir', type=str, default='../experiment/',
                    help='file name to save')
parser.add_argument('--pretrain_models_dir', type=str, default='../pretrained_models/',
                    help='file name to save')
parser.add_argument('--save', type=str, default='default_save',
                    help='file name to save')
parser.add_argument('--save_middle_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', action='store_true',
                    help='resume from the latest if true')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_images', default=True, action='store_true',
                    help='save images')
parser.add_argument('--max_iter_save', type=int, default=2000,
                    help='how many batches to wait before logging training status')


args = parser.parse_args()
args = template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
