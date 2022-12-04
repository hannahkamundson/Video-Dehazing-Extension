import torch

import data
import model
import loss
import option
from trainer.trainer_dehaze import Trainer_Dehaze
from trainer.trainer_pre_dehaze import Trainer_Pre_Dehaze
from logger import logger

args = option.args
torch.manual_seed(args.seed)

loss = loss.Loss(args)
init_loss_log = loss.get_init_loss_log()

chkp = logger.Logger(args, init_loss_log)
model = model.Model(args, chkp)
loader: data.Data = data.Data(train_dataset_name=args.data_train, 
    test_dataset_name=args.data_test, 
    test_only=args.test_only,
    batch_size=args.batch_size, 
    number_of_threads=args.n_threads,
    is_cpu=args.cpu,
    namespace=args)

print("Selected task: {}".format(args.task))
if args.task == 'PreDehaze':
    t = Trainer_Pre_Dehaze(args, loader, model, loss, chkp)
elif args.task == 'ImageDehaze':
    t = Trainer_Dehaze(args, loader, model, loss, chkp)
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

while not t.terminate():
    t.train()
    t.test()

chkp.done()
