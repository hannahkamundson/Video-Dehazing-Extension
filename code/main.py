import torch
import time

import data
import model
import loss
import option
from trainer.trainer_dehaze import Trainer_Dehaze
from trainer.trainer_pre_dehaze import Trainer_Pre_Dehaze
from logger import logger

beginning_time = time.perf_counter()
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

if not args.cpu:
    print(f'Running # GPUs: {torch.cuda.device_count()}')
print("Selected task: {}".format(args.task))
task_type: str = args.task
if task_type == 'PreDehaze':
    t = Trainer_Pre_Dehaze(args, loader, model, loss, chkp)
elif task_type == 'ImageDehaze':
    t = Trainer_Dehaze(args, loader, model, loss, chkp)
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

while not t.terminate():
    t.train()
    t.test()

chkp.done()

end_time = time.perf_counter()
print(f"Ran in {end_time - beginning_time:0.4f} seconds")