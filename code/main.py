import torch
import time

import data
import model as md
from loss import Loss
import option
from trainer.trainer_dehaze import Trainer_Dehaze
from trainer.trainer_pre_dehaze import Trainer_Pre_Dehaze
from logger import data_dirs, logger
from argparse import Namespace
from par import DistributedManager, create, slurm
from utils.print import print_pretty


def do_run(args: Namespace):
    torch.manual_seed(args.seed)
    
    # If we should be adding slurm variables to the namespace, do it
    if args.slurm_env_var:
        args = slurm.add_slurm_env_vars(args)
        
    distributed_manager: DistributedManager = create(args)

    loss = Loss(args)
    init_loss_log = loss.get_init_loss_log()

    # Create info about the data directory structure
    dirs = data_dirs.DataDirectory(args,
                                   # Only write if it isn't distributed or it is the parent gpu
                                   should_write=not distributed_manager.is_distributed or distributed_manager.is_parent_gpu())
    chkp = logger.Logger(args, init_loss_log, dirs, distributed_manager=distributed_manager)

    # Load the model wrapper which chooses the appropriate model
    model = md.Model(is_cpu=args.cpu,
                    args=args,
                    number_gpus=args.n_GPUs,
                    save_middle_models=args.save_middle_models,
                    model_type=args.model,
                    resume_previous_run=args.resume,
                    auto_pre_train=args.auto_pre_train,
                    pre_train_path=args.pre_train,
                    test_only=args.test_only,
                    ckp=chkp, 
                    dirs=dirs,
                    distributed_manager=distributed_manager)

    # Load the data based oin what type of data was specified
    loader: data.Data = data.Data(train_dataset_name=args.data_train, 
        test_dataset_name=args.data_test, 
        test_only=args.test_only,
        batch_size=args.batch_size, 
        number_of_threads=args.n_threads,
        is_cpu=args.cpu,
        namespace=args,
        distributed_manager=distributed_manager)

    if not args.cpu:
        print_pretty(f'Running # GPUs: {torch.cuda.device_count()}')

    # Run the selected task
    print_pretty("Selected task: {}".format(args.task))
    task_type: str = args.task
    if task_type == 'PreDehaze':
        t = Trainer_Pre_Dehaze(args, loader, model, loss, chkp, distributed_manager)
    elif task_type == 'ImageDehaze':
        t = Trainer_Dehaze(args, loader, model, loss, chkp, distributed_manager)
    else:
        raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

    # While we haven't done all the epochs, train, test/validate, and step to the next epoch
    while not t.terminate():
        t.pre_train()
        t.train()
        t.validate()
        t.step_next()

    # Close out the logger
    chkp.done()


# Run the script
beginning_time = time.perf_counter()
args = option.args
do_run(args)

end_time = time.perf_counter()
print_pretty(f"Ran in {end_time - beginning_time:0.4f} seconds")
    

