from argparse import Namespace
from enum import Enum
from utils.print import print_pretty

class DatasetName(Enum):
    RESIDE_1 = 'RESIDE_1'
    RESIDE_3 = 'RESIDE_3'
    RESIDE_10 = 'RESIDE_10'
    OHAZE = 'OHAZE'
    REVIDE = 'REVIDE'
    REVIDE_REDUCED = 'REVIDE_REDUCED'

def _set_base(args: Namespace) -> Namespace:
    """
    Set some basic parameters that we typically use for all the templates.

    Args:
        args: The argparse namespace you want modified
    
    Returns:
        The argparse namespace that was entered + the new changes we want to the args
    """
    args.data_train = 'RESIDE'
    args.data_test = 'RESIDE'
    args.dir_data_test = '../dataset/SOTS/indoor'
    args.t_channels = 1
    args.n_feat = 32
    args.n_resblock = 3
    args.size_must_mode = 4
    args.loss = '1*L1'
    args.lr = 1e-4
    args.lr_decay = 200
    args.epochs = 5
    args.batch_size = 32*6
    args.mid_loss_weight = 0.05
    args.save_middle_models = True
    args.save_images = False
    # args.resume = True
    # args.load = args.save
    # args.test_only = True

    return args

def set_template(args: Namespace) -> Namespace:
    """
    Set the arparse namespace for different templates we have. For example, we have a "Pre_Dehaze".

    Args:
        args: The argparse namespace you want modified
    
    Returns:
        The argparse namespace that was entered + the new changes we want to the args
    """
    baseResideTrainDSPath = '../dataset/RESIDE/RESIDE/ITS_train'
    template_type: str = args.template
    args = _set_base(args)

    # Sorry this if statement is messy and def not DRY but I'm just keeping it how it is
    # Be careful if you decide to change things cuz there is a lot of overlapping stuff
    if template_type.startswith('Pre_Dehaze'):
        args.task = "PreDehaze"
        args.model = "PRE_DEHAZE_T"
        args.other_loss = 'grad+others'

        # All of these are stored in the RESIDE path
        if template_type == 'Pre_Dehaze':
            print_pretty("Creating the template for Pre_Dehaze 1x1")
            args.dir_data = baseResideTrainDSPath
            args.save = DatasetName.RESIDE_1.name
        elif template_type == 'Pre_Dehaze_3':
            print_pretty("Creating the template for Pre_Dehaze 3x3")
            args.dir_data = baseResideTrainDSPath + '_3'
            args.save = DatasetName.RESIDE_3.name
        elif template_type == 'Pre_Dehaze_10':
            print_pretty("Creating the template for Pre_Dehaze 10x10")
            args.dir_data = baseResideTrainDSPath + '_10'
            args.save = DatasetName.RESIDE_10.name
        # This one isn't RESIDE but we are still storing it in the same path
        elif template_type == "Pre_Dehaze_ohaze":
            print_pretty("Creating the template for Pre_Dehaze Ohaze")
            args.dir_data = baseResideTrainDSPath + '_ohaze'
            args.save = DatasetName.OHAZE.name
        # REVIDE dataset is really similar but we have a few more modifications so there is more here
        elif template_type == "Pre_Dehaze_revide":
            print_pretty("Creating the template for Pre_Dehaze REVIDE")
            args.data_train = 'REVIDE'
            args.data_test = 'REVIDE'
            # dir_data is the path to the training data
            args.dir_data = '../dataset/REVIDE/Train'
            args.save = DatasetName.REVIDE.name
        # REVIDE reduced dataset
        elif template_type == "Pre_Dehaze_revidereduced":
            print_pretty("Creating the template for Pre_Dehaze REVIDE reduced")
            args.data_train = 'REVIDE'
            args.data_test = 'REVIDE'
            # dir_data is the path to the training data
            args.dir_data = '../dataset/REVIDE_REDUCED/Train'
            args.save = DatasetName.REVIDE_REDUCED.name
        else:
            raise NotImplementedError('Template Pre Dehaze [{:s}] is not found'.format(args.template))

        return args
    elif template_type.startswith('Dehaze') or template_type == 'ImageDehaze_SGID_PFF':
        args.other_loss = 'grad+refer+others'
        args.task = "ImageDehaze"
        args.model = "DEHAZE_SGID_PFF"
        
        # If it is what was used previously in the codebase, continue that
        if template_type == 'ImageDehaze_SGID_PFF':
            args.dir_data = baseResideTrainDSPath
            args.save = "ImageDehaze_SGID_PFF"
        # If it is the REVIDE reduced, do that
        elif template_type == 'Dehaze_revidereduced':
            # dir_data is the path to the training data
            args.dir_data = '../dataset/REVIDE_REDUCED/Train'
            args.save = DatasetName.REVIDE_REDUCED.name
            args.data_train = 'REVIDE'
            args.data_test = 'REVIDE'
            
            # We need to have an associated timestamp to read in the model
            # This is required when we are saving in the logger that way we can save Pre_Dehaze with the
            # Dehaze and keep everything organized
            # Throw an error if there isn't one
            if args.prev_timestamp is None:
                raise ValueError("You need to have a previous timestamp if you are running Dehaze. Specify with command line --prev_timestamp")

        return args
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
