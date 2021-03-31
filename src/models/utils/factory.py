import logging

logger = logging.getLogger(__name__)

from ..transformer_model import STAM_224


def create_model(args):
    """Create a model
    """
    # args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='stam_16':
      args.frames_per_clip = 16
    elif args.model_name=='stam_32':
      args.frames_per_clip = 32
      args.frame_rate = 3.2
    elif args.model_name=='stam_64':
      args.frames_per_clip = 64
      args.frame_rate = 6.4
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    model_params = {'args': args, 'num_classes': args.num_classes}
    model = STAM_224(model_params)

    return model