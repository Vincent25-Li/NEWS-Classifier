"""Train a model on NEWS"""

import utils

from json import dumps
from tensorboardX import SummaryWriter

def main(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))