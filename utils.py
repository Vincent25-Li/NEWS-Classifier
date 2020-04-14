"""Utility classes and methods"""

import os
import numpy as np
import torch
import torch.utils.data as data

from skimage import io

class NEWS(data.Dataset):
    """NEWS Dataset.
    
    Each item in the dataset is a tuple with the following entries (in order):
        - input_idxs: Indices of the tokens in title and content.
            Shape (input_len,).
        - atten_masks: Masks of input indices.
            Shape (input_len,).
        - img: Image of NEWS.
            Shape (x, x, 3).
        - y: Category index of NEWS
    
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset and image paths.
        transform (obj): Transform to apply on image.
    """

    def __init__(self, data_path, transform):
        super(NEWS, self).__init__()

        dataset = np.load(data_path)

        self.input_idxs = torch.from_numpy(dataset['input_idxs']).long()
        self.atten_masks = torch.from_numpy(dataset['atten_masks']).long()
        self.img_paths = dataset['img_paths']
        self.y = torch.from_numpy(dataset['y']).long()
        self.transform = transform
    
    def __getitem__(self, idx):
        image = io.read(self.img_paths[idx])
        image = self.transform(image)

        example = (self.input_idxs[idx],
                   self.atten_masks[idx],
                   image,
                   self.y[idx])
        return example

    def __len__(self):
        return len(self.y)

def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `NEWS.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (input_idxs, atten_masks, images, y)

    Returns:
        examples (tuple): Tuple of tensors (input_idxs, atten_masks, images, y).
        All of shape (batch_size, ...)
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_input(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, max(lengths)
    
    def merge_mask(arrays, length, dtype=torch.int64):
        merged = torch.zeros(len(arrays), length, dtype=dtype)
        for i, seq in enumerate(arrays):
            merged[i] = seq[:length]
        return merged
    
    def merge_image(arrays, dtype=torch.float64):
        channel, height, weight = arrays[0].size()
        merged = torch.zeros(len(arrays), channel, height, weight, dtype=dtype)
        for i, image in enumerate(arrays):
            merged[i] = image
        return merged
    
    # Group by tensor type
    input_idxs, atten_masks, images, y = zip(*examples)

    # Merge into batch tensors
    input_idxs, length = merge_input(input_idxs)
    atten_masks = merge_mask(atten_masks, length)
    images = merge_image(images)
    y = merge_0d(y)

    return (input_idxs, atten_masks, images, y)

# Credit to Chris Chute (chute@stanford.edu)
def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')

# Credit to Chris Chute (chute@stanford.edu)
def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Credit to Chris Chute (chute@stanford.edu)
def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids