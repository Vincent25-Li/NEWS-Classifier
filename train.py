"""Train a model on NEWS"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.transforms as transforms
import torch.utils.data as data
import utils

from json import dumps
from ujson import load as json_load
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import DistilBERT
from transformers import DistilBertConfig
from utils import NEWS, collate_fn
from args import get_train_args

def main(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    config = DistilBertConfig()
    model = DistilBERT(config, args.num_labels,
                       use_img=args.use_img,
                       img_size=args.img_size)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()

    # Get saver
    saver = utils.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)
    
    # Set optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), args.lr,
                            weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    train_dataset = NEWS(args.train_record_file, transform=transform)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = NEWS(args.dev_record_file, transform=transform)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for input_idxs, atten_masks, images, ids, y in train_loader:
                # Setup for forward
                input_idxs = input_idxs.to(device)
                atten_masks = atten_masks.to(device)
                images = images.to(device)
                y = y.to(device)
                batch_size = input_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p = model(input_idxs, atten_masks, images)
                loss = F.nll_loss(log_p, y)
                loss_val = loss.item()

                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step(step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    results = evaluate(model, dev_loader, device)
                    saver.save(step, model, results[args.metric_name], device)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)


def evaluate(model, data_loader, device):
    nll_meter = utils.AverageMeter()
    
    model.eval()
    pred_dict = {}
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for input_idxs, atten_masks, images, ids, y in data_loader:
                # Setup for forward
                input_idxs = input_idxs.to(device)
                atten_masks = atten_masks.to(device)
                images = images.to(device)
                y = y.to(device)
                batch_size = input_idxs.size(0)

                # Forward
                log = model(input_idxs, atten_masks, images)
                loss = F.nll_loss(log, y)
                nll_meter.update(loss.item(), batch_size)

                # Get F1 score
                preds = utils.get_pred_ans_pair(ids.tolist(),
                                                log.exp().tolist(),
                                                y.tolist())
                pred_dict.update(preds)

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(NLL=nll_meter.avg)
    
    model.train()

    F1 = utils.compute_f1(pred_dict)
    results = {'NLL': nll_meter.avg,
               'F1': F1}

    return results

if __name__ == '__main__':
    main(get_train_args())