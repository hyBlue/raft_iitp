from __future__ import print_function, division
import argparse
import sys
sys.path.append('core')
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from raft import RAFT, replace_module
import evaluate
import tqdm
import datasets
from pruner import init_pruner

from nipq.nipq import QuantOps as Q
from nipq.train_test import (
    resume_checkpoint,
    create_checkpoint,
    test,
    train_aux,
    # train_aux_target_avgbit,
    train_aux_target_bops,
    CosineWithWarmup,
    bops_cal,
    bit_loss,
    accuracy,
    categorize_param,
    bit_cal,
    get_optimizer
)
from torch.utils.tensorboard import SummaryWriter



try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] /
                        SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    if args.pruning and not args.quantize:
        model = RAFT(args)
    else:
        model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    # if args.restore_ckpt is not None:
    #     model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    if args.pruning and not args.quantize:
        checkpoint_dict = torch.load(args.restore_ckpt, map_location='cuda')
        state_dict = checkpoint_dict
        model_dict = model.state_dict()

        state_idx = 0
        for key, weight in state_dict.items():
            model_dict[key.replace('module.', '')] = weight.clone()

        model.load_state_dict(model_dict)

        # model.load_state_dict(model_dict)
        print('LOAD CHECKPOINT')

        # args.pruner : str(KETIPrunerStructured), wgt(KETIPrunerWeight)
        pruner = init_pruner(model, args, None)
        model = pruner.prune()
        model = nn.DataParallel(model, device_ids=args.gpus)

    if args.quantize:
        replace_module(model, pruning=args.pruning, last_fp=args.last_fp)
        print(f'** act_q : {args.a_scale} / weight_q : {args.w_scale}')

        if args.restore_ckpt:
            checkpoint_dict = torch.load(
                args.restore_ckpt, map_location='cuda')
            state_dict = checkpoint_dict
            model_dict = model.state_dict()

            for key, weight in state_dict.items():
                model_dict[key] = weight.clone()

            model.load_state_dict(model_dict)
            # print('LOAD CHECKPOINT')

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True

    bn_tune = False
    if args.quantize:
        Q.initialize(model, act=args.a_scale > 0, weight=args.w_scale > 0)
        print('init quant')

        def forward_hook(module, inputs, outputs):
            module.out_shape = outputs.shape
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish)):
                hooks.append(module.register_forward_hook(forward_hook))
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(forward_hook))

        for data_blob in train_loader:
            break
        image1, image2, flow, valid = [x.cuda() for x in data_blob]
        model.eval()
        model.cuda()
        model.module(image1[:1], image2[:1], iters=1)
        for hook in hooks:
            hook.remove()

        weight, bnbias = categorize_param(model)

        if args.optim == 'adam':
            optimizer = optim.AdamW(params=[
                {'params': bnbias, 'weight_decay': 0., 'lr': args.lr},
                {'params': weight, 'weight_decay': args.wdecay, 'lr': args.lr}],
                eps=args.epsilon)
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(params=[
                {'params': bnbias, 'weight_decay': 0., 'lr': args.lr},
                {'params': weight, 'weight_decay': 1e-5, 'lr': args.lr},
            ], lr=args.lr, momentum=0.9, nesterov=True)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                                                  pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    with tqdm.tqdm(total=args.num_steps + 1) as pbar:
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                if not bn_tune and (total_steps > args.ft_epoch) and args.quantize:
                    print('BN FT START')
                    Q.initialize(model, act=args.a_scale > 0,
                                 weight=args.w_scale > 0, noise=False)

                    for name, module in model.named_modules():
                        if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish, Q.Conv2d, Q.Linear)):
                            module.bit.requires_grad = False
                    bn_tune = True
                optimizer.zero_grad()
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                if args.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*
                                                          image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*
                                                          image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = model(image1, image2, iters=args.iters)

                loss, metrics = sequence_loss(
                    flow_predictions, flow, valid, args.gamma)

                if args.quantize and not bn_tune:
                    loss_bit = bit_loss(model, 0, bit_scale_a=args.a_scale,
                                        bit_scale_w=args.w_scale, target_bit=args.target, is_linear=False)
                    loss += loss_bit

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                logger.push(metrics)

                if total_steps % VAL_FREQ == VAL_FREQ - 1:
                    PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                    torch.save(model.state_dict(), PATH)

                    results = {}
                    for val_dataset in args.validation:
                        if val_dataset == 'chairs':
                            results.update(
                                evaluate.validate_chairs(model.module))
                        elif val_dataset == 'sintel':
                            results.update(
                                evaluate.validate_sintel(model.module))
                        elif val_dataset == 'kitti':
                            results.update(
                                evaluate.validate_kitti(model.module))

                    logger.write_dict(results)

                    model.train()
                    if args.stage != 'chairs':
                        model.module.freeze_bn()

                    if args.quantize:
                        a_bit, au_bit, w_bit, wu_bit = bit_cal(model)
                        bops_total = bops_cal(model)
                        print(
                            f'Iter : [{total_steps}] / a_bit : {au_bit}bit / w_bit : {wu_bit}bit / bops : {bops_total.item()}GBops')

                total_steps += 1
                pbar.update(1)

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument(
        '--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int,
                        nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    ## QUANTIZATION
    parser.add_argument('--quantize',  action='store_true',
                        help='save results')
    parser.add_argument("--target", default=8, type=float,
                        help='target bitops or avgbit')
    parser.add_argument("--ft_epoch", default=3, type=int, help='tuning epoch')
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--last_fp", action='store_true')
    parser.add_argument("--a_scale", default=1, type=float)
    parser.add_argument("--w_scale", default=1, type=float)

    ## PRUNING
    parser.add_argument("--pruning", action='store_true')     
    parser.add_argument("--prune_ratio_method", type=str, choices=['global', 'uniform', 'adaptive', 'manual'], default='uniform')   # Pruning ratio 적용 방식 지정
    parser.add_argument("--importance_metric", type=str, choices=['l1', 'l2', 'entropy'], default='l1')                             # Importance metric 지정
    parser.add_argument("--prune_ratio", type=float, default=0.7)                                                                   # Sparsity 지정 (= 1 - density)
    parser.add_argument("--applyFirstLastLayer", action='store_true')     
    parser.add_argument('--pruner', choices=['str', 'wgt'], default='wgt')                                   # Pruner 유형 선택 : Channel(str), Weight(wgt)
    parser.add_argument("--iterative", action='store_true')                                                                         # Iterative Pruning (if false: One-shot)
    parser.add_argument("--iter_num", type=int, default=5)
    parser.add_argument("--KD", action='store_true')       

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
