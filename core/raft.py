import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.append('..')
from nipq.nipq import Q_Conv2d, Q_ReLU, Q_Linear, Q_Sym
from utils.utils import bilinear_sampler, coords_grid, upflow8
from corr import CorrBlock, AlternateCorrBlock
from extractor import BasicEncoder, SmallEncoder
from update import BasicUpdateBlock, SmallUpdateBlock
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(
                output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(
                output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.relu = nn.ReLU()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(
                fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = self.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


def replace_module(module, name='', pruning=False, last_fp=False):
    names = [i for i, j in module.named_children()]
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.Conv2d):
            if last_fp and  'update_block.flow_head.conv2' in f'{name}.{child_name}' :
                pass
            elif 'Conv2d' in str(child_module):
                setattr(module, child_name, Q_Conv2d(child_module.in_channels, child_module.out_channels,
                        child_module.kernel_size, child_module.stride, child_module.padding, mask=pruning))

        elif isinstance(child_module, nn.Linear):

            if last_fp and child_name == names[-1]:
                pass
            else:
                if child_module.bias is not None:
                    bias = True
                else:
                    bias = False
                setattr(module, child_name, Q_Linear(child_module.in_features, child_module.out_features,
                        bias, mask=pruning))
        elif isinstance(child_module, nn.ReLU):
            setattr(module, child_name, Q_ReLU())
        elif isinstance(child_module, nn.Identity):
            setattr(module, child_name, Q_Sym())
        else:
            replace_module(
                child_module, name=f'{name}.{child_name}' if name else child_name, pruning=pruning, last_fp=last_fp)


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

    parser.add_argument('--quantize',  action='store_true',
                        help='save results')
    parser.add_argument("--target", default=8, type=float,
                        help='target bitops or avgbit')
    parser.add_argument("--ft_epoch", default=3, type=int, help='tuning epoch')
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--last_fp", action='store_true')

    args = parser.parse_args()
    model = RAFT(args)

    replace_module(model, last_fp=True)
    print(model)
