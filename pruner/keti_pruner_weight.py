"""
KETI Pruner 2023 : Weight Pruning (Unstructured Pruning)

*** Author : Jonghee Park & Hojong Shin @ Korea Electronics Technology Institute
*** E-mail : jpark19@keti.re.kr
"""


import sys
import os
sys.path.insert(1, os.getcwd())
from pruner.utils.calculate_zero_ratio import calculate_zero_ratio
from pruner.utils.logger import Logger
# from pruner.utils.trainer import Trainer
# from backbone import build_backbone
from math import ceil, sqrt
from functools import reduce
import argparse
import copy
import torch.nn as nn
import torch
def get_n_params_(model):
    n_params = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            n_params += module.weight.numel()
            if hasattr(module, 'bias') and type(module.bias) != type(None):
                n_params += module.bias.numel()
            if hasattr(module, 'running_mean') and type(module.running_mean) != type(None):
                n_params += module.running_mean.numel()
            if hasattr(module, 'running_var') and type(module.running_var) != type(None):
                n_params += module.running_var.numel()
    return n_params

class PruneLinear(nn.Linear):
    """
    nn.Linear를 상속받은 Prunable module

    :param origin_linear: 대상 원본 module (nn.Linear)
    :param importance_metric: weight pruning의 importance metric (l1 or l2)
    """

    def __init__(self, origin_linear: nn.Linear, importance_metric):
        super(PruneLinear, self).__init__(origin_linear.in_features,
                                          origin_linear.out_features, origin_linear.bias != None)
        self.pruningRatio = 0
        self.importance_metric = importance_metric
        # self.weight_original = copy.deepcopy(origin_linear.weight)
        # self.bias_original = copy.deepcopy(origin_linear.bias)
        self.mask = nn.Parameter(torch.ones_like(self.weight.data).cuda().flatten(), requires_grad=False)

    def forward(self, x):
        mask = self.mask.view_as(self.weight.data)
        self.weight.data = self.weight.data.mul_(mask)

        return super(PruneLinear, self).forward(x)

    def getImportanceScore(self):
        """Module 내의 개별 weight에 대한 importance 계산 (L1 or L2 기반)"""
        score = None
        if self.importance_metric == 'l1':
            score = self.weight.abs().flatten()
        elif self.importance_metric == 'l2':
            score = self.weight.square().flatten()
        elif self.importance_metric == 'entropy':
            score = torch.special.entr(self.weight.abs()).flatten()

        return score.cuda()

    def resetUnpruneState(self):
        """Module 내의 mask 및 pruning ratio 관련 정보를 초기화"""
        self.mask = torch.ones_like(self.weight.data).cuda().flatten()
        # self.weight = copy.deepcopy(self.weight_original)
        # self.bias = copy.deepcopy(self.bias_original)
        self.pruningRatio = 0

    def generateLinear(self):
        """Pruning 적용 이후 고유 weight model을 생성 (nn.Linear)"""
        result = nn.Linear(self.in_features, self.out_features)
        result.weight = copy.deepcopy(self.weight)
        mask = self.mask.view_as(self.weight.data)
        result.weight.data = result.weight.data.mul_(mask)
        result.bias = copy.deepcopy(self.bias)

        return result.cuda()

    def genWeightMaskbyPruningRatio(self, pruningRatio):
        """
        대상 module의 mask를 특정 ratio(layerwise sparsity)를 기준으로 생성한다.\n
        (args.prune_ratio_method 가 uniform일 경우에 사용)

        :param pruningRatio: layerwise sparsity
        """
        score = self.getImportanceScore()
        elemNum = len(score)
        n_pruned = min(ceil(pruningRatio * elemNum), elemNum - 1)
        prune_idx = score.sort()[1][:n_pruned]
        self.mask[prune_idx] = 0


class PruneConv2d(nn.Conv2d):
    """
    nn.Conv2d를 상속받은 Prunable module

    :param origin_bn: 대상 원본 module (nn.BatchNorm2d)
    :param importance_metric: weight pruning의 importance metric (l1 or l2)
    """

    def __init__(self, origin_conv: nn.Conv2d, importance_metric):
        super(PruneConv2d, self).__init__(
            origin_conv.in_channels, origin_conv.out_channels, origin_conv.kernel_size, origin_conv.stride, origin_conv.padding, origin_conv.dilation, origin_conv.groups, origin_conv.bias != None)
        self.weight = copy.deepcopy(origin_conv.weight)
        self.bias = copy.deepcopy(origin_conv.bias)

        # self.weight_original = copy.deepcopy(origin_conv.weight)
        # if origin_conv.bias is not None:
        #    self.bias_original = copy.deepcopy(origin_conv.bias)
        # else:
        #    self.bias_original = None

        self.mask = nn.Parameter(torch.ones_like(self.weight.data).cuda().flatten(), requires_grad=False)
        self.importance_metric = importance_metric
        self.pruningRatio = 0

    def forward(self, x):
        mask = self.mask.view_as(self.weight.data)
        self.weight.data = self.weight.data.mul_(mask)
        return super(PruneConv2d, self).forward(x)

    def generateConv2d(self):
        """
        Pruning 적용 이후 고유 weight model를 생성 (nn.Conv2d)

        :return: result (=pruned model)
        """
        result = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                           self.padding, self.dilation, self.groups, self.bias != None).cuda()
        result.weight = copy.deepcopy(self.weight)
        mask = self.mask.view_as(self.weight.data)
        result.weight.data = result.weight.data.mul_(mask)
        result.bias = copy.deepcopy(self.bias)

        return result.cuda()

    def genWeightMaskbyPruningRatio(self, pruningRatio):
        """
        대상 module의 weight mask를 특정 ratio(layerwise sparsity)를 기준으로 생성한다.\n
        (args.prune_ratio_method 가 uniform일 경우에 사용)

        :param pruningRatio: layerwise sparsity
        """
        score = self.getImportanceScore()
        elemNum = len(score)
        n_pruned = min(ceil(pruningRatio * elemNum), elemNum - 1)
        prune_idx = score.sort()[1][:n_pruned]
        self.mask[prune_idx] = 0

    def getImportanceScore(self):
        """
        Module 내의 개별 weight에 대한 importance 계산 (L1, L2, Entropy)

        :return: importance score
        """
        score = None
        if self.importance_metric == 'l1':
            score = self.weight.abs().flatten()
        elif self.importance_metric == 'l2':
            score = self.weight.square().flatten()
        elif self.importance_metric == 'entropy':
            score = torch.special.entr(self.weight.abs()).flatten()
        return score

    def genWeightMaskbyThreshold(self, threshold):
        """
        대상 module의 weight mask를 특정 threshold를 기준으로 생성한다.\n
        (args.prune_ratio_method 가 global일 경우에 사용)

        :param threshold: global threshold
        """
        score = self.getImportanceScore()
        self.mask = (score > threshold).float()

    def resetUnpruneState(self):
        """Module 내의 mask 및 pruning ratio 관련 정보를 초기화"""
        self.mask = nn.Parameter(torch.ones_like(self.weight.data).cuda().flatten(), requires_grad=False)
        # self.weight = copy.deepcopy(self.weight_original)
        # self.bias = copy.deepcopy(self.bias_original)
        self.pruningRatio = 0


class KETIPrunerWeight():
    """
    주어진 조건들을 기반으로 Weight Pruning(untructured)을 진행하는 Pruner 클래스

    :param model: 대상 원본 model (torch.nn.Module)
    :param args: 전달받은 argument parser 인자
    :param logger: global logger
    """

    def __init__(self, model, args, logger):
        self.originalModel = copy.deepcopy(model)
        self.pruneTargetModule = (nn.Conv2d, nn.Linear)
        self.prune_ratio_method = args.prune_ratio_method
        self.prune_ratio = args.prune_ratio

        self.importance_metric = args.importance_metric
        self.applyFirstLastLayer = args.applyFirstLastLayer
        self.firstLayer = None
        self.lastLayer = None

        # Iterative Pruning Option
        self.iterative = args.iterative
        self.iter_prune_ratio = 0
        self.iter_num = args.iter_num
        # self.shortTermTrainer = Trainer(args, logger, args.st_epochs, args.st_lr, args.st_warmup_epochs, args.st_cooldown_epochs)

        # Self distillation Option
        self.KD = args.KD

        # ShortTerm Finetuning 을 위한 학습 Hyper parameter 는 LongTerm과 공유
        # 향후 변경 필요한지 검토 필요
        self.args = args

        if self.iterative is True:
            assert self.iter_num > 0

    def prune(self, getOriginalArchtecture=False):
        """
        주어진 조건 하에서 Pruning 진행.

            - iterative / one-shot
            - masked model / original architecture 

        :param getOriginalArchtecture: (unstructured이므로) 고유 weight model 생성 여부 (default: fine-tuning 단계에선 False)
        :return: retModel (=pruned model)
        """
        if self.iterative:
            prunedModel = self.gen_prunedModelIter()
        else:
            prunedModel = self.gen_prunedModel()

        if getOriginalArchtecture:
            retModel = self.genOriginalArch(prunedModel)
        else:
            retModel = prunedModel

        return retModel

    def gen_prunedModelIter(self):
        """Iterative Pruning"""
        pr_step_each_iter = 1 - (1 - self.prune_ratio) ** (1. / self.iter_num)
        prevModel = self.originalModel

        for n in range(1, self.iter_num + 1):
            pr_each_time = pr_step_each_iter * \
                ((1 - pr_step_each_iter) ** (n - 1))
            self.iter_prune_ratio = pr_each_time + self.iter_prune_ratio
            prunedModel = self.gen_prunedModel(
                prevModel, self.iter_prune_ratio)

            if self.KD:
                bestAcc = self.shortTermTrainer.train(
                    prunedModel, teacher_model=prevModel)
            else:
                bestAcc = self.shortTermTrainer.train(prunedModel)

            print(f'iteration {n} best accuracy {bestAcc} ')

            tmp = self.genOriginalArch(prevModel)
            totalEleNum, zeroEleNum, zero_rate = calculate_zero_ratio(tmp)
            print(
                f'Total Element Number: {totalEleNum}, Zero Element Number : {zeroEleNum}, Zero Rate: {zero_rate}')
            prevModel = prunedModel

        return prevModel

    def gen_prunedModel(self, targetModel=None, prune_ratio=None):
        """One-Shot Pruining"""
        if targetModel is None:
            prunedModel = copy.deepcopy(self.originalModel)
        else:
            prunedModel = copy.deepcopy(targetModel)

        convIdx = 0
        allWeight = None

        if prune_ratio is not None:
            self.prune_ratio = prune_ratio

        for name, m in prunedModel.named_modules():
            tmp = None
            if isinstance(m, nn.Conv2d):
                tmp = PruneConv2d(m, self.importance_metric).cuda()
                convIdx += 1
            elif isinstance(m, nn.Linear):
                tmp = PruneLinear(m, self.importance_metric).cuda()

            if tmp is not None:
                tmp.weight = copy.deepcopy(m.weight)
                tmp.bias = copy.deepcopy(m.bias)

                if convIdx == 1:
                    self.firstLayer = tmp

                if self.prune_ratio_method == 'global':
                    if allWeight == None:
                        allWeight = tmp.getImportanceScore()
                    else:
                        allWeight = torch.cat(
                            (allWeight, tmp.getImportanceScore()), dim=0)
                elif self.prune_ratio_method == 'uniform':
                    tmp.genWeightMaskbyPruningRatio(self.prune_ratio)

                zeroRatio = (torch.numel(
                    tmp.mask) - torch.count_nonzero(tmp.mask)) / torch.numel(tmp.mask) * 100.0
                print(f'{name} prune ratio : {zeroRatio}')

                namesplit = name.split('.')
                att = reduce(getattr, namesplit[:-1], prunedModel)
                setattr(att, namesplit[-1], tmp)

                self.lastLayer = tmp

        if self.prune_ratio_method == 'global':
            elemNum = len(allWeight)
            val, idx = allWeight.sort()
            globalTrh = val[int(elemNum*self.prune_ratio)]

            for name, m in prunedModel.named_modules():
                if isinstance(m, PruneConv2d):
                    m.genWeightMaskbyThreshold(globalTrh)

        if not self.applyFirstLastLayer:
            self.firstLayer.resetUnpruneState()
            self.lastLayer.resetUnpruneState()

        return prunedModel

    def genOriginalArch(self, prunedModel):
        """
        취득한 pruned model의 고유 architecture 생성\n
        (unstructured case 이므로 고유 architecture가 아닌 고유 weight)

        :param prunedModel: masked model (원본 baseline과 동일한 규격)
        :return: retModel (=pruned model with its original architecture)
        """
        retModel = copy.deepcopy(prunedModel)

        for name, m in retModel.named_modules():
            tmp = None
            if isinstance(m, PruneConv2d):
                zeroRatio = (torch.numel(
                    m.mask) - torch.count_nonzero(m.mask)) / torch.numel(m.mask) * 100.0
                print(f'Final {name} prune ratio : {zeroRatio}')

                tmp = m.generateConv2d()
            elif isinstance(m, PruneLinear):

                zeroRatio = (torch.numel(
                    m.mask) - torch.count_nonzero(m.mask)) / torch.numel(m.mask) * 100.0
                print(f'Final {name} prune ratio : {zeroRatio}')

                tmp = m.generateLinear()

            if tmp is not None:
                namesplit = name.split('.')
                att = reduce(getattr, namesplit[:-1], retModel)
                setattr(att, namesplit[-1], tmp)

        return retModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="vgg7_bn")
    parser.add_argument("--dataset", type=str, default='cifar100')

    # weight pruning parameter
    parser.add_argument("--prune_ratio_method", type=str,
                        choices=['global', 'uniform', 'adaptive'], default='uniform')
    parser.add_argument("--importance_metric", type=str,
                        choices=['l1', 'l2', 'entropy'], default='l1')
    parser.add_argument("--prune_ratio", type=float, default=0.7)
    parser.add_argument("--applyFirstLastLayer", action='store_true')

    # For iterative pruning method
    parser.add_argument("--iterative", action='store_true')
    parser.add_argument("--iter_num", type=int, default=10)

    # Short term trainer parameter for interative method
    parser.add_argument('--data_path', metavar='DIR', help='path to dataset')
    parser.add_argument('-b', '--batch-size', '--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size_prune', type=int, default=64)
    parser.add_argument('--st_epochs', default=2, type=int,
                        metavar='N', help='short term finetune epochs')
    parser.add_argument('--st_cooldown_epochs', type=int, default=1, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--st_warmup-epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--st_lr', '--st-learning-rate', default=0.01, type=float, metavar='LR', help='short term initial learning rate',
                        dest='st_lr')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--sched', default='cosine', type=str,
                        metavar='SCHEDULER', help='LR scheduler (default: "step"')
    parser.add_argument("--KD", action='store_true')                        # Iterative KD 활용 여부

    # Parameter for logger
    parser.add_argument('--Exps_Dir', metavar='DIR',
                        help='path to output', default='/home/keti/jp_ubuntu_shared')
    parser.add_argument('--project', type=str, default="")
    parser.add_argument('--screen_print', action="store_true")

    args = parser.parse_args()
    args.num_classes = 100
    # logger = Logger(args)

    model = build_backbone(args)
    model.cuda()
    # print(model)
    pruner = KETIPrunerWeight(model, args, None)
    pruneModel = pruner.prune(getOriginalArchtecture=False)

    sd = pruneModel.state_dict()
    
    mask_dict = {}
    for n, m in pruneModel.named_modules():
        if hasattr(m, 'mask'):
            mask_dict[f'{n}.mask'] = m.mask
    sd['mask'] = mask_dict
    

            


    # pruneModel = model
    # for i in range(10):
    #     pruneModel = pruner.gen_prunedModel(pruneModel)

    # print(pruneModel)
    # mod = pruner.genOriginalArch(pruneModel)

