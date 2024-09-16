"""
KETI Pruner 2023 : Channel/Filter Pruning (Structured Pruning)

*** Author : Jonghee Park & Hojong Shin @ Korea Electronics Technology Institute
*** E-mail : jpark19@keti.re.kr
"""


import os
import sys
sys.path.insert(1, os.getcwd())

import json
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from functools import reduce
from math import ceil, sqrt

# from pruner.utils.trainer import Trainer
from pruner.utils.logger import Logger
# from backbone import build_backbone
from pruner.utils.calculate_zero_ratio import calculate_zero_ratio

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
class MaskGenerator:
    """
    Sturctured pruning을 위한 mask를 생성하는 class
    
    :param args: 전달받은 argument parser 인자
    :param schemaData: schema dict (from json file)
    """
    def __init__(self, args, schemaData):
        self.args = args
        self.minKeepChannelRatio = args.minKeepChannelRatio
        self.maskMergeType = args.maskMergeType
        self.schemaData = schemaData
        self.prune_ratio = args.prune_ratio
        self.channel_rounding = args.channel_rounding
        
    def genOutputMaskbyThreshold(self, module, threshold):
        """
        대상 module의 mask를 특정 threshold를 기준으로 생성한다. 
        
            - args.prune_ratio_method 가 global일 경우에 사용한다.
            - 최소로 남겨야 하는 비율(minKeepChannelRatio)을 두고 threshold에 무관하게 해당 비율보다 많이 제거할 수 없도록 한다.
        
        :param module: mask를 생성할 대상 module (torch.nn.Module)
        :param threshold: global threshold
        """
        minKeepRatio = self.minKeepChannelRatio

        score = module.getImportanceScore()
        tmpMask = torch.greater(score, threshold).float()

        if torch.count_nonzero(tmpMask) / len(tmpMask) < minKeepRatio:
            self.genOutputMaskbyPruningRatio(1 - minKeepRatio)
        else:
            module.out_mask = tmpMask

        compressRatio = (len(module.out_mask) - torch.count_nonzero(module.out_mask)) / len(module.out_mask) * 100
        print(f'{module.name} layer: {compressRatio} % removed')

    def genOutputMaskbyPruningRatio(self, module, pruningRatio):
        """
        대상 module의 mask를 특정 ratio(layerwise sparsity)를 기준으로 생성한다. 
        
            - args.prune_ratio_method 가 uniform일 경우에 사용한다.
            - args.channel_rounding 이 존재할 경우, channel 수 계산을 이에 맞춘다.
        
        :param module: mask를 생성할 대상 module (torch.nn.Module)
        :param pruningRatio: layerwise sparsity
        """
        score = module.getImportanceScore()
        elemNum = len(score)
        n_pruned = min(ceil(pruningRatio * elemNum), elemNum - 1)
        if self.channel_rounding is not None:
            n_pruned = self.applyChannelRounding(elemNum, n_pruned, self.channel_rounding)
        prune_idx = score.sort()[1][:n_pruned]
        module.out_mask[prune_idx] = 0

    def applyChannelRounding(self, n_total, n_pruned, align_val):
        """
        일부 accelerator 환경에 맞춰 pruning을 적용하고 남은 channel 수를 특정 배수로 조정한다. 
        
        :param n_total: 전체 channel 수 (기존)
        :param n_pruned: ratio에 따른 pruned channel 수
        :param align_val: channel_rounding 값
        :return: 조정된 n_pruned 값
        """
        pruned_ch = n_total - n_pruned
        retval = n_pruned

        if pruned_ch % align_val > 0:
            retval = max( n_total - (int(pruned_ch / align_val) + 1) * align_val, 0 )

        if retval > 0:
            assert (n_total - retval) % align_val == 0

        return retval

    def mergingPruningMask(self, model):
        """
        동일한 group 상의 module 들에 대해 합리적으로 mask를 합치는 방법 선택 \n
        (channel number -> channel index의 문제)
        
        :param model: 내부 module 추적을 위해 전달받은 model (torch.nn.Module)
        """
        if self.schemaData is not None:
            for parentKey, childKeys in self.schemaData.items():
                parentKey = parentKey[:parentKey.find('_')]
                namesplit = parentKey.split('.')
                parentModule = reduce(getattr, namesplit, model)
                parent_out_mask = parentModule.out_mask
                out_mask_dict = dict()
                out_score_dict = dict()
                out_mask_dict[parentKey] = parent_out_mask
                out_score_dict[parentKey] = parentModule.score

                for childKey in childKeys:
                    childType = childKey[childKey.find('_') + 1:]
                    childKey = childKey[:childKey.find('_')]
                    namesplit = childKey.split('.')
                    childModule = reduce(getattr, namesplit, model)
                    if childType == 'out':
                        out_mask_dict[childKey] = childModule.out_mask
                        out_score_dict[childKey] = childModule.score

                mergeMask = None

                if self.maskMergeType == 'AND':
                    mergeMask = torch.ones_like(parent_out_mask)
                    for _, mask in out_mask_dict.items():
                        mergeMask = torch.logical_and( mergeMask, mask ).float()
                elif self.maskMergeType == 'OR':
                    mergeMask = torch.zeros_like(parent_out_mask)
                    for _, mask in out_mask_dict.items():
                        mergeMask = torch.logical_or( mergeMask, mask ).float()
                elif self.maskMergeType == 'SCORE':
                    mergeMask = torch.ones_like(parent_out_mask)
                    accumScore = torch.zeros_like(parentModule.score)
                    for _, score in out_score_dict.items():
                        accumScore = accumScore + (score/torch.sum(score))
                    elemNum = len(accumScore)
                    n_pruned = min(ceil(self.prune_ratio * elemNum), elemNum - 1)
                    if self.channel_rounding is not None:
                        n_pruned = self.applyChannelRounding(elemNum, n_pruned, self.channel_rounding)
                    prune_idx = accumScore.sort()[1][:n_pruned]
                    mergeMask[prune_idx] = 0

                parentModule.out_mask = mergeMask
                for childKey in childKeys:
                    childType = childKey[childKey.find('_') + 1:]
                    childKey = childKey[:childKey.find('_')]
                    namesplit = childKey.split('.')

                    childModule = reduce(getattr, namesplit, model)

                    if childType == 'in':
                        childModule.in_mask = mergeMask
                    elif childType == 'out':
                        childModule.out_mask = mergeMask
                    elif childType == 'bn':
                        childModule.mask = mergeMask

    def printPruneInfo(self, model):
        """
        mask 생성 및 각 wrapper에의 전달 이후, 전체 model 내의 각 module들에 대한 prunig 정보 출력
        
        :param model: 내부 module 추적을 위해 전달받은 model (torch.nn.Module)
        """
        for name, module in model.named_modules():
            if isinstance(module, PruneConv2d):
                print(f'{name} : original : {module.out_channels}ch -> pruned : { torch.count_nonzero(module.out_mask) }ch ')
            elif isinstance(module, PruneLinear):
                print(f'{name} : original : {module.out_features}ch -> pruned : { torch.count_nonzero(module.out_mask) }ch ')
            elif isinstance(module, PruneBN):
                print(f'{name} : original : {module.num_features}ch -> pruned : { torch.count_nonzero(module.mask) }ch ')


class PruneLinear(nn.Linear):
    """
    nn.Linear를 상속받은 Prunable module
    
    :param origin_linear: 대상 원본 module (nn.Linear)
    :param importance_metric: filter/channel pruning의 importance metric (l1 or l2)
    """
    def __init__(self, origin_linear: nn.Linear, importance_metric):
        super(PruneLinear, self).__init__(origin_linear.in_features,
                                          origin_linear.out_features,
                                          origin_linear.bias != None)
        self.pruningRatio = 0
        self.importance_metric = importance_metric

        self.weight.data = copy.deepcopy(origin_linear.weight)
        self.bias.data = copy.deepcopy(origin_linear.bias)

        self.in_mask = torch.ones(origin_linear.in_features)
        self.out_mask = torch.ones(origin_linear.out_features)

        self.name = None

    def forward(self, x):
        result = super(PruneLinear, self).forward(x)
        out_mask = self.out_mask.view(1, -1).cuda()
        out_mask = out_mask.expand(result.size())
        result = result.mul(out_mask)

        return result

    def getImportanceScore(self):
        """Module 내의 개별 weight에 대한 importance 계산 (L1 or L2 기반)"""
        outCh = self.weight.shape[0]
        self.score = torch.zeros(outCh).cuda()

        if self.importance_metric == 'l1':
            for j in range(outCh):
                # score[j] = torch.sum(torch.abs(self.weight[j, :]))    # deprecated version
                self.score[j] = torch.sum(torch.abs(self.weight[j, :])) / self.in_features   # normailized
        elif self.importance_metric == 'l2':
            for j in range(outCh):
                self.score[j] = torch.sum(torch.square(self.weight[j, :]))

        return self.score

    def resetUnpruneState(self):
        """Module 내의 mask 및 pruning ratio 관련 정보를 초기화"""
        self.out_mask = torch.ones(self.out_features).cuda().flatten()
        self.pruningRatio = 0

    def generateLinear(self, newInCh, prevMask):
        """
        Pruning 적용 이후 고유 architecture를 생성 (nn.Linear)
        
        :param newInCh: 새로운 고유 구조의 input channel 수를 전달 
        :param prevMask: 기존 nonzero input channel index를 파악하기 위해 전달
        """
        newOutCh = torch.count_nonzero(self.out_mask)
        result = nn.Linear(newInCh, newOutCh)

        newOutIdx = 0

        for i in range(self.out_features):
            if self.out_mask[i] == 1:
                if prevMask == None:
                    result.weight.data[newOutIdx, :] = copy.deepcopy(self.weight.data[i, :])
                else:
                    newInIdx = 0
                    for j in range(len(prevMask)):
                        if prevMask[j] == 1:
                            result.weight.data[newOutIdx, newInIdx] = copy.deepcopy(self.weight.data[i, j])
                            newInIdx += 1

                if result.bias is not None:
                    result.bias.data[newOutIdx] = copy.deepcopy(self.bias.data[i])

                newOutIdx += 1

        return result.cuda()


class PruneBN(nn.BatchNorm2d):
    """
    nn.BatchNorm2d를 상속받은 Prunable module
    
    :param origin_bn: 대상 원본 module (nn.BatchNorm2d)
    """
    def __init__(self, origin_bn: nn.BatchNorm2d):
        super(nn.BatchNorm2d, self).__init__(num_features=origin_bn.num_features,
                                             eps=origin_bn.eps,
                                             momentum=origin_bn.momentum,
                                             affine = origin_bn.affine,
                                             track_running_stats = origin_bn.track_running_stats)
        self.bias = copy.deepcopy(origin_bn.bias)
        self.weight = copy.deepcopy(origin_bn.weight)
        self.running_var = copy.deepcopy(origin_bn.running_var)
        self.running_mean = copy.deepcopy(origin_bn.running_mean)
        self.num_batches_tracked = copy.deepcopy(origin_bn.num_batches_tracked)

        self.mask = None
        self.name = None
        
    def generateBN(self, newInputCh):
        """
        Pruning 적용 이후 고유 architecture를 생성 (nn.BatchNorm2d)
        
        :param newInputCh: 새로운 고유 구조의 channel 수를 전달
        """
        result = nn.BatchNorm2d(newInputCh, self.eps, self.momentum, self.affine, self.track_running_stats)
        newOutIdx = 0

        # 새로 생성한 ckpt에서 불러올 경우 >> TypeError: 'torch.Size' object cannot be interpreted as an integer (임시 방편)
        self.num_features = self.num_features[0] if isinstance(self.num_features, torch.Size) else self.num_features

        for i in range(self.num_features):
            if self.mask[i] == 1:
                result.weight.data[newOutIdx] = copy.deepcopy(self.weight.data[i])
                result.bias.data[newOutIdx] = copy.deepcopy(self.bias.data[i])
                result.running_var.data[newOutIdx] = copy.deepcopy(self.running_var.data[i])
                result.running_mean.data[newOutIdx] = copy.deepcopy(self.running_mean.data[i])

                newOutIdx += 1

        return result.cuda()

    def forward(self, x):
        self.mask = self.mask.cuda()
        self.bias.data = self.bias.data * self.mask
        self.weight.data = self.weight.data * self.mask
        self.running_var.data = self.running_var.data * self.mask
        self.running_mean.data = self.running_mean.data * self.mask

        result = super(PruneBN, self).forward(x)

        return result


class PruneConv2d(nn.Conv2d):
    """
    nn.Conv2d를 상속받은 Prunable module
       
    :param origin_bn: 대상 원본 module (nn.BatchNorm2d)
    :param importance_metric: filter/channel pruning의 importance metric (l1 or l2)
    """
    def __init__(self, origin_conv: nn.Conv2d, importance_metric):
        super(PruneConv2d, self).__init__(origin_conv.in_channels,
                                          origin_conv.out_channels,
                                          origin_conv.kernel_size,
                                          origin_conv.stride,
                                          origin_conv.padding,
                                          origin_conv.dilation,
                                          origin_conv.groups,
                                          origin_conv.bias != None)

        self.in_mask = torch.ones(origin_conv.in_channels)
        self.out_mask = torch.ones(origin_conv.out_channels)
        self.importance_metric = importance_metric
        self.pruningRatio = 0
        self.channelConstraints = None
        self.name = None
        #self.weight_original = copy.deepcopy(origin_conv.weight)
        self.weight = copy.deepcopy(origin_conv.weight)
        #self.bias_original = copy.deepcopy(origin_conv.bias)
        self.bias = copy.deepcopy(origin_conv.bias)

    def forward(self, x):
        result = super(PruneConv2d, self).forward(x)
        out_mask = self.out_mask.view(1,-1,1,1).cuda()
        out_mask = out_mask.expand(result.size())
        result = result.mul(out_mask)

        return result

    def setConstraint(self, constraints):
        """
        동일한 이름의 module에 대해 schema dict로 부터 파악한 constraint를 받아와서 내부 메서드에 저장.
        
        :param constraints: schema dict로 부터 파악한 constraint
        """
        self.channelConstraints = constraints

    def generateConv2d(self, newInCh, parentMask):
        """
        Pruning 적용 이후 고유 architecture를 생성 (nn.Conv2d)
        
        :param newInCh: 새로운 고유 구조의 input channel 수를 전달 
        :param prevMask: 기존 nonzero input channel index를 파악하기 위해 전달
        """
        newOutCh = torch.count_nonzero(self.out_mask)
        result = nn.Conv2d(newInCh,
                           newOutCh,
                           self.kernel_size,
                           self.stride,
                           self.padding,
                           self.dilation,
                           newInCh if (self.groups != 1) else self.groups,    # deothwise 감안하여 수정
                           self.bias != None).cuda()
        newOutIdx = 0

        for i in range(self.out_channels):
            if self.out_mask[i] == 1:
                if parentMask == None or (self.groups != 1):    # 첫 conv이거나 depthwise 인 경우
                    result.weight.data[newOutIdx,:,:,:] = copy.deepcopy(self.weight.data[i,:,:,:])
                else:
                    newInIdx = 0

                    for j in range(len(parentMask)):
                        if parentMask[j] == 1:
                            result.weight.data[newOutIdx, newInIdx, :, :] = copy.deepcopy(self.weight.data[i, j, :, :])
                            newInIdx += 1

                if result.bias is not None:
                    result.bias.data[newOutIdx] = copy.deepcopy(self.bias.data[i])

                newOutIdx+=1

        return result.cuda()

    def getImportanceScore(self):
        """Module 내의 개별 Filter weight에 대한 importance 계산 (L1 or L2 기반, 각 output channel의 중요도에 대응)"""
        outCh = self.weight.shape[0]
        self.score = torch.zeros(outCh).cuda()

        if self.importance_metric == 'l1':
            for j in range(outCh):
                # score[j] = torch.sum(torch.abs(self.weight[j, :, :, :]))  # deprecated version
                self.score[j] = torch.sum( torch.abs( self.weight[j,:,:,:] ) ) / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])  # normalized
        elif self.importance_metric == 'l2':
            for j in range(outCh):
                self.score[j] = torch.sum(torch.square(self.weight[j, :, :, :]))

        return self.score

    def resetUnpruneState(self):
        """Module 내의 mask 및 pruning ratio 관련 정보를 초기화"""
        self.out_mask = torch.ones(self.out_channels).cuda()        # applyFirstLastLayer 미적용시, error 발생으로 인한 수정.
        self.pruningRatio = 0


class KETIPrunerStructured():
    """
    주어진 조건들을 기반으로 Structured Pruning을 진행하는 Pruner 클래스
    
    :param model: 대상 원본 model (torch.nn.Module)
    :param args: 전달받은 argument parser 인자
    :param logger: global logger
    """
    def __init__(self, model, args, logger):
        self.originalModel = copy.deepcopy(model)
        self.pruneTargetModule = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
        self.prune_ratio_method = args.prune_ratio_method
        self.prune_ratio = args.prune_ratio
        self.channel_rounding = args.channel_rounding
        self.importance_metric = args.importance_metric
        self.applyFirstLastLayer = args.applyFirstLastLayer
        self.firstLayer = None
        self.lastLayer = None

        #Iterative Pruning Option
        self.iterative = args.iterative
        self.iter_prune_ratio = 0
        self.iter_num = args.iter_num
        # self.shortTermTrainer = Trainer(args, logger, args.st_epochs, args.st_lr, args.st_warmup_epochs, args.st_cooldown_epochs)

        #Self distillation Option
        self.KD = args.KD
        
        self.schema = os.path.join(args.schema_path, args.backbone +'.json')
        self.schemaData = None
        with open(self.schema,'r') as f:
            self.schemaData = json.load(f)

        # Channel Rounding & Aligning Option
        self.maskGenerator = MaskGenerator(args, self.schemaData)

        ## ShortTerm Finetuning 을 위한 학습 Hyper parameter 는 LongTerm과 공유
        ## 향후 변경 필요한지 검토 필요
        self.args = args
        if self.iterative is True:
            assert self.iter_num > 0

    def getNewInputChannel(self, name):
        """
        Schema를 통해 얻은 대상 module(wrapper)의 parent module(wrapper)로부터 channel 갯수 및 mask를 받아온다.
        
        :param name: 새로운 input channel 정보를 얻고자 하는 대상 module의 이름
        :return: newInputCh, prarentMask
        """
        parentKey = None
        prarentMask = None

        # if 'module.' in name:
        #     name = name[name.find('.')+1:]  # or name = name[7:]

        if self.schemaData is not None:
            for key, value in self.schemaData.items():
                for valueItem in value:
                    if valueItem == name:
                        parentKey = key

        if parentKey is not None:
            parentKey = parentKey[:parentKey.find('_')]
            keysplit = parentKey.split('.')
            parentModule = reduce(getattr, keysplit, self.prunedModel)
            newInputCh = torch.count_nonzero(parentModule.mask) if isinstance(parentModule, PruneBN) else torch.count_nonzero(parentModule.out_mask)
            prarentMask = parentModule.mask if isinstance(parentModule, PruneBN) else parentModule.out_mask
        else:
            name = name[:name.find('_')]
            keysplit = name.split('.')
            selfModule = reduce(getattr, keysplit, self.prunedModel)
            newInputCh = selfModule.in_channels

        return newInputCh, prarentMask

    def findInputName(self, name):
        """
        Constraint 저장을 위해 schema를 참고하여 요청한 module이 포함된 group 정보를 받아온다.
        
        :param name: group 정보를 얻고자 하는 대상 module의 이름
        :return: group 이름에서 module 이름 부분만을 반환 ('_out' 제거)
        """
        retval = None
        if self.schemaData is not None:
            for key, value in self.schemaData.items():
                for valueItem in value:
                    idx = valueItem.find('_')
                    if valueItem[:idx] == name:
                        retval = key

        if retval is not None:
            idx = retval.find('_')
            retval = retval[:idx]
        return retval

    def getOutMaskbyKey(self, key):
        """
        prunedModel 상에서 주어진 key에 대응하는 module의 mask를 가져온다.
        
        :param key: mask 요청 대상의 key (module 이름) 
        :return: 대응하는 mask
        """
        keysplit = key.split('.')
        att = reduce(getattr, keysplit, self.prunedModel)
        return att.out_mask

    def getOutChannelConstraintGroup(self, name):
        """
        Constraint 저장을 위해 schema를 참고하여 group 정보를 받아온다.
        
        :param name: group 이름에서 module 이름 부분만 전달 (self.findInputName 참고)
        :return: constraints
        """
        constraints = None
        if self.schemaData is not None:
            tmpName = name +'_out'
            constraints = self.schemaData[tmpName]
        return constraints

    def prune(self, getOriginalArchtecture=False):
        """
        주어진 조건 하에서 Pruning 진행.
            
            - iterative / one-shot
            - masked model / original architecture
        
        :param getOriginalArchtecture: 고유 architecture 생성 여부 (default: fine-tuning 단계에선 False)
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
        final_rounding = self.channel_rounding
        
        for n in range(1, self.iter_num + 1):
            if n == self.iter_num:
                self.maskGenerator.channel_rounding = final_rounding
            else:
                self.maskGenerator.channel_rounding = None

            pr_each_time = pr_step_each_iter * ((1 - pr_step_each_iter) ** (n - 1))
            self.iter_prune_ratio = pr_each_time + self.iter_prune_ratio
           
            prunedModel = self.gen_prunedModel(prevModel, self.iter_prune_ratio)

            if self.KD:
                bestAcc = self.shortTermTrainer.train(prunedModel, teacher_model=prevModel)
            else:
                bestAcc = self.shortTermTrainer.train(prunedModel)

            print(f'iteration {n} best accuracy {bestAcc} ')

            tmp = self.genOriginalArch(prevModel)  
            totalEleNum, zeroEleNum, zero_rate = calculate_zero_ratio(tmp) 
            print(f'Total Element Number: {totalEleNum}, Zero Element Number : {zeroEleNum}, Zero Rate: {zero_rate}')
            prevModel = prunedModel

        return prevModel

    def gen_prunedModel(self, targetModel = None, prune_ratio = None):
        """One-Shot Pruining"""
        if targetModel is None:
            self.prunedModel = copy.deepcopy(self.originalModel)
        else:
            self.prunedModel = copy.deepcopy(targetModel)

        if self.args.model_ema:
            self.prunedModelEMA = copy.deepcopy(self.originalModel)

        convIdx = 0
        allWeight = None

        if prune_ratio is not None:
            self.prune_ratio = prune_ratio

        ## next bn 초기화 코드 추가 필요

        self.prunedModel.eval()

        for name, m in self.prunedModel.named_modules():
            tmp = None
            isBN = False

            if isinstance(m, nn.Conv2d):
                tmp = PruneConv2d(m, self.importance_metric).cuda()
                try:
                    tmp.setConstraint(self.getOutChannelConstraintGroup(name))
                except KeyError:    # Update for Residual Shortcut
                    tmp.setConstraint(self.getOutChannelConstraintGroup(self.findInputName(name)))
                convIdx += 1
            elif isinstance(m, nn.Linear):
                tmp = PruneLinear(m, self.importance_metric).cuda()
            elif isinstance(m, nn.BatchNorm2d):
                tmp = PruneBN(m).cuda()
                isBN = True

            if tmp is not None:
                tmp.name = name
                if isBN is False:
                    if convIdx == 1:
                        self.firstLayer = tmp

                    if self.prune_ratio_method == 'global':
                        if allWeight == None:
                            allWeight = tmp.getImportanceScore()
                        else:
                            allWeight =torch.cat( (allWeight, tmp.getImportanceScore() ), dim = 0 )
                    elif self.prune_ratio_method == 'uniform':
                        self.maskGenerator.genOutputMaskbyPruningRatio(tmp, self.prune_ratio)

                    #zeroRatio =  (torch.numel(tmp.out_mask) - torch.count_nonzero(tmp.out_mask)) / torch.numel(tmp.out_mask) * 100.0
                    #print(f'{name} prune ratio : {zeroRatio}')

                namesplit = name.split('.')
                att = reduce(getattr, namesplit[:-1], self.prunedModel)
                setattr(att, namesplit[-1], tmp)
                self.lastLayer = tmp

        if self.prune_ratio_method == 'global':
            elemNum = len(allWeight)
            val, idx  = allWeight.sort()
            globalTrh = val[ int(elemNum * self.prune_ratio) ]

            for name, m in self.prunedModel.named_modules():
                if isinstance(m, PruneConv2d) or isinstance(m, PruneLinear):
                    self.maskGenerator.genOutputMaskbyThreshold(m, globalTrh)

            # for name, m in self.prunedModel.named_modules():
            #     if isinstance(m, PruneBN):
            #         prevConvKey = self.findInputName(name)
            #         prevConvMask = self.getOutMaskbyKey(prevConvKey)
            #         m.mask = prevConvMask

        if self.KD and self.maskGenerator.prune_ratio != self.prune_ratio:  # KD 적용시 갱신 필요함.
            self.maskGenerator.prune_ratio = self.prune_ratio

        self.maskGenerator.mergingPruningMask(self.prunedModel)
        self.lastLayer.resetUnpruneState()

        if not self.applyFirstLastLayer:
            self.firstLayer.resetUnpruneState()

        # if self.args.local_rank == 0:
        self.maskGenerator.printPruneInfo(self.prunedModel)

        return self.prunedModel

    def genOriginalArch(self, prunedModel):
        """
        취득한 pruned model의 고유 architecture 생성
        
        :param prunedModel: masked model (원본 baseline과 동일한 규격)
        :return: retModel (=pruned model with its original architecture)
        """
        prunedModel.eval()
        if isinstance(prunedModel, DistributedDataParallel) or isinstance(prunedModel, DataParallel):
            retModel = nn.parallel.replicate(prunedModel, [self.args.device])[0]._modules['module']
        else:
            retModel = nn.parallel.replicate(prunedModel, [self.args.device])[0]
        # retModel = nn.parallel.replicate(prunedModel, [self.args.device])[0]

        for name, m in retModel.named_modules():
            tmp = None
            if isinstance(m, PruneConv2d):
                zeroRatio = (torch.numel(m.out_mask) - torch.count_nonzero(m.out_mask)) / torch.numel(m.out_mask) * 100.0
                print(f'Final {name} prune ratio : {zeroRatio}')

                newInput, parentMask = self.getNewInputChannel(name+'_in')
                tmp = m.generateConv2d(newInput, parentMask)

            elif isinstance(m, PruneLinear):
                zeroRatio = (torch.numel(m.out_mask) - torch.count_nonzero(m.out_mask)) / torch.numel(m.out_mask) * 100.0
                print(f'Final {name} prune ratio : {zeroRatio}')

                newInput, parentMask = self.getNewInputChannel(name + '_in')
                tmp = m.generateLinear(newInput, parentMask)

            elif isinstance(m, PruneBN):
                newInput, parentMask = self.getNewInputChannel(name + '_bn')
                tmp = m.generateBN(newInput)

            if tmp is not None:
                namesplit = name.split('.')
                att = reduce(getattr, namesplit[:-1], retModel)
                setattr(att, namesplit[-1], tmp)

        return retModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="efficientnet_es_postech")
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--minKeepChannelRatio", type=float, default=0.15)                                                          # 성능 보호를 위한 Lower bound 지정
    parser.add_argument('--maskMergeType',help='channel mask merging type', choices=['AND', 'OR', 'SCORE'], default='SCORE')        # Structured : 모듈 간 mask 병합 방식 선택

    #weight pruning parameter
    parser.add_argument("--prune_ratio_method", type=str, choices=['global','uniform','adaptive'], default='uniform')
    parser.add_argument("--importance_metric", type=str, choices=['l1', 'l2', 'entropy'], default='l1')
    parser.add_argument("--prune_ratio", type=float, default=0.7)
    parser.add_argument("--applyFirstLastLayer", action='store_true')
    parser.add_argument("--channel_rounding", type=int, help='manually set channel rounding', default=None)                         # 채널 배수 (OPENEDGES: 8)

    #For iterative pruning method
    parser.add_argument("--iterative", action='store_true')
    parser.add_argument("--iter_num", type=int, default=10)

    #Short term trainer parameter for interative method
    parser.add_argument('--data_path', metavar='DIR', help='path to dataset')
    parser.add_argument('-b', '--batch-size', '--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size_prune', type=int, default=64)
    parser.add_argument('--st_epochs', default=2, type=int, metavar='N', help='short term finetune epochs')
    parser.add_argument('--st_cooldown_epochs', type=int, default=1, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--st_warmup-epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--st_lr', '--st-learning-rate', default=0.01, type=float, metavar='LR', help='short term initial learning rate',
                        dest='st_lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "step"')
    parser.add_argument("--KD", action='store_true')                        # Iterative KD 활용 여부
    parser.add_argument('--device', default='cuda:0')

    #Parameter for logger
    parser.add_argument('--Exps_Dir', metavar='DIR', help='path to output', default='/home/keti/jp_ubuntu_shared')
    parser.add_argument('--project', type=str, default="")
    parser.add_argument('--screen_print', action="store_true")
    parser.add_argument('--schema_path', metavar='DIR', help='path to schema path', default='schema')             # Arch. analyze 결과를 저장 및 불러올 경로 
    parser.add_argument('--model-ema', action='store_true', default=False,                                  # DDP 사용시, EMA 적용 여부 
                    help='Enable tracking moving average of model weights')
    args = parser.parse_args()
    args.num_classes = 100
    # logger = Logger(args)

    model = build_backbone(args)
    model.cuda()
    print(get_n_params_(model))
    pruner = KETIPrunerStructured(model,args, None)

    pruneModel = pruner.prune(True)
    print(get_n_params_(pruneModel))
