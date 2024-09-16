import torch
import torch.nn as nn


def calculate_zero_ratio(model):
    """
    model 내의 named_modules 중 nn.Conv2d 및 nn.Linear 정보를 읽어 아래 정보를 반환한다.
    
        - totalElemNum : model의 총 학습 가능한 parameter 수
        - totalZeroNum : model의 총 zero parameter 수
        - Zero Ratio : model의 총 학습 가능한 parameter들 중 zero parameter의 비율(%)
    
    :param model: 내부 module 추적을 위해 전달받은 model (torch.nn.Module)
    :return: totalElemNum, totalZeroNum, Zero Ratio 
    """
    totalElemNum = 0
    totalNonZeroNum = 0
    totalZeroNum = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            for _, param in module.named_parameters():
                totalNonZeroNum += torch.count_nonzero(param.clone())
                totalElemNum += param.numel()
                totalZeroNum += ( param.numel() - param.nonzero().size(0) )

    print(f'totalElemNum : {totalElemNum}, totalNonZeroNum : {totalNonZeroNum}, totalZeroNum : {totalZeroNum}')
    print(f'Zero Ratio : {totalZeroNum/totalElemNum*100}%')

    return totalElemNum, totalZeroNum, totalZeroNum/totalElemNum*100
