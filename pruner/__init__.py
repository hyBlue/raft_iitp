"""
KETI Pruner 2023 : Pruner

*** Author : Jonghee Park & Hojong Shin @ Korea Electronics Technology Institute
*** E-mail : jpark19@keti.re.kr
"""  
from pruner.keti_pruner_weight import KETIPrunerWeight
from pruner.keti_pruner_structured import KETIPrunerStructured


__PRUNER__ = {
    'str': KETIPrunerStructured,
    'wgt': KETIPrunerWeight,
}


def init_pruner(model, args, logger):
    """
    argument 중 지정된 pruner type을 불러와 초기화하여 반환해준다. (args.pruner)
    
        - str : Structured prunung (Channel/Filter)
        - wgt : Unstructured pruning (Weight)
        - jnt : Joint pruning (with manually-set config., work in progress)
    
    :param model: 대상 원본 model (torch.nn.Module) -> pruner 초기화 인자
    :param args: 전달받은 argument parser 인자      -> pruner type 선택 및 pruner 초기화 인자 포함
    :param logger: global logger                   -> pruner 초기화 인자
    :return: 초기화된 pruner 
    """
    return __PRUNER__[args.pruner](model, args, logger)