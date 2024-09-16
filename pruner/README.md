# Pruner

## Base Schema

네트워크 model을 읽고 architecture를 파악하여 layer 간의 종속성을 기록한 schema 파일.

- 예시 : `vgg9_bn.json`
    
    ```json
    {
      "module.features.0_out": [
        "module.features.1_bn",
        "module.features.0_out",
        "module.features.3_in"
      ],
      "module.features.3_out": [
        "module.features.4_bn",
        "module.features.3_out",
        "module.features.7_in"
      ],
      "module.features.7_out": [
        "module.features.8_bn",
        "module.features.7_out",
        "module.features.10_in"
      ],
      "module.features.10_out": [
        "module.features.11_bn",
        "module.features.10_out",
        "module.features.14_in"
      ],
      "module.features.14_out": [
        "module.features.15_bn",
        "module.features.14_out",
        "module.features.17_in"
      ],
      "module.features.17_out": [
        "module.features.18_bn",
        "module.features.17_out",
        "module.classifier.1_in"
      ],
      "module.classifier.1_out": [
        "module.classifier.4_in"
      ],
      "module.classifier.4_out": [
        "module.classifier.6_in"
      ],
      "module.classifier.6_out": []
    }
    ```
    

## Pruner

Structured Pruning(`KETIPunerStructured`) 및 Weight Pruning(`KETIPunerWeight`) 등을 지원하고 있다.

- `KETIPunerStructured`
    - Mask Generation : 아래 여러 조건을 고려하여 structured pruning mask를 생성한다.
        - Schema를 기반으로 constraint를 생성하여 이어진 부분에 대한 channel 수 규격을 맞춘다.
        - Mask Merge Type : Residual이 포함된 경우, 합리적으로 mask를 합치는 방법 선택. (`AND`, `OR`, `SCORE`)
        - Pruning Ratio Type : Layer-wise sparsity 선택 (`global`, `uniform`, `adaptive`)
        - Importance Type : Filter weight의 L1 혹은 L2 기반 importance. (`l1`, `l2`, `entropy`)
        - Channel Rounding : 일부 가속기에 맞춰 pruning을 적용하고 남은 channel 수의 배수를 조정한다. (`32`, `64`)
    - Prunable Wrapper : nn.Conv2d, nn.BatchNorm2d, nn.Linear를 상속받은 prunable module 생성
        - Importance score 측정.
        - Forward : 생성된 mask를 적용하여 forward 진행한다.
    - Original Architecture Generation : Pruned model의 고유 architecture model를 생성한다. (실제로 깎인 구조를 의미한다.)
- `KETIPunerWeight`
    - Prunable Wrapper : nn.Conv2d, nn.Linear를 상속받은 prunable module 생성
        - Mask Generation : 아래 여러 조건을 고려하여 structured pruning mask를 생성한다.
            - Pruning Ratio Type : Layer-wise sparsity 선택 (`global`, `uniform`, `adaptive`)
            - Importance Type : 개별 weight의 L1 혹은 L2 기반 importance. (`l1`, `l2`, `entropy`)
        - Forward : 생성된 mask를 적용하여 forward 진행한다.
    - Original Architecture Generation : Pruned model의 고유 architecture model를 생성한다. (실제로 깎인 구조를 의미한다.)
