def export_onnx(model, img_size, checkpoint):
    """
    전달받은 model을 onnx 파일로 저장해준다. (full-dense, pruned 무관) 

        - PyTorch 의 hooking function 을 이용하여 임의의 input 을 inference 하여 수행한다.
        - instance key 값을 기준으로 앞/뒤 레이어의 종속성을 생성하고, schema 생성 후에는 hook 이 자동으로 해제된다.
        - json 형식의 파일로 결과를 저장한다. (경로 : {$PROJECT_ROOT}/schema/{$BACKBONE}.json)

    :param model: network model (torch.nn.Module)
    :param img_size: dummy input을 생성할 image size (int : 가로, 세로 같다고 가정)
    :param checkpoint: onnx 파일 저장 경로 설정을 위한 원본 checkpoint(*.pth) 경로 (str)
    """
    import torch.onnx
    model.eval().cuda()
    x = torch.randn(1, 3, img_size, img_size, requires_grad=True).cuda()
    output = checkpoint.replace('.pth', '.onnx')
    # dummy_out = model(dummy_x)
    torch.onnx.export(model,  # 실행될 모델
                      x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      output,  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=10,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output'],  # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                    'output': {0: 'batch_size'}})
    print(f"onnx file exported to {output}")