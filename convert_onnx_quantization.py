import os
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from thop import profile
from torchvision import transforms

if not os.path.isfile("hsemotion1.onnx"):

    model_path = os.path.join(os.getcwd().split('/src')[0], 'models/enet_b0_8_va_mtl.pt')
    net = torch.load(model_path, map_location=torch.device('cpu')) # timm==0.6.5
    net.eval()

    class HSEMotion(nn.Module):

        def __init__(self, net):
            super(HSEMotion, self).__init__()
            net.eval()
            self.backbone = net

        def forward(self, x):
            x = self.backbone(x) # 1, 1280, 7, 7 | 0.7초 소요
            return {'exp': x[:, :8], 'valence': x[:, -2], 'arousal': x[:, -1]}

    model = HSEMotion(net)

    # 동적 양자화
    '''양자화 튜토리얼 https://tutorials.pytorch.kr/recipes/quantization.html'''
    '''동적 양자화 및 정적 양자화 차이 https://brunch.co.kr/@bfcced03a6054df/3'''
    model_dynamic_quantized = torch.quantization.quantize_dynamic(model,
                                                                  qconfig_spec={torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)

    # 정적 양자화
    if False:
        backend = "qnnpack"
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.quantization.prepare(model, inplace=False)
        model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

    # 모델 입력 텐서 생성
    dummy_input = torch.randn(1, 3, 224, 224)

    # FLOPs 계산
    # flops, params = profile(model_dynamic_quantized, inputs=(dummy_input,))
    # print(f"FLOPs: {flops}")  # 3.6e+8
    # print(f"Parameters: {params}")  # 3.9e+6

    # 모델 변환
    output_names = ['expression', 'valence', 'arousal']
    torch.onnx.export(model_dynamic_quantized,  # 실행될 모델
                      dummy_input,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      "/Users/dkcns/PycharmProjects/hsemotion-onnx/hsemotion_quantization.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=17,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=output_names,  # 모델의 출력값을 가리키는 이름
                      verbose=True
                      )

# ONNX 모델 로드
onnx_model_path = "hsemotion_quantization.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# 입력 텐서 생성
input_name = session.get_inputs()[0].name
input_data = np.ones((1, 3, 224, 224))

input_data = input_data.astype(np.float32)

# 예측 수행
output_names = [output.name for output in session.get_outputs()]
outputs = session.run(output_names, {input_name: input_data})
print(outputs[0].shape)

# onnx to ort
# python -m onnxruntime.tools.convert_onnx_models_to_ort hsemotion.onnx

