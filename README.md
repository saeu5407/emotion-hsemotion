# emotion-hsemotion

AffectNet SOTA 모델인 Effectnet 기반 멀티 태스크 러닝 모델(hsemotion) 데모 및 ONNX 구현<br>
<br>
`enet_b0_8_va_mtl.pt`를 기반으로 4가지 output을 출력하는 ONNX모델 생성 완료<br>
- embedding vector : (1, 1280)
- expression(emotion) : (1, 8)
- valence : (1, )
- arousal : (1, )

## Type

Weight : enet_b0_8_va_mtl.pt<br>
Backbone : Effectnet_B0<br>
Input Image Shape(Face Croped) : 224, 224<br>
Output : embedding vector, emotion(8 category), valence, arousal<br>

## Demo
<img width="80%" src="https://github.com/saeu5407/emotion-hsemotion/blob/main/demo.gif"/>
```
# Run
python3 ./src/HSEmotion.py
```