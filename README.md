# emotion-hsemotion

AffectNet SOTA 모델인 Effectnet 기반 멀티 태스크 러닝 모델(hsemotion) 구현

 
## Type

Weight : enet_b0_8_va_mtl.pt<br>
Backbone : Effectnet_B0<br>
Input Image Shape(Face Croped) : 224, 224<br>
Output : emotion(8 category), valence, arousal<br>

## Demo
<img width="80%" src="https://github.com/saeu5407/emotion-hsemotion/blob/main/demo.gif"/>
```
# Run
python3 ./src/HSEmotion.py
```