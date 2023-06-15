# hsemotion-onnx

AffectNet SOTA 모델인 Effectnet 기반 멀티 태스크 러닝 모델(Hsemotion) 데모 및 ONNX 구현<br>
악간의 변경점이 있습니다. Output을 4가지로 변형하였습니다.<br><br>
`enet_b0_8_va_mtl.pt`를 기반으로 4가지 output을 출력하는 ONNX모델 생성 완료<br>
- embedding vector : (1, 1280)
- expression(emotion) : (1, 8)
- valence : (1, )
- arousal : (1, )

hsemotion link : https://github.com/HSE-asavchenko/face-emotion-recognition

## Type

Weight : enet_b0_8_va_mtl.pt<br>
Backbone : Effectnet_B0<br>
Input Image Shape(Face Croped) : 224, 224<br>
Output : embedding vector, emotion(8 category), valence, arousal<br>

## Demo
<img width="80%" src="https://github.com/saeu5407/hsemotion-onnx/blob/main/demo.gif"/>
<br>

```
# Run
python3 demo.py
```

## Research papers

If you use our models, please cite the following papers:
```BibTex
@inproceedings{savchenko2021facial,
  title={Facial expression and attributes recognition based on multi-task learning of lightweight neural networks},
  author={Savchenko, Andrey V.},
  booktitle={Proceedings of the 19th International Symposium on Intelligent Systems and Informatics (SISY)},
  pages={119--124},
  year={2021},
  organization={IEEE},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@inproceedings{Savchenko_2022_CVPRW,
  author    = {Savchenko, Andrey V.},
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior on Mobile Devices Using EfficientNets},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022},
  pages     = {2359-2366},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@inproceedings{Savchenko_2022_ECCVW,
  author    = {Savchenko, Andrey V.},
  title     = {{MT-EmotiEffNet} for Multi-task Human Affective Behavior Analysis and Learning from Synthetic Data},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV 2022) Workshops},
  pages={45--59},
  year={2023},
  organization={Springer},
  url={https://arxiv.org/abs/2207.09508}
}
```


```BibTex
@article{savchenko2022classifying,
  title={Classifying emotions and engagement in online learning based on a single facial expression recognition neural network},
  author={Savchenko, Andrey V and Savchenko, Lyudmila V and Makarov, Ilya},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/9815154}
}
```
