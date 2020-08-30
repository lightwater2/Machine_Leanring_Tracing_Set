# Machine_Leanring_Tracing_Set
### 파이썬을 이용하여 머신러닝을 할 때 컴파일 시 코드, 히스토리, 모델, 결과예측을 저장해 주는 소스입니다.
### 환경(Environment)
`python 3.7.9`  
`tensorflow 2.3.0 (cuda 10.1, cuDNN 7.6.5, GTX1650 Super)`  
`keras 2.4.3`
---

## 파일 구조
```
dacon.io.SMILES_AI
├──lib
│   └──FileManager.py : 파일 관리용 자체 라이브러리
└──Mnist
    ├──archive : 정확도 및 컴파일 시점 코드 저장
    ├──input : 훈련용 DataSet
    ├──models : 훈련된 모델 저장
    ├──predicts : 예측 결과 저장
    └──train.py : 실행할 훈련 소스
```