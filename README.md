# Integrated Gradients 기반 트리거 삽입을 통한 효율적 은닉성 백도어 공격 기법
 
### Preliminaries
```
pip install -r requirements.txt
```

### Code Execution
```
python main.py
```

### gen_xai.py
CIFAR-10 혹은 STL-10 데이터셋에 대해 합성곱 신경망 모델을 학습하고, 주어진 합성곱 신경망에서 정상 데이터 각각에 대한 XAI value를 추출하는 스크립트.

### resnet50_models.py
ImageNet에서 사전학습된 ResNet50 모델 아키텍처(ResNet50에서 요구되는 입력 크기를 조절하기 위하여 UpSampling 사용)

### utils.py
CIFAR-10 혹은 STL-10 데이터셋 로드 등 유틸리티 스크립트

### main.py
학습 데이터셋에 Trigger가 삽입된 데이터를 포함하고, 이를 학습한 심층신경망은 Trigger가 삽입된 데이터에 대하여 오분류를 발생. XAI 기법 중 하나인 IG를 활용하여 트리거 삽입을 통한 은닉성 백도어 공격 기법 스크립트
- `report/{DATASET}_{TRIGGER_METHOD}_adv` 파일 생성
  - Poisoning attack을 수행하였을 때, Backdoor 데이터 공격 정확도
- `report/{DATASET}_{TRIGGER_METHOD}_cln` 파일 생성
  - Poisoning attack을 수행하였을 때, 정상 데이터 분류 정확도