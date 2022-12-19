# 국방 AI 경진대회 코드 사용법
- Deep Sleeping 팀
- 배준익(Heatz), 김주영(TalkingPotato), 서강민(kangrnin), 이찬(초콜릿맛프로틴)

# https://github.com/megvii-research/NAFNet



​
# 핵심 파일 설명
  ## 폴더 설명 
  - 학습 및 시험 데이터 경로: `/workspace/junik/dataset/fold_0.pkl`
  - 추론한 결과 이미지를 저장하는 경로: `../result`
  - 모델의 가중치를 저장하는 경로: `./save` 
  - 코드 파일 경로: `/Final_Submission/image-restoration-main` 
​
  ## /Final_Submission/image-restoration-main 파일 설명 
  - 학습 메인 파일: `./train.py`
  - 테스트 메인 파일: `./test.py`
  - 설정 파일: `./config.py`
  
  - 신경망 모델 파일: `./model/naf_net.py`
  - 데이터셋 클래스 파일: `./data/dataset.py` 
  - 손실함수 클래스 파일: `./module/loss.py`
  - 데이터 증강/변환 관련 파일: `./module/transform.py`
  - 기타 함수 파일: `./module/utils.py`
​
## 코드 구조 설명
- `./model/naf_net.py` 
  - 신경망 모델은 U-Net의 형태를 띄고 있으며 마지막 디코더 블럭의 출력값을 컨벌루션하여 복원된 이미지를 출력합니다
  - NAFNet 아키텍쳐를 사용했습니다 
  (https://github.com/megvii-research/NAFNet)
​
- `./data/dataset.py` 
  - PyTorch의 Dataset 클래스를 상속하는 커스텀 데이터셋 클래스입니다
  - __getitem__ 메서드가 호출되면 해당하는 인덱스의 경로에서 이미지를 불러온후 RGB 채널의 형태로 변환합니다 
  - 노이즈 이미지는 정규화를 진행하여 float 형태의 텐서로 변환하며 GT 이미지는 파이토치 형태로 체널 순서만 변경 후 출력합니다
  
- `./module/loss.py`
  - 학습에 사용한 손실함수인 PSNRLoss 를 정의하였습니다. 손실은 모델의 출력값에 전처리 과정의 정규화를 역으로 진행하여 복원한 이미지에 대해서 구합니다. 
​
- `./module/transform.py`
  - 학습 또는 시험 데이터에서 쓰이는 실시간 변환을 정의하였습니다
​
- `./config.py`
  - 여러 하이퍼파라미터를 편의를 위해 Dict 형태로 불러올 수 있게 하였습니다
​
- `./train.py`
  - 훈련 데이터셋 및 검증 데이터셋을 준비합니다. 검증 데이터셋은 훈련 데이터셋에서 5% 만큼을 분리하여 구성합니다.
  - 모델 훈련에 필요한 여러 컴포넌트를 생성합니다. 각각 optimizer, scheduler, criterion (loss function), metric, scaler (가중치 조정) 입니다 
  - Train Loop 를 iter 만큼 진행합니다 
    - 훈련 데이터 미니 배치에 대해서 loss 값을 구합니다. 
    - 정해진 스텝 수마다 검증 데이터 셋에 대하여 성능을 평가합니다. 평가는 validate 함수를 통해 psnr을 계산하여 진행하며 이 값이 
      기존보다 클 경우 모델의 가중치를 중간 저장합니다 
      
- `./test.py`
  - 학습된 가중치를 바탕으로 모델을 불러와 테스트 셋에 대해 예측을 수행합니다. 출력값은 역정규화 과정을 거친 후 0 ~ 255 사이에 매핑될 수 있게 클립핑 및 반올림 과정을 거치고 저장됩니다. 이때 출력 이미지가 필요 이상으로 저조도로 복원되는 오류를 보정하기 위해 훈련 데이터셋의 확률분포에 따르게끔 통계량이 보정됩니다. 
  
​
- **최종 제출 파일 : result.zip**
- **학습된 가중치 파일 : ./save/naf_net.pth**
​
## 1. 아나콘다 가상환경 설정 

conda activate KJY

## 2. 필요한 모듈 설치

albumentations            1.3.0                    pypi_0    pypi
numpy                     1.23.1          py310h1794996_0    anaconda
opencv-python             4.5.5.62                 pypi_0    pypi
pytorch                   1.12.1          py3.10_cuda11.3_cudnn8.3.2_0    pytorch
torchmetrics              0.10.3             pyhd8ed1ab_0    conda-forge
tqdm                      4.64.1             pyhd8ed1ab_0    conda-forge

mkdir data result save
```
​
먼저 data 구성의 경우 /workspace/data를 그대로 사용했습니다.
```
이후
```
python3 train.py
```
​
학습 후 save/xxxxxxxx.pth가 생성된 것을 확인  
save/xxxxxxxx.pth를 save/naf_net.pth로 이름 변경
​
​
### 추론
```
python3 test.py
```
​
추론 후 result/에 output 파일이 저장된 것을 확인
```
​
```
​