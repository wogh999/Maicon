# 국방 AI 경진대회 코드 사용법
- 위대한감자팀: 이재호, 박서정, 오승훈, 정유빈
- 닉네임 : 파키케투스, Maeve, 파인애플, 으넴짱짱


# 핵심 파일 설명
  - 학습 데이터 경로: './data/train/x/' + os.path.basename(self.x_paths[id_]) #Image
                     './data/train/y/' + os.path.basename(self.y_paths[id_]) #Mask
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: 
      architecture: Unet
      encoder: timm-efficientnet-b0 #timm-regnety_016
      encoder_weight: noisy-student #imagenet
      depth: 5
      n_classes: 4
      activation: null

  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개
      architecture: DeepLabV3Plus
      encoder: timm-efficientnet-b0 
      encoder_weight: noisy-student 
      depth: 5
      n_classes: 4
      activation: null

      architecture: DeepLabV3Plus
      encoder: timm-efficientnet-b0 
      encoder_weight: noisy-student 
      depth: 5
      n_classes: 4
      activation: sigmoid

      architecture: DeepLabV3Plus
      encoder: resnet101 
      encoder_weight: noisy-student
      depth: 5
      n_classes: 4
      activation: null

  - 학습 실행 스크립트: train.py
  - 학습 메인 코드: train.py
  - 테스트 실행 스크립트: predict.py
  - 테스트 메인 코드: predict.py
  - 테스트 이미지, 마스크 경로: './data/test/x/' + os.path.basename(self.x_paths[id_])
  - 테스트 결과 이미지 경로: "./subtest_v3_1/" + os.path.basename(filename_)

## 코드 구조 설명
- segmentation_models_pytorch as smp을 backend로 사용하여 학습 및 테스트
    - 최종 사용 모델 : segmentation_models_pytorch에서 제공하는 DeepLabV3Plus 모델
   
- **최종 제출 파일 : subtest_v3.zip**
- **학습된 가중치 파일 : C:\Users\User\Desktop\ai\baseline\results\train\20221108_235819\model.pt**

## 주요 설치 library
- torch==1.13.0
- albumentations
- python==3.7

# 실행 환경 설정

  - 소스 코드 및 conda 환경 설치
    conda create -n baseline python=3.7
    pip install numpy
    torch==1.13.0 version internet에서 install
    pip install matploblib
    pip install scikit-learn
    pip install albumentation
    pip install pickle5
    pip install matplotlib
    pip install pandas
    pip install tqdm
    pip install segmentation_models_pytorch

# 학습 실행 방법

  - 학습 데이터 경로 설정
    - ./config/train.yaml  내의 경로명을 실제 학습 환경에 맞게 수정
      data_root_dir: C:\Users\User\Desktop\ai\baseline\data\train\x\*.png   #image
                     C:\Users\User\Desktop\ai\baseline\data\train\y\*.png   #mask
                     # 학습 데이터 절대경로명
      out_root_dir: C:\Users\User\Desktop\ai\baseline\results\train\20221108_235819\ #학습 결과물 절대경로명
      tb_dir: C:\Users\User\Desktop\ai\baseline\results\train\20221108_235819\train.log  # 학습 로그 절대경로명

  - 학습 스크립트 실행
    python train.py
    
  - 학습 스크립트 내용
    python train.py
    image size는 원본 이미지 비율을 유지하여 width: 1280, height: 640로 사용함
    transform을 적용하기 위해 albumentation 라이브러리를 설치한 후,
    Randowshadow,ColorJitter,VerticalFlip,CLAHE,RandomFog,RGBShift,Normalize,ToTensorV2 등을 적용함 -> 이후 성능 저하로 transform을 제거함
    데이터 augmentation 적용 후에 성능이 잘 나오지 않아 학습 데이터 수 증가를 위해 train/val 비율을 9:1로 나누어 진행함
    model encoder에 resnet101, resnet34, timm-efficientnet-b0, timm-efficientnet-b1, timm-efficientnet-b3, efficientnet-b0 등을 적용해 봄 -> timm-efficientnet-b0 선택
    encoder weight는 noisy-student 와 imagnet을 적용해본 결과 noisy-student으로 선택함
    activation은 모델 자체에서 relu가 사용되어 activation을 추가하지 않았음. -> 실제로 sigmoid를 사용한 결과 성능이 하락함
    loss 부분에서는 학습 과정에서 iou1, iou2, iou3 중 iou3 값이 가장 낮아 iou3에 가중치를 더 주었지만, 성능변화에 큰 변화가 없어 그대로 진행함
  
# 테스트 실행 방법

  - 테스트 스크립트 실행
    python predict.py 

  - 테스트 스크립트 내용
    python predict.py
    

