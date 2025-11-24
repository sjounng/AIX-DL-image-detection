# AI 생성 이미지 판별 프로젝트
## AI-Generated Image Detection Project

---

## 📋 목차 (Table of Contents)

- [프로젝트 개요](#프로젝트-개요)
- [팀원 소개](#팀원-소개)
- [I. Proposal](#i-proposal)
- [II. Datasets](#ii-datasets)
- [III. Methodology](#iii-methodology)
- [IV. Evaluation & Analysis](#iv-evaluation--analysis)
- [V. Related Work](#v-related-work)
- [VI. Conclusion](#vi-conclusion)
- [발표 영상](#발표-영상)

---

## 프로젝트 개요

이 프로젝트는 AI로 생성된 이미지와 실제 이미지를 구분하는 딥러닝 모델을 개발하고 분석합니다.

---

## 팀원 소개 (Members)

| 이름  | 학과         | 이메일                  | 역할            |
|-----|------------|----------------------|---------------|
| 송준우 | 정보시스템학과    | jwsong5160@gmail.com | 모델 구현, 블로그 관리 |
| 전용현 | 데이터사이언스학과  | example2@university.edu | 모델 구현         |
| 서채원 | 컴퓨터소프트웨어학과 | example3@university.edu | 모델 구현         |


---

## I. Proposal

### Option A 선택: 데이터셋 분석 및 AI/ML 기법 적용

### 1. Motivation (동기)

최근 Stable Diffusion, DALL-E, Midjourney 등 생성형 AI의 발전으로 고품질 이미지 생성이 가능해졌습니다. 하지만 이러한 기술의 발전은 다음과 같은 문제들을 야기합니다:

- **허위 정보 확산**: AI로 생성된 가짜 뉴스 이미지
- **저작권 침해**: 실제 작품으로 위장한 AI 생성 작품
- **신뢰성 문제**: 온라인 콘텐츠의 진위 여부 판단 어려움

따라서 AI 생성 이미지를 자동으로 탐지하는 기술의 필요성이 증가하고 있습니다.

### 2. What do you want to see at the end? (목표)

이 프로젝트를 통해 다음을 달성하고자 합니다:

1. **분류 모델 구축**: CNN 기반 딥러닝 모델로 AI 생성 이미지와 실제 이미지를 구분
2. **성능 분석**: 다양한 모델 아키텍처 비교 및 평가
3. **특징 시각화**: AI 생성 이미지의 특징을 Grad-CAM 등으로 시각화
4. **실용적 활용**: 실제 환경에서 활용 가능한 판별 시스템 제시

---

## II. Datasets

### 데이터셋 정보

- **데이터셋 이름**: My Sampled Art Dataset 40k
- **출처**: [Kaggle - My Sampled Art Dataset 40k](https://www.kaggle.com/datasets/mkevinrinaldi/my-sampled-art-dataset-40k/data)
- **데이터 크기**: 약 40,000장의 이미지
- **구성**: 
  - AI 생성 이미지 (AI-generated artwork)
  - 실제 예술 작품 이미지 (Real artwork)

### 데이터 구조

```
data/raw/
├── FAKE/                  # AI 생성 이미지 (~20,000장)
│   ├── 0-100086213-128066_AI_SD_art_nouveau.jpg
│   ├── 0-100157086-913505_AI_LD_art_nouveau.jpg
│   └── ...
└── REAL/                  # 실제 예술 작품 (~20,000장)
    ├── a--y---jackson_barns-1926_Human_Post_Impressionism.jpg
    ├── a--y---jackson_grey-day-laurentians-1928_Human_Art_Nouveau_Modern.jpg
    └── ...
```

- **FAKE 폴더**: AI로 생성된 이미지 (Stable Diffusion, Latent Diffusion 등)
- **REAL 폴더**: 실제 예술가가 그린 작품 (Post-Impressionism, Art Nouveau 등)

### 데이터 전처리

1. **이미지 리사이징**: 모든 이미지를 224x224 픽셀로 통일
2. **정규화 (Normalization)**: 픽셀 값을 [0, 1] 범위로 스케일링
3. **데이터 증강 (Data Augmentation)**:
   - Random Horizontal Flip
   - Random Rotation (±15도)
   - Random Brightness/Contrast 조정
4. **데이터 분할**: 
   - Training: 70%
   - Validation: 15%
   - Test: 15%

### 데이터 다운로드 및 준비

#### 방법 1: Kaggle API 사용 (자동)

```bash
# Kaggle API를 사용한 데이터셋 다운로드
kaggle datasets download -d mkevinrinaldi/my-sampled-art-dataset-40k
unzip my-sampled-art-dataset-40k.zip -d ./data/raw
```

#### 방법 2: 수동 다운로드 (추천)

1. [Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/mkevinrinaldi/my-sampled-art-dataset-40k/data) 방문
2. `Download` 버튼 클릭 (Kaggle 로그인 필요)
3. 다운로드한 `archive.zip` 파일을 압축 해제:
   ```bash
   unzip ~/Downloads/archive.zip -d ./data/raw/
   ```
4. 데이터 구조 확인:
   ```bash
   ls data/raw/
   # 출력: FAKE  REAL
   ```

### 데이터 특성 분석

- 이미지 해상도 분포
- 클래스별 이미지 수
- 색상 분포 분석
- 이미지 품질 평가

*(실제 분석 결과는 프로젝트 진행 중 업데이트 예정)*

---

## III. Methodology

### 1. 알고리즘 선택 (Choice of Algorithms)

본 프로젝트에서는 다음과 같은 딥러닝 모델들을 비교 분석합니다:

#### A. Convolutional Neural Network (CNN)
- **기본 CNN 모델**: 커스텀 아키텍처로 베이스라인 성능 측정
- **구조**: Conv2D → ReLU → MaxPooling → Flatten → Dense → Softmax

#### B. 전이학습 (Transfer Learning) 모델들
1. **ResNet50**
   - 잔차 연결(Residual Connection)을 통한 깊은 네트워크 학습
   - ImageNet 사전학습 가중치 활용
   
2. **EfficientNetB0**
   - 효율적인 모델 스케일링
   - 적은 파라미터로 높은 성능

3. **VGG16**
   - 단순하지만 강력한 아키텍처
   - 전이학습 벤치마크로 활용

#### C. Vision Transformer (ViT) - 선택사항
- Transformer 구조를 이미지 분류에 적용
- 최신 기법 성능 비교

### 2. 특징 추출 (Feature Engineering)

AI 생성 이미지와 실제 이미지를 구분하는 주요 특징:

1. **저수준 특징 (Low-level features)**
   - 픽셀 단위 노이즈 패턴
   - JPEG 압축 아티팩트
   - 색상 분포 이상

2. **고수준 특징 (High-level features)**
   - 물체 경계의 부자연스러움
   - 텍스처 일관성
   - 의미론적 이상 (예: 손가락 개수, 텍스트 오류)

3. **주파수 도메인 분석**
   - FFT (Fast Fourier Transform)를 통한 주파수 특성 분석
   - AI 생성 이미지의 특정 주파수 패턴 탐지

### 3. 모델 학습 프로세스

```
1. 데이터 로딩 및 전처리
   ↓
2. 모델 아키텍처 정의
   ↓
3. 손실 함수: Binary Cross-Entropy
   최적화: Adam Optimizer (lr=0.001)
   ↓
4. 학습 (Epochs: 50, Batch size: 32)
   - Early Stopping (patience=5)
   - ReduceLROnPlateau
   ↓
5. 검증 및 평가
   ↓
6. 최적 모델 저장
```

### 4. 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| Learning Rate | 0.001 (초기값) |
| Batch Size | 32 |
| Epochs | 50 (max) |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Dropout Rate | 0.5 |

### 5. 평가 지표 (Evaluation Metrics)

- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (AI 생성으로 예측한 것 중 실제 AI 생성 비율)
- **Recall**: 재현율 (실제 AI 생성 이미지를 올바르게 탐지한 비율)
- **F1-Score**: Precision과 Recall의 조화평균
- **ROC-AUC**: ROC 곡선 아래 면적
- **Confusion Matrix**: 혼동 행렬 분석

---

## IV. Evaluation & Analysis

### 1. 모델 성능 비교

*(프로젝트 진행 중 업데이트 예정)*

| 모델 | Accuracy | Precision | Recall | F1-Score | Training Time |
|------|----------|-----------|--------|----------|---------------|
| Basic CNN | TBD | TBD | TBD | TBD | TBD |
| ResNet50 | TBD | TBD | TBD | TBD | TBD |
| EfficientNetB0 | TBD | TBD | TBD | TBD | TBD |
| VGG16 | TBD | TBD | TBD | TBD | TBD |

### 2. 학습 곡선 (Learning Curves)

- Training Loss vs Validation Loss
- Training Accuracy vs Validation Accuracy
- 과적합(Overfitting) 여부 분석

*(그래프는 프로젝트 진행 중 추가 예정)*

### 3. Confusion Matrix

각 모델의 혼동 행렬을 통한 오분류 패턴 분석

### 4. 시각화 분석

#### A. Grad-CAM (Gradient-weighted Class Activation Mapping)
- 모델이 이미지의 어느 부분을 보고 판단하는지 시각화
- AI 생성 이미지의 특징적인 영역 탐지

#### B. 오분류 사례 분석
- False Positive: 실제 이미지를 AI 생성으로 잘못 분류
- False Negative: AI 생성 이미지를 실제로 잘못 분류
- 각 사례에 대한 원인 분석

### 5. 통계 분석

- 클래스별 정확도 분포
- 이미지 특성에 따른 성능 차이
- 신뢰도(Confidence) 분석

---

## V. Related Work

### 1. 참고 논문 및 연구

- [논문 제목 1] - 출처 링크
- [논문 제목 2] - 출처 링크
- *(프로젝트 진행 중 추가 예정)*

### 2. 사용한 라이브러리 및 도구

#### 딥러닝 프레임워크
- **PyTorch** / **TensorFlow+Keras**: 모델 구현 및 학습
- **torchvision** / **tf.keras.applications**: 사전학습 모델

#### 데이터 처리
- **NumPy**: 수치 연산
- **Pandas**: 데이터 관리
- **OpenCV / Pillow**: 이미지 처리
- **Albumentations**: 데이터 증강

#### 시각화
- **Matplotlib / Seaborn**: 그래프 및 차트
- **Plotly**: 인터랙티브 시각화
- **TensorBoard**: 학습 과정 모니터링

#### 기타
- **scikit-learn**: 평가 지표 계산
- **Kaggle API**: 데이터셋 다운로드

### 3. 참고 블로그 및 튜토리얼

- [Kaggle - Image Classification Tutorials](https://www.kaggle.com/learn/computer-vision)
- [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- *(추가 자료는 프로젝트 진행 중 업데이트)*

### 4. 기존 연구 및 Kaggle 대회

- AI Generated Image Detection 관련 Kaggle 대회
- 유사 프로젝트 및 솔루션 분석

---

## VI. Conclusion

### 프로젝트 결과 요약

*(프로젝트 완료 후 작성 예정)*

- 최고 성능 모델 및 정확도
- 주요 발견 사항
- AI 생성 이미지 판별의 핵심 특징

### 한계점 및 개선 방향

- 프로젝트의 한계점
- 추후 개선 가능한 부분
- 추가 실험 아이디어

### 배운 점 및 느낀 점

- 딥러닝 모델 구현 경험
- 팀 협업 과정에서의 배움
- 실제 문제 해결을 위한 AI 적용 경험

### 팀원 역할 분담

| 팀원 | 담당 업무 |
|------|----------|
| 홍길동 | 모델 구현, 학습 파이프라인 구축 |
| 김철수 | 데이터 수집 및 전처리, EDA |
| 이영희 | 결과 분석, 시각화, 문서 작성, 영상 제작 |

> **Note**: 실제 역할에 맞게 업데이트해주세요.

---

## 발표 영상

### 🎥 프로젝트 발표 영상 (5-10분)

<!-- 영상 제작 후 링크 추가 -->
[![프로젝트 발표 영상](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

> **Note**: YouTube 업로드 후 링크를 업데이트해주세요.

**영상 내용**:
- 프로젝트 소개 및 동기
- 데이터셋 설명
- 모델 구조 및 학습 과정
- 결과 분석 및 시연
- 결론 및 배운 점

---

## 프로젝트 구조

```
ai-image-detection/
├── README.md                 # 프로젝트 문서 (현재 파일)
├── requirements.txt          # 필요한 패키지 목록
├── .gitignore               # Git 제외 파일 목록
│
├── data/                    # 데이터셋 (용량 큰 파일은 .gitignore)
│   ├── raw/                 # 원본 데이터
│   ├── processed/           # 전처리된 데이터
│   └── README.md            # 데이터 설명
│
├── notebooks/               # Jupyter 노트북
│   ├── 01_EDA.ipynb        # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_final_model.ipynb
│
├── src/                     # 소스 코드
│   ├── data_loader.py      # 데이터 로딩
│   ├── models.py           # 모델 정의
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 스크립트
│   └── utils.py            # 유틸리티 함수
│
├── models/                  # 저장된 모델
│   └── best_model.pth
│
├── results/                 # 결과 파일
│   ├── figures/            # 그래프 및 이미지
│   ├── metrics/            # 평가 지표
│   └── reports/            # 분석 보고서
│
└── docs/                    # 추가 문서
    └── presentation.pdf    # 발표 자료 (선택사항)
```

---

## 실행 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/ai-image-detection.git
cd ai-image-detection

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 다운로드

```bash
# Kaggle API 설정 (kaggle.json 필요)
kaggle datasets download -d mkevinrinaldi/my-sampled-art-dataset-40k
unzip my-sampled-art-dataset-40k.zip -d ./data/raw
```

### 3. 모델 학습

```bash
python src/train.py --model resnet50 --epochs 50 --batch-size 32
```

### 4. 모델 평가

```bash
python src/evaluate.py --model-path models/best_model.pth
```

---

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

---

## 참고사항

- **제출 일정**:
  - 블로그 진행 상황: Nov. 25
  - 최종 블로그: TBD
- **프로젝트 스프레드시트**: [Google Sheets Link](https://docs.google.com/spreadsheets/d/18EDcCtfwc_LhaHkfw67yGCPDyLbR49uDt6rwYwjUVoA/edit?usp=sharing)

---

**Last Updated**: 2024-11-24