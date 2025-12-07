# AI 생성 이미지 판별 프로젝트
## AI-Generated Image Detection Project

---

## 📋 목차 (Table of Contents)

- [Quick Start - 바로 사용하기](#quick-start---바로-사용하기)
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

## Quick Start - 바로 사용하기

```bash
# 1. 저장소 클론 및 설치
git clone https://github.com/sjounng/AIX-DL-image-detection.git
cd AIX-DL-image-detection
pip install -r requirements.txt

# 2. 이미지 판별 (최고 정확도: Ensemble Soft Voting 99.20%)
python src/ensemble.py --method soft --image "your_image.jpg" --batch-size 32 --num-workers 0
```

자세한 모델 선택 가이드는 [VI. Conclusion](#vi-conclusion) 참조

---

## 프로젝트 개요

이 프로젝트는 AI로 생성된 이미지와 실제 이미지를 구분하는 딥러닝 모델을 개발하고 분석합니다.

---

## 팀원 소개 (Members)

| 이름  | 학과         | 이메일                | 역할            |
|-----|------------|--------------------|---------------|
| 송준우 | 정보시스템학과    | jwsong5160@gmail.com | 모델 구현, 블로그 관리 |
| 전용현 | 데이터사이언스학과  | jeonyh010328@gmail.com | 모델 구현         |
| 서채원 | 컴퓨터소프트웨어학과 | tjcodnjs111@gmail.com | 모델 구현         |


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

#### 완료된 전처리 결과

**데이터 분할 현황:**
- **Training Set**: 28,000장 (70%)
  - FAKE: 14,000장
  - REAL: 14,000장
- **Validation Set**: 6,000장 (15%)
  - FAKE: 3,000장
  - REAL: 3,000장
- **Test Set**: 6,000장 (15%)
  - FAKE: 3,000장
  - REAL: 3,000장

**전처리 파이프라인:**
- 이미지 크기: 224x224 픽셀로 자동 리사이즈
- 정규화: ImageNet 평균/표준편차 사용
- 데이터 증강 (Training만):
  - Random Horizontal Flip (p=0.5)
  - Random Rotation (±15도)
  - Color Jitter (brightness, contrast, saturation, hue)

**전처리 결과 파일:**
- `data/processed/train.csv` - 28,000개 샘플
- `data/processed/val.csv` - 6,000개 샘플
- `data/processed/test.csv` - 6,000개 샘플

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

4. **ConvNeXt (Tiny)**
   - CNN의 장점과 Transformer의 설계 철학 결합
   - 현대적인 CNN 아키텍처
   - ImageNet 사전학습 가중치 활용


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

#### 테스트 세트 평가 결과 (전체 7개 모델 완료)

| 순위 | 모델 | Test Accuracy | Precision | Recall | F1-Score | ROC AUC | 파라미터 수 |
|------|------|--------------|-----------|--------|----------|---------|------------|
| **1위** | **Ensemble (Soft)** | **99.20%** | **99.43%** | **98.97%** | **99.20%** | **0.9996** | ~27M |
| 2위 | EfficientNetB0 | 98.97% | 99.13% | 98.80% | 98.96% | 0.9996 | ~4M |
| 3위 | ResNet50 | 98.78% | 99.13% | 98.43% | 98.78% | 0.9993 | ~23M |
| 4위 | Ensemble (Hard) | 98.75% | 99.59% | 97.90% | 98.74% | 0.9996 | ~27M |
| 5위 | VGG16 | 98.65% | 98.86% | 98.43% | 98.65% | 0.9988 | ~134M |
| 6위 | ConvNeXt | 97.80% | 99.46% | 96.13% | 97.76% | 1.0000 | ~28M |
| 7위 | SimpleCNN | 97.18% | 97.80% | 96.53% | 97.16% | 0.9961 | ~2M |

**주요 발견:**
- Ensemble (Soft)이 최고 성능 (99.20%), 개별 모델 대비 +0.23%p 향상
- EfficientNetB0가 단일 모델 중 최고 (98.97%), 가장 효율적 (4M 파라미터)
- ConvNeXt가 최고 Precision (99.46%), False Positive 최소화
- 전이학습 모델이 SimpleCNN 대비 1.5%p 이상 높은 성능
- 모든 모델 ROC AUC > 0.99, 판별 작업에 매우 효과적

### 2. 학습 및 앙상블 결과

**개별 모델 학습:**
- Epochs: 50 (최대), Early Stopping 적용
- Optimizer: Adam
- Learning Rate: 초기 0.001, ReduceLROnPlateau로 자동 감소

**최종 테스트 성능:**

| 모델 | Test Accuracy | Precision | Recall | F1-Score |
|------|--------------|-----------|--------|----------|
| EfficientNetB0 | 98.97% | 99.13% | 98.80% | 98.96% |
| ResNet50 | 98.78% | 99.13% | 98.43% | 98.78% |
| VGG16 | 98.65% | 98.86% | 98.43% | 98.65% |
| ConvNeXt | 97.80% | 99.46% | 96.13% | 97.76% |
| SimpleCNN | 97.18% | 97.80% | 96.53% | 97.16% |

**앙상블 모델 (상위 2개 조합):**

EfficientNetB0 + ResNet50 조합으로 두 가지 앙상블 방식을 구현:

| 앙상블 방식 | Test Accuracy | Precision | Recall | F1-Score | 설명 |
|------------|--------------|-----------|--------|----------|------|
| Soft Voting | 99.20% | 99.43% | 98.97% | 99.20% | 확률 평균 |
| Hard Voting | 98.75% | 99.59% | 97.90% | 98.74% | 다수결 |

**앙상블 효과:**
- Soft Voting이 개별 모델 대비 +0.23%p 성능 향상
- 개별 모델의 약점을 상호 보완하여 안정적인 예측

**생성된 결과 파일:**
- `results/figures/` - 학습 곡선, 혼동 행렬, ROC 곡선
- `results/metrics/` - 테스트 결과 CSV, 전체 모델 비교

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

본 프로젝트에서는 AI 생성 이미지와 실제 이미지를 구분하는 딥러닝 모델을 성공적으로 구현하고 평가했습니다.

**총 7개 모델을 학습 및 평가하여 최적의 솔루션을 도출했습니다.**

#### 주요 성과

**1. 최고 성능 모델: Ensemble (Soft Voting)**
- **Test Accuracy: 99.20%** - 6,000개 테스트 이미지 중 5,952개 정확 분류
- **ROC AUC: 0.9996** - 거의 완벽한 분류 성능
- **앙상블 효과**: 개별 모델 대비 +0.23%p 성능 향상 (EfficientNetB0 98.97% → 99.20%)
- **구성**: EfficientNetB0 + ResNet50 (상위 2개 모델 조합)
- **균형잡힌 성능**: Precision 99.43%, Recall 98.97%

**2. 단일 모델 최고 성능: EfficientNetB0**
- **Test Accuracy: 98.97%** - 단일 모델 중 최고 성능
- **효율성**: 약 4M 파라미터로 최고 성능 달성 (VGG16 대비 1/33 크기, 앙상블 대비 1/7 크기)
- **실용성**: 낮은 메모리 사용량으로 실시간 처리 가능

**3. 모델 비교 분석 완료**
- 7가지 모델 학습 및 평가 완료 (SimpleCNN, ResNet50, EfficientNetB0, VGG16, ConvNeXt, Ensemble Soft, Ensemble Hard)
- 전이학습의 효과 입증: 사전학습 모델들이 SimpleCNN 대비 1.5%p+ 높은 성능
- 앙상블 학습 효과 입증: Soft Voting이 Hard Voting보다 우수한 성능 (99.20% vs 98.75%)
- 모델 크기와 성능이 비례하지 않음: EfficientNetB0 > ResNet50 > VGG16 (효율성 순)
- ConvNeXt: 높은 Precision (99.46%)으로 False Positive 최소화에 강점

**4. 실용적 활용 가능성**
- **Ensemble (Soft Voting) 99.20%**: 최고 정확도로 실제 환경에서 활용 가능
- 6,000개 테스트 이미지 중 5,952개 정확 분류 (오류율 0.8%)
- 평가 시스템 구현으로 즉시 사용 가능
- 단일 모델 사용 시 빠른 추론 속도 옵션 제공

#### 주요 발견 사항

1. **전이학습의 중요성**
   - ImageNet 사전학습 가중치가 AI 이미지 판별에도 매우 효과적
   - 적은 데이터로도 높은 성능 달성 가능

2. **모델 효율성**
   - EfficientNet의 Compound Scaling 기법이 효과적
   - 파라미터 수가 많다고 반드시 성능이 좋은 것은 아님

3. **클래스 균형**
   - FAKE/REAL 클래스 간 성능 차이 < 0.5%p로 매우 균형잡힌 분류
   - 데이터 증강 및 균형잡힌 데이터셋의 효과

4. **AI 생성 이미지의 특징**
   - 딥러닝 모델이 인간이 감지하기 어려운 패턴 학습
   - 픽셀 수준의 미세한 차이로도 98%+ 정확도 달성 가능

5. **앙상블 학습의 효과**
   - 상위 2개 모델(EfficientNetB0 + ResNet50) 조합으로 성능 향상
   - Soft Voting(확률 평균)이 Hard Voting(다수결)보다 우수
   - 개별 모델의 약점을 상호 보완하여 더 안정적인 예측

### 한계점 및 개선 방향

#### 한계점
1. **데이터셋 특성**
   - 예술 작품 위주 데이터셋으로, 일반 사진이나 다른 도메인에서의 성능은 검증 필요
   - 특정 AI 생성 도구(Stable Diffusion, Latent Diffusion)에 제한

2. **최신 생성 모델 대응**
   - DALL-E 3, Midjourney v6 등 최신 모델 생성 이미지는 미포함
   - 생성 기술 발전에 따른 지속적인 모델 업데이트 필요

3. **설명 가능성**
   - 모델이 어떤 특징을 보고 판단하는지 완전히 이해하기 어려움
   - Grad-CAM 등 시각화 기법 추가 필요

#### 개선 방향
1. **데이터 확장**
   - 다양한 도메인(풍경, 인물, 사물 등) 이미지 추가
   - 최신 AI 생성 도구의 이미지 포함

2. **더 다양한 앙상블 기법**
   - Stacking, Weighted Ensemble 등
   - 3개 이상 모델 조합 실험

3. **설명 가능한 AI**
   - Grad-CAM, LIME 등을 통한 판단 근거 시각화
   - 사용자 신뢰도 향상

4. **실시간 웹 서비스**
   - Gradio/Streamlit 기반 웹 인터페이스 개발
   - 일반 사용자도 쉽게 사용 가능한 서비스 구축

5. **경량화**
   - 모델 양자화(Quantization) 및 프루닝(Pruning)
   - 모바일 환경에서도 동작 가능하도록 최적화

### 프로젝트를 통해 배운 점

1. **전이학습의 효과**
   - ImageNet 사전학습 모델이 AI 이미지 판별에도 효과적
   - EfficientNet이 파라미터 대비 가장 우수한 성능

2. **앙상블의 실전 적용**
   - Soft Voting이 Hard Voting보다 우수
   - 상위 2개 모델 조합으로 +0.23%p 성능 향상

3. **실용적 시스템 구축**
   - 99.20% 정확도로 실제 환경에서 활용 가능
   - 추론 시스템 구현으로 즉시 사용 가능

### 최종 권장사항

본 프로젝트를 통해 총 7개의 모델을 평가한 결과, **최고 정확도 달성을 위해 다음을 권장합니다:**

### 모델 성능 분석

| 순위 | 모델 | 정확도 | Precision | Recall | F1-Score | 특징 |
|------|------|--------|-----------|--------|----------|------|
| **1위** | **Ensemble (Soft)** | **99.20%** | **99.43%** | **98.97%** | **99.20%** | **최고 정확도** |
| 2위 | EfficientNetB0 | 98.97% | 99.13% | 98.80% | 98.96% | 단일 모델 중 최고 |
| 3위 | ResNet50 | 98.78% | 99.13% | 98.43% | 98.78% | 안정적 성능 |
| 4위 | Ensemble (Hard) | 98.75% | 99.59% | 97.90% | 98.74% | 높은 Precision |

**결론**: 본 프로젝트의 목표는 AI 생성 이미지 판별의 **최대 정확도 달성**이므로, **Ensemble (Soft Voting) 99.20%**를 최종 모델로 권장합니다.


---

## 발표 영상

### 🎥 프로젝트 발표 영상 (5-10분)

<!-- 영상 제작 후 링크 추가 -->
[![프로젝트 발표 영상](https://img.youtube.com/vi/O5qpDoHcEqI/0.jpg)](https://youtu.be/O5qpDoHcEqI)


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
│   ├── __init__.py         # 패키지 초기화
│   ├── data_loader.py      # 데이터 로딩 및 전처리
│   ├── models.py           # 모델 정의 (ResNet50, EfficientNetB0, VGG16, ConvNeXt, SimpleCNN)
│   ├── preprocessing.py    # 데이터 전처리 및 분할
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 스크립트
│   ├── ensemble.py         # 앙상블 모델 평가 스크립트
│   ├── compare_results.py  # 전체 모델 비교 스크립트
│   └── inference.py        # 이미지 판별 스크립트
│
├── models/                  # 저장된 모델 체크포인트
│   ├── efficientnet_b0_best.pth  # EfficientNetB0 (98.97% 정확도) 
│   ├── resnet50_best.pth         # ResNet50 (98.78% 정확도)
│   ├── vgg16_best.pth            # VGG16 (98.65% 정확도)
│   ├── convnext_best.pth         # ConvNeXt (97.80% 정확도)
│   └── simple_cnn_best.pth       # SimpleCNN (97.18% 정확도)
│
├── results/                 # 결과 파일
│   ├── figures/            # 그래프 및 시각화
│   │   ├── efficientnet_b0_training_curves.png
│   │   ├── efficientnet_b0_confusion_matrix.png
│   │   ├── efficientnet_b0_roc_curve.png
│   │   ├── resnet50_training_curves.png
│   │   ├── resnet50_confusion_matrix.png
│   │   ├── resnet50_roc_curve.png
│   │   ├── vgg16_training_curves.png
│   │   ├── vgg16_confusion_matrix.png
│   │   ├── vgg16_roc_curve.png
│   │   ├── simple_cnn_training_curves.png
│   │   ├── simple_cnn_confusion_matrix.png
│   │   ├── simple_cnn_roc_curve.png
│   │   ├── confusion_matrix_convnext.png
│   │   ├── roc_curve_convnext.png
│   │   ├── ensemble_soft_confusion_matrix.png
│   │   ├── ensemble_soft_roc_curve.png
│   │   ├── ensemble_hard_confusion_matrix.png
│   │   ├── ensemble_hard_roc_curve.png
│   │   ├── model_comparison_all_metrics.png
│   │   └── model_comparison_roc_auc.png
│   ├── metrics/            # 평가 지표 CSV
│   │   ├── efficientnet_b0_training_history.csv
│   │   ├── efficientnet_b0_test_results.csv
│   │   ├── resnet50_training_history.csv
│   │   ├── resnet50_test_results.csv
│   │   ├── vgg16_training_history.csv
│   │   ├── vgg16_test_results.csv
│   │   ├── simple_cnn_training_history.csv
│   │   ├── simple_cnn_test_results.csv
│   │   ├── convnext_test_results.csv
│   │   ├── ensemble_soft_test_results.csv
│   │   ├── ensemble_hard_test_results.csv
│   │   └── all_models_comparison.csv
│   └── predictions/        # Inference 결과
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

### 3. 데이터 전처리

```bash
# 데이터 전처리 및 Train/Val/Test 분할
python src/preprocessing.py
```

### 4. 모델 학습

```bash
# EfficientNetB0 학습 (권장)
python src/train.py --model efficientnet_b0 --epochs 50 --batch-size 32 --num-workers 0

# ResNet50 학습
python src/train.py --model resnet50 --epochs 50 --batch-size 32 --num-workers 0

# VGG16 학습
python src/train.py --model vgg16 --epochs 50 --batch-size 32 --num-workers 0

# SimpleCNN 학습
python src/train.py --model simple_cnn --epochs 50 --batch-size 32 --num-workers 0
```

### 5. 모델 평가

```bash
# EfficientNetB0 평가
python src/evaluate.py --model efficientnet_b0 --batch-size 32 --num-workers 0

# ResNet50 평가
python src/evaluate.py --model resnet50 --batch-size 32 --num-workers 0
```

### 6. 앙상블 모델 평가

```bash
# Soft Voting 앙상블 (권장 - 최고 성능 99.20%)
python src/ensemble.py --method soft --batch-size 32 --num-workers 0

# Hard Voting 앙상블
python src/ensemble.py --method hard --batch-size 32 --num-workers 0

# 전체 모델 성능 비교 (앙상블 포함)
python src/compare_results.py
```

### 7. 이미지 판별 (Inference) - 실제 사용 방법

#### 권장: Ensemble (Soft Voting) 사용 (최고 정확도 99.20%)

```bash
# 테스트 세트 전체 평가 (권장)
python src/ensemble.py --method soft --batch-size 32 --num-workers 0

# 출력:
# - Test Accuracy: 99.20%
# - Confusion Matrix 및 ROC Curve 생성
# - 결과 저장: results/metrics/ensemble_soft_test_results.csv
```

**Ensemble 모델 특징:**
- **최고 정확도**: 99.20% (6,000개 테스트 이미지 중 5,952개 정확 분류)
- **구성**: EfficientNetB0 + ResNet50 (상위 2개 모델 조합)
- **방식**: Soft Voting (확률 평균)
- **성능 향상**: 개별 모델 대비 +0.23%p

#### 대안: 단일 모델 사용 (빠른 추론)

빠른 추론이 필요한 경우 EfficientNetB0 사용:

```bash
# 단일 이미지 판별
python src/inference.py --model efficientnet_b0 --image "path/to/image.jpg"

# 출력 예시:
# Image: path/to/image.jpg
# Prediction: FAKE (AI-generated)
# Confidence: 99.8%
# Probabilities: FAKE: 0.998, REAL: 0.002

# 여러 이미지 한번에 판별
python src/inference.py --model efficientnet_b0 --image "img1.jpg" "img2.jpg" "img3.jpg"

# 폴더 내 모든 이미지 판별 (대량 처리)
python src/inference.py --model efficientnet_b0 --image-dir "path/to/images"

# 결과를 CSV로 저장
python src/inference.py --model efficientnet_b0 --image "image.jpg" --output "results/predictions.csv"
```

**단일 모델 특징:**
- **정확도**: 98.97% (단일 모델 중 최고)
- **속도**: 매우 빠름 (앙상블 대비 약 2배)
- **메모리**: 약 4M 파라미터 (앙상블 대비 1/7)





## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

---

## 참고사항

- **제출 일정**:
  - 블로그 진행 상황: Nov. 25
  - 최종 블로그: Dec. 9

---

## 프로젝트 진행 현황

### 완료된 작업 (프로젝트 완료)
- [x] 데이터셋 다운로드 및 구조 확인
- [x] 데이터 전처리 및 Train/Val/Test 분할 (70/15/15)
- [x] PyTorch Dataset 및 DataLoader 구현
- [x] 모델 아키텍처 구현 (SimpleCNN, ResNet50, EfficientNetB0, VGG16)
- [x] 학습 파이프라인 구축 (Early Stopping, ReduceLROnPlateau)
- [x] **전체 5개 개별 모델 학습 완료**
  - [x] EfficientNetB0 (98.97% 정확도)
  - [x] ResNet50 (98.78% 정확도)
  - [x] VGG16 (98.65% 정확도)
  - [x] ConvNeXt (97.80% 정확도)
  - [x] SimpleCNN (97.18% 정확도)
- [x] 평가 스크립트 작성 및 **전체 모델 테스트 세트 평가 완료**
- [x] **앙상블 모델 구현 및 평가 완료** 
  - [x] Ensemble Soft Voting (99.20% 정확도) 
  - [x] Ensemble Hard Voting (98.75% 정확도)
- [x] **전체 7개 모델 성능 비교 분석 완료 (ConvNeXt 포함)**
- [x] 추론(Inference) 시스템 구현
- [x] 혼동 행렬, ROC 곡선, 학습 곡선 생성 (전체 모델)
- [x] **README 문서 최종 업데이트 (앙상블 포함)**
- [x] **프로젝트 결과 문서화 완료**

### 최종 결과
- **최고 성능**: Ensemble Soft Voting - 99.20% 정확도
- **최고 효율**: EfficientNetB0 - 98.97% 정확도 (4M 파라미터)
- **앙상블 효과**: +0.23%p 성능 향상


---

**Last Updated**: 2025-12-06 (앙상블 모델 및 ConvNeXt 추가)