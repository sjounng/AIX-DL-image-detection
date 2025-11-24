# AI 이미지 판별 프로젝트 - 개발 가이드

## 개발 단계 (Development Steps)

이 문서는 AI 생성 이미지 판별 프로젝트의 전체 개발 단계를 정리한 가이드입니다.

---

## Phase 1: 환경 설정 (Environment Setup)

### 1.1 requirements.txt 작성 
**목표**: 프로젝트에 필요한 모든 라이브러리 정의

**필수 라이브러리**:
- PyTorch + torchvision (딥러닝 프레임워크)
- numpy, pandas (데이터 처리)
- matplotlib, seaborn, plotly (시각화)
- Pillow, opencv-python (이미지 처리)
- scikit-learn (평가 지표)
- albumentations (데이터 증강)
- tensorboard (학습 모니터링)
- jupyter, ipykernel (노트북)

**결과물**: `requirements.txt`

### 1.2 가상환경 설정 및 패키지 설치 
**목표**: 개발 환경 구축

**명령어**:
```bash
# 패키지 설치
pip install -r requirements.txt

# 설치 확인
python -c "import torch; print(torch.__version__)"
```

**결과물**: 모든 패키지 설치 완료

---

## Phase 2: 탐색적 데이터 분석 (EDA) 

### 2.1 데이터 기본 정보 확인
**목표**: 데이터셋의 구조와 크기 파악

**작업 내용**:
- 총 이미지 수 확인
- 클래스 분포 확인 (FAKE vs REAL)
- 데이터 균형 검증

### 2.2 이미지 특성 분석
**목표**: 이미지의 물리적 특성 파악

**작업 내용**:
- 이미지 크기 분포 (너비, 높이)
- 이미지 해상도 범위
- 파일 크기 분포
- 샘플 이미지 시각화 (각 클래스별 10개씩)

### 2.3 색상 및 통계 분석
**목표**: 이미지의 색상 패턴 분석

**작업 내용**:
- RGB 채널별 분포
- FAKE vs REAL 색상 차이
- 통계 요약 테이블

**결과물**: `notebooks/01_EDA.ipynb`

**주요 발견**:
- 클래스 균형: FAKE와 REAL 각각 약 20,000개
- 이미지 크기 다양 → 리사이징 필요 (224x224 추천)
- 색상 분포 차이 존재 → 모델 학습 가능 특징

---

## Phase 3: 데이터 전처리 (Data Preprocessing)

### 3.1 데이터 분할
**목표**: Train/Validation/Test 세트 분할

**분할 비율**:
- Training: 70% (~28,000장: 14,000 FAKE + 14,000 REAL)
- Validation: 15% (~6,000장: 3,000 FAKE + 3,000 REAL)
- Test: 15% (~6,000장: 3,000 FAKE + 3,000 REAL)

**작업 내용**:
```python
from sklearn.model_selection import train_test_split

# 데이터 분할 로직
# - Stratified split (클래스 비율 유지)
# - Random seed 고정 (재현성)
```

### 3.2 이미지 전처리 파이프라인
**목표**: 학습을 위한 이미지 표준화

**전처리 단계**:
1. **리사이징**: 224x224 또는 256x256
2. **정규화**:
   - ImageNet 평균/표준편차 사용
   - mean=[0.485, 0.456, 0.406]
   - std=[0.229, 0.224, 0.225]
3. **데이터 증강 (Training only)**:
   - Random Horizontal Flip (p=0.5)
   - Random Rotation (±15도)
   - Random Brightness/Contrast 조정
   - Color Jitter

**작업 내용**:
```python
from torchvision import transforms
from albumentations import (
    Compose, HorizontalFlip, Rotate,
    ColorJitter, Normalize, Resize
)

# Transform 정의
train_transform = Compose([...])
val_test_transform = Compose([...])
```

**결과물**:
- `notebooks/02_preprocessing.ipynb`
- `data/processed/train/`, `data/processed/val/`, `data/processed/test/`

---

## Phase 4: 데이터 로더 구현

### 4.1 PyTorch Dataset 클래스
**목표**: 커스텀 데이터셋 클래스 구현

**작업 내용**:
```python
from torch.utils.data import Dataset

class AIImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 로드 및 변환
        pass
```

### 4.2 DataLoader 설정
**목표**: 배치 처리 및 효율적인 데이터 로딩

**하이퍼파라미터**:
- Batch size: 32 (또는 64)
- Shuffle: True (training), False (val/test)
- Num workers: 4 (CPU 코어 수에 따라 조정)
- Pin memory: True (GPU 사용 시)

**작업 내용**:
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

**결과물**: `src/data_loader.py`

---

## Phase 5: 모델 구현

### 5.1 베이스라인 모델 (간단한 CNN)
**목표**: 성능 기준점 설정

**모델 구조**:
```
Input (3, 224, 224)
    ↓
Conv2D (64) → ReLU → MaxPool
    ↓
Conv2D (128) → ReLU → MaxPool
    ↓
Conv2D (256) → ReLU → MaxPool
    ↓
Flatten → Dense (512) → ReLU → Dropout(0.5)
    ↓
Dense (2) → Softmax
```

**작업 내용**:
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 레이어 정의
        pass

    def forward(self, x):
        # Forward pass
        pass
```

### 5.2 전이학습 모델들
**목표**: 사전학습된 모델로 성능 향상

**모델 후보**:

#### A. ResNet50
- **특징**: Residual Connection으로 깊은 네트워크 학습
- **파라미터**: ~25M
- **장점**: 안정적인 학습, 높은 성능

```python
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 2)  # 마지막 레이어 교체
```

#### B. EfficientNetB0
- **특징**: 효율적인 모델 스케일링
- **파라미터**: ~5M
- **장점**: 적은 파라미터, 빠른 학습

```python
from torchvision.models import efficientnet_b0

efficientnet = efficientnet_b0(pretrained=True)
efficientnet.classifier[1] = nn.Linear(1280, 2)
```

#### C. VGG16
- **특징**: 단순하지만 강력한 구조
- **파라미터**: ~138M
- **장점**: 전이학습 벤치마크

```python
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier[6] = nn.Linear(4096, 2)
```

### 5.3 모델 학습 전략
**Fine-tuning 전략**:
1. **Feature Extraction**:
   - 사전학습된 레이어 동결
   - 마지막 레이어만 학습
2. **Fine-tuning**:
   - 전체 레이어 학습
   - 낮은 learning rate 사용

**결과물**:
- `src/models.py`
- `notebooks/03_baseline_model.ipynb`

---

## Phase 6: 학습 파이프라인

### 6.1 학습 스크립트
**목표**: 모델 학습 자동화

**하이퍼파라미터**:
```python
HYPERPARAMETERS = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'weight_decay': 1e-4,
    'dropout': 0.5
}
```

**학습 로직**:
```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)
```

### 6.2 검증 및 평가
**목표**: 모델 성능 모니터링

**평가 지표**:
- **Accuracy**: 전체 정확도
- **Precision**: AI 생성으로 예측한 것 중 실제 비율
- **Recall**: 실제 AI 생성을 올바르게 탐지한 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **ROC-AUC**: 모델의 전반적 성능

**검증 로직**:
```python
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return val_loss / len(loader), accuracy
```

### 6.3 모델 저장 및 조기 종료
**목표**: 최적 모델 저장 및 과적합 방지

**Early Stopping**:
```python
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'models/best_model.pth')
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break
```

**Learning Rate Scheduler**:
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3
)
```

**결과물**:
- `src/train.py`
- `notebooks/04_final_model.ipynb`
- `models/best_model.pth`

---

## Phase 7: 평가 및 분석

### 7.1 성능 평가
**목표**: Test set에서 최종 성능 측정

**평가 항목**:
1. **Confusion Matrix**:
```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []

# 예측 수집
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

2. **ROC Curve & AUC**:
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
```

### 7.2 시각화
**목표**: 학습 과정 및 결과 시각화

**시각화 항목**:
1. **학습 곡선**:
   - Training Loss vs Validation Loss
   - Training Accuracy vs Validation Accuracy

2. **Grad-CAM**:
   - 모델이 주목하는 영역 시각화
   - FAKE/REAL 각각의 중요 특징 확인

```python
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
grayscale_cam = cam(input_tensor=input_image)
```

3. **오분류 사례 분석**:
   - False Positive: 실제를 AI로 잘못 분류
   - False Negative: AI를 실제로 잘못 분류
   - 각 사례의 특징 분석

### 7.3 모델 비교
**목표**: 여러 모델의 성능 비교

**비교 테이블**:
| 모델 | Accuracy | Precision | Recall | F1-Score | Training Time | Params |
|------|----------|-----------|--------|----------|---------------|---------|
| SimpleCNN | TBD | TBD | TBD | TBD | TBD | ~5M |
| ResNet50 | TBD | TBD | TBD | TBD | TBD | ~25M |
| EfficientNetB0 | TBD | TBD | TBD | TBD | TBD | ~5M |
| VGG16 | TBD | TBD | TBD | TBD | TBD | ~138M |

**결과물**:
- `src/evaluate.py`
- `results/figures/` (모든 그래프)
- `results/metrics/` (평가 지표 CSV)
- `results/reports/` (분석 보고서)

---

## Phase 8: 문서화 및 마무리

### 8.1 README.md 업데이트
**목표**: 실험 결과를 문서에 반영

**업데이트 항목**:
- Section IV: Evaluation & Analysis
  - 모델 성능 비교 테이블 작성
  - 학습 곡선 이미지 추가
  - Confusion Matrix 추가
  - 주요 발견 사항 작성

- Section VI: Conclusion
  - 최고 성능 모델 및 정확도
  - 주요 발견 사항
  - 한계점 및 개선 방향
  - 배운 점

### 8.2 발표 자료/영상 준비
**목표**: 프로젝트 결과 발표

**발표 구성** (5-10분):
1. 프로젝트 소개 및 동기 (1분)
2. 데이터셋 설명 (1분)
3. 모델 구조 및 학습 과정 (2-3분)
4. 결과 분석 및 시연 (2-3분)
5. 결론 및 배운 점 (1분)

**발표 자료**:
- `docs/presentation.pdf` 또는 PPT
- 데모 영상 또는 라이브 시연

### 8.3 코드 정리 및 리팩토링
**목표**: 코드 품질 향상

**정리 항목**:
- [ ] 주석 추가
- [ ] Docstring 작성
- [ ] 불필요한 코드 제거
- [ ] 코드 스타일 통일 (PEP8)
- [ ] 함수/클래스 재사용성 개선

**최종 체크리스트**:
- [ ] 모든 노트북이 에러 없이 실행되는가?
- [ ] README.md가 최신 상태인가?
- [ ] 결과 파일들이 적절히 저장되었는가?
- [ ] .gitignore가 제대로 작동하는가?
- [ ] 발표 자료가 준비되었는가?

---

## 📊 권장 작업 일정

### Week 1: 환경 설정 및 데이터 탐색
- ✅ Phase 1: 환경 설정 (완료)
- ✅ Phase 2: EDA (완료)

### Week 2: 데이터 전처리 및 로더
- Phase 3: 데이터 전처리
- Phase 4: 데이터 로더 구현

### Week 3: 베이스라인 모델
- Phase 5.1: SimpleCNN 구현
- Phase 6: 학습 파이프라인 (베이스라인)

### Week 4: 전이학습 모델
- Phase 5.2: ResNet50, EfficientNet, VGG16
- Phase 6: 각 모델 학습 및 비교

### Week 5: 평가 및 문서화
- Phase 7: 평가 및 시각화
- Phase 8: 문서화 및 발표 준비

---

## 🎯 현재 진행 상황

- [x] Phase 1.1: requirements.txt 작성
- [x] Phase 1.2: 패키지 설치
- [x] Phase 2: EDA 노트북 작성
- [ ] Phase 3: 데이터 전처리
- [ ] Phase 4: 데이터 로더
- [ ] Phase 5: 모델 구현
- [ ] Phase 6: 학습 파이프라인
- [ ] Phase 7: 평가 및 분석
- [ ] Phase 8: 문서화

---

## 💡 추가 참고 사항

### 유용한 리소스
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Grad-CAM 논문](https://arxiv.org/abs/1610.02391)

### 트러블슈팅
- GPU 메모리 부족 → Batch size 줄이기
- 과적합 → Dropout, Data Augmentation 강화
- 학습 속도 느림 → num_workers 조정, Mixed Precision Training

### 실험 관리
- TensorBoard로 학습 과정 기록
- 각 실험마다 하이퍼파라미터 기록
- 최고 성능 모델 별도 저장

---

**Last Updated**: 2024-11-24
