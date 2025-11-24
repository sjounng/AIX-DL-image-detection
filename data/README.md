# 데이터셋 (Dataset)

## 데이터셋 정보

- **이름**: My Sampled Art Dataset 40k
- **출처**: [Kaggle](https://www.kaggle.com/datasets/mkevinrinaldi/my-sampled-art-dataset-40k/data)
- **크기**: 약 40,000장

## 디렉토리 구조

```
data/
├── raw/                  # 원본 데이터 (다운로드한 그대로)
│   ├── FAKE/            # AI 생성 이미지 (~20,000장)
│   │   └── *_AI_*.jpg
│   └── REAL/            # 실제 예술 작품 (~20,000장)
│       └── *_Human_*.jpg
└── processed/           # 전처리된 데이터
    ├── train/
    │   ├── fake/
    │   └── real/
    ├── val/
    │   ├── fake/
    │   └── real/
    └── test/
        ├── fake/
        └── real/
```

## 데이터 다운로드 방법

### Kaggle API 사용

```bash
# Kaggle API 설치
pip install kaggle

# 데이터셋 다운로드
kaggle datasets download -d mkevinrinaldi/my-sampled-art-dataset-40k

# 압축 해제
unzip my-sampled-art-dataset-40k.zip -d ./data/raw
```

### 수동 다운로드 (추천)

1. [Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/mkevinrinaldi/my-sampled-art-dataset-40k/data) 방문
2. `Download` 버튼 클릭하여 데이터셋 다운로드
3. 다운로드한 파일을 `data/raw/` 디렉토리에 압축 해제:
   ```bash
   unzip ~/Downloads/archive.zip -d ./data/raw/
   ```
4. 구조 확인:
   ```bash
   ls data/raw/
   # 출력: FAKE  REAL

   ls data/raw/FAKE | head -3
   # 출력: *_AI_*.jpg 형식의 파일들

   ls data/raw/REAL | head -3
   # 출력: *_Human_*.jpg 형식의 파일들
   ```

## 주의사항

- 용량이 큰 파일이므로 `.gitignore`에 추가되어 있습니다
- Git에 업로드하지 마세요
