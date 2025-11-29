"""
PyTorch Dataset 및 DataLoader 구현

이 모듈은 AI 이미지 판별 데이터셋을 로딩하고 전처리하는 기능을 제공합니다.

주요 클래스:
    - AIImageDataset: 커스텀 PyTorch Dataset
    - get_transforms: Transform 파이프라인 생성
    - get_dataloaders: Train/Val/Test DataLoader 생성
"""

import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AIImageDataset(Dataset):
    """
    AI 이미지 판별을 위한 커스텀 Dataset 클래스

    Args:
        csv_file (str or Path): 이미지 경로와 레이블이 담긴 CSV 파일
        transform (callable, optional): 이미지 변환 함수

    Attributes:
        data (DataFrame): 이미지 경로와 레이블 데이터
        transform: 이미지 전처리 함수
    """

    def __init__(self, csv_file, transform=None):
        """
        Dataset 초기화

        Args:
            csv_file: CSV 파일 경로
            transform: torchvision transforms
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지와 레이블 반환

        Args:
            idx (int): 데이터 인덱스

        Returns:
            tuple: (image_tensor, label)
                - image_tensor: 전처리된 이미지 텐서
                - label: 클래스 레이블 (0: FAKE, 1: REAL)
        """
        # 이미지 경로 및 레이블
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')

        # Transform 적용
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    """
    이미지 전처리 Transform 파이프라인 생성

    Args:
        image_size (int): 리사이징할 이미지 크기 (기본값: 224)

    Returns:
        dict: 'train'과 'val_test' Transform을 담은 딕셔너리
    """
    # ImageNet 평균 및 표준편차
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training Transform (데이터 증강 포함)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Validation & Test Transform (증강 없음)
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        'train': train_transform,
        'val_test': val_test_transform
    }


def get_dataloaders(data_dir, batch_size=32, num_workers=4, image_size=224):
    """
    Train/Validation/Test DataLoader 생성

    Args:
        data_dir (str or Path): CSV 파일이 있는 디렉토리 (data/processed/)
        batch_size (int): 배치 크기 (기본값: 32)
        num_workers (int): 데이터 로딩 워커 수 (기본값: 4)
        image_size (int): 이미지 크기 (기본값: 224)

    Returns:
        dict: 'train', 'val', 'test' DataLoader를 담은 딕셔너리
    """
    data_dir = Path(data_dir)

    # Transform 생성
    transforms_dict = get_transforms(image_size)

    # Dataset 생성
    train_dataset = AIImageDataset(
        csv_file=data_dir / 'train.csv',
        transform=transforms_dict['train']
    )

    val_dataset = AIImageDataset(
        csv_file=data_dir / 'val.csv',
        transform=transforms_dict['val_test']
    )

    test_dataset = AIImageDataset(
        csv_file=data_dir / 'test.csv',
        transform=transforms_dict['val_test']
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Training은 셔플
        num_workers=num_workers,
        pin_memory=True  # GPU 사용 시 성능 향상
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Validation은 셔플 안 함
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Test도 셔플 안 함
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def test_dataloader():
    """
    DataLoader 테스트 함수

    DataLoader가 제대로 작동하는지 확인합니다.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'

    print("="*60)
    print("DataLoader 테스트")
    print("="*60)

    # DataLoader 생성
    dataloaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=16,
        num_workers=2,
        image_size=224
    )

    # 각 DataLoader 정보 출력
    for split_name, loader in dataloaders.items():
        print(f"\n{split_name.upper()} Loader:")
        print(f"  Dataset size: {len(loader.dataset):,}")
        print(f"  Batch size: {loader.batch_size}")
        print(f"  Number of batches: {len(loader)}")

    # Train loader에서 샘플 배치 가져오기
    print("\n" + "="*60)
    print("샘플 배치 확인")
    print("="*60)

    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))

    print(f"\n배치 shape:")
    print(f"  Images: {images.shape}")  # [batch_size, 3, 224, 224]
    print(f"  Labels: {labels.shape}")  # [batch_size]

    print(f"\n배치 내용:")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image min/max: {images.min():.3f} / {images.max():.3f}")
    print(f"  Labels: {labels[:8].tolist()}")  # 처음 8개 레이블

    # 레이블 분포
    fake_count = (labels == 0).sum().item()
    real_count = (labels == 1).sum().item()
    print(f"\n배치 내 레이블 분포:")
    print(f"  FAKE (0): {fake_count}개")
    print(f"  REAL (1): {real_count}개")

    # 이미지 시각화 (선택적)
    print("\n이미지 시각화 중...")

    # 정규화 해제를 위한 평균/표준편차
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # 정규화 해제
            img = images[i] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = img.clip(0, 1)

            ax.imshow(img)
            label_name = 'FAKE' if labels[i] == 0 else 'REAL'
            ax.set_title(f'{label_name}', fontsize=10)
            ax.axis('off')

    plt.suptitle('DataLoader 샘플 이미지', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 저장
    save_path = project_root / 'results' / 'figures' / 'dataloader_test.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"시각화 저장: {save_path}")

    print("\n" + "="*60)
    print("✅ DataLoader 테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    # 테스트 실행
    test_dataloader()
