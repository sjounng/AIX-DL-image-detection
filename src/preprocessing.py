"""
데이터 전처리 스크립트

이 스크립트는 AI 이미지 판별 데이터셋을 Train/Validation/Test로 분할하고,
전처리 파이프라인을 정의합니다.

실행 방법:
    python src/preprocessing.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 랜덤 시드 고정 (재현성)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 시각화 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_directories():
    """프로젝트 디렉토리 설정"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results' / 'figures'

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        'project_root': project_root,
        'data_dir': data_dir,
        'output_dir': output_dir,
        'results_dir': results_dir,
        'fake_dir': data_dir / 'FAKE',
        'real_dir': data_dir / 'REAL'
    }


def collect_image_paths(dirs):
    """이미지 파일 경로 수집"""
    print("이미지 파일 경로 수집 중...")

    fake_images = list(dirs['fake_dir'].glob('*.jpg')) + list(dirs['fake_dir'].glob('*.png'))
    real_images = list(dirs['real_dir'].glob('*.jpg')) + list(dirs['real_dir'].glob('*.png'))

    print(f"\nFAKE 이미지: {len(fake_images):,}개")
    print(f"REAL 이미지: {len(real_images):,}개")
    print(f"전체 이미지: {len(fake_images) + len(real_images):,}개")

    # 경로와 레이블 생성
    image_paths = fake_images + real_images
    labels = [0] * len(fake_images) + [1] * len(real_images)  # 0: FAKE, 1: REAL

    return image_paths, labels


def split_dataset(image_paths, labels):
    """데이터셋을 Train/Validation/Test로 분할"""
    print("\n" + "="*60)
    print("데이터 분할 중...")
    print("="*60)

    # Train / (Val + Test) 분할 (70% / 30%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    print(f"\n1단계: Train / Temp 분할")
    print(f"  Train: {len(train_paths):,}개 ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Temp:  {len(temp_paths):,}개 ({len(temp_paths)/len(image_paths)*100:.1f}%)")

    # Val / Test 분할 (각각 15%)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,  # 30%의 절반 = 15%
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    print(f"\n2단계: Val / Test 분할")
    print(f"  Val:   {len(val_paths):,}개 ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths):,}개 ({len(test_paths)/len(image_paths)*100:.1f}%)")

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def print_split_statistics(splits):
    """분할 결과 통계 출력"""
    print("\n" + "="*60)
    print("데이터 분할 결과 (클래스별)")
    print("="*60)

    for split_name, (paths, labels) in splits.items():
        fake_count = labels.count(0)
        real_count = labels.count(1)
        total = len(paths)

        print(f"\n{split_name.upper()} Set: {total:,}개")
        print(f"  FAKE: {fake_count:,}개 ({fake_count/total*100:.1f}%)")
        print(f"  REAL: {real_count:,}개 ({real_count/total*100:.1f}%)")

    print("\n" + "="*60)
    print("✅ 클래스 균형이 모든 세트에서 유지되고 있습니다!")
    print("="*60)


def save_to_csv(splits, output_dir):
    """분할된 데이터를 CSV로 저장"""
    print("\n" + "="*60)
    print("CSV 파일 저장 중...")
    print("="*60)

    for split_name, (paths, labels) in splits.items():
        df = pd.DataFrame({
            'image_path': [str(p) for p in paths],
            'label': labels
        })

        csv_path = output_dir / f'{split_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ {split_name}.csv ({len(df):,}개) 저장 완료")

    print(f"\n저장 위치: {output_dir}")
    print("="*60)


def visualize_split_distribution(splits, results_dir):
    """분할 결과 시각화"""
    print("\n시각화 생성 중...")

    # 데이터 준비
    datasets = []
    fake_counts = []
    real_counts = []

    for split_name, (paths, labels) in splits.items():
        datasets.append(split_name.capitalize())
        fake_counts.append(labels.count(0))
        real_counts.append(labels.count(1))

    # 막대 그래프
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#FF6B6B', '#4ECDC4']

    for ax, dataset, fake, real in zip(axes, datasets, fake_counts, real_counts):
        bars = ax.bar(['FAKE', 'REAL'], [fake, real], color=colors,
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title(f'{dataset} Set', fontsize=14, fontweight='bold')
        ax.set_ylabel('이미지 개수', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Train/Validation/Test 분할 결과', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = results_dir / 'data_split_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 분할 결과 그래프 저장: {save_path.name}")

    # 파이 차트
    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [len(paths) for paths, _ in splits.values()]
    labels_pie = ['Train\n(70%)', 'Validation\n(15%)', 'Test\n(15%)']
    colors_pie = ['#FF9999', '#66B2FF', '#99FF99']
    explode = (0.05, 0, 0)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels_pie,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    ax.set_title('데이터셋 분할 비율', fontsize=16, fontweight='bold', pad=20)

    save_path = results_dir / 'data_split_ratio.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 분할 비율 파이 차트 저장: {save_path.name}")


def print_summary(output_dir, results_dir):
    """작업 요약 출력"""
    print("\n" + "="*60)
    print("작업 완료!")
    print("="*60)
    print("\n📁 생성된 파일:")
    print(f"\nCSV 파일 ({output_dir}):")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print(f"\n시각화 파일 ({results_dir}):")
    print("  - data_split_distribution.png")
    print("  - data_split_ratio.png")
    print("\n🎯 다음 단계:")
    print("  Phase 4: PyTorch Dataset & DataLoader 구현")
    print("="*60)


def main():
    """메인 실행 함수"""
    print("="*60)
    print("AI 이미지 판별 - 데이터 전처리")
    print("="*60)
    print(f"Random Seed: {RANDOM_SEED}\n")

    # 1. 디렉토리 설정
    dirs = setup_directories()

    # 2. 이미지 경로 수집
    image_paths, labels = collect_image_paths(dirs)

    # 3. 데이터 분할
    splits = split_dataset(image_paths, labels)

    # 4. 통계 출력
    print_split_statistics(splits)

    # 5. CSV 저장
    save_to_csv(splits, dirs['output_dir'])

    # 6. 시각화
    visualize_split_distribution(splits, dirs['results_dir'])

    # 7. 요약
    print_summary(dirs['output_dir'], dirs['results_dir'])


if __name__ == "__main__":
    main()
