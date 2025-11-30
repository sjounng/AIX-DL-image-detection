"""
Data Preprocessing Script

This script splits the AI image detection dataset into Train/Validation/Test sets
and prepares the preprocessing pipeline.

Usage:
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

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_directories():
    """Setup project directories"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results' / 'figures'

    # Create output directories
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
    """Collect image file paths"""
    print("Collecting image file paths...")

    fake_images = list(dirs['fake_dir'].glob('*.jpg')) + list(dirs['fake_dir'].glob('*.png'))
    real_images = list(dirs['real_dir'].glob('*.jpg')) + list(dirs['real_dir'].glob('*.png'))

    print(f"\nFAKE images: {len(fake_images):,}")
    print(f"REAL images: {len(real_images):,}")
    print(f"Total images: {len(fake_images) + len(real_images):,}")

    # Create paths and labels
    image_paths = fake_images + real_images
    labels = [0] * len(fake_images) + [1] * len(real_images)  # 0: FAKE, 1: REAL

    return image_paths, labels


def split_dataset(image_paths, labels):
    """Split dataset into Train/Validation/Test"""
    print("\n" + "="*60)
    print("Splitting dataset...")
    print("="*60)

    # Train / (Val + Test) split (70% / 30%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    print(f"\nStep 1: Train / Temp split")
    print(f"  Train: {len(train_paths):,} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Temp:  {len(temp_paths):,} ({len(temp_paths)/len(image_paths)*100:.1f}%)")

    # Val / Test split (15% each)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,  # 50% of 30% = 15%
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    print(f"\nStep 2: Val / Test split")
    print(f"  Val:   {len(val_paths):,} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths):,} ({len(test_paths)/len(image_paths)*100:.1f}%)")

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def print_split_statistics(splits):
    """Print split statistics"""
    print("\n" + "="*60)
    print("Dataset Split Results (by class)")
    print("="*60)

    for split_name, (paths, labels) in splits.items():
        fake_count = labels.count(0)
        real_count = labels.count(1)
        total = len(paths)

        print(f"\n{split_name.upper()} Set: {total:,} samples")
        print(f"  FAKE: {fake_count:,} ({fake_count/total*100:.1f}%)")
        print(f"  REAL: {real_count:,} ({real_count/total*100:.1f}%)")

    print("\n" + "="*60)
    print("[OK] Class balance is maintained across all sets!")
    print("="*60)


def save_to_csv(splits, output_dir):
    """Save split data to CSV"""
    print("\n" + "="*60)
    print("Saving CSV files...")
    print("="*60)

    for split_name, (paths, labels) in splits.items():
        df = pd.DataFrame({
            'image_path': [str(p) for p in paths],
            'label': labels
        })

        csv_path = output_dir / f'{split_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  [OK] {split_name}.csv ({len(df):,} samples) saved")

    print(f"\nSaved to: {output_dir}")
    print("="*60)


def visualize_split_distribution(splits, results_dir):
    """Visualize split distribution"""
    print("\nGenerating visualizations...")

    # Prepare data
    datasets = []
    fake_counts = []
    real_counts = []

    for split_name, (paths, labels) in splits.items():
        datasets.append(split_name.capitalize())
        fake_counts.append(labels.count(0))
        real_counts.append(labels.count(1))

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#FF6B6B', '#4ECDC4']

    for ax, dataset, fake, real in zip(axes, datasets, fake_counts, real_counts):
        bars = ax.bar(['FAKE', 'REAL'], [fake, real], color=colors,
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title(f'{dataset} Set', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Display values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Train/Validation/Test Split Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = results_dir / 'data_split_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Split distribution graph saved: {save_path.name}")

    # Pie chart
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

    ax.set_title('Dataset Split Ratio', fontsize=16, fontweight='bold', pad=20)

    save_path = results_dir / 'data_split_ratio.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Split ratio pie chart saved: {save_path.name}")


def print_summary(output_dir, results_dir):
    """Print summary"""
    print("\n" + "="*60)
    print("Task Completed!")
    print("="*60)
    print("\n[FILES] Generated files:")
    print(f"\nCSV files ({output_dir}):")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print(f"\nVisualization files ({results_dir}):")
    print("  - data_split_distribution.png")
    print("  - data_split_ratio.png")
    print("\n[NEXT] Next step:")
    print("  Phase 4: Implement PyTorch Dataset & DataLoader")
    print("="*60)


def main():
    """Main execution function"""
    print("="*60)
    print("AI Image Detection - Data Preprocessing")
    print("="*60)
    print(f"Random Seed: {RANDOM_SEED}\n")

    # 1. Setup directories
    dirs = setup_directories()

    # 2. Collect image paths
    image_paths, labels = collect_image_paths(dirs)

    # 3. Split dataset
    splits = split_dataset(image_paths, labels)

    # 4. Print statistics
    print_split_statistics(splits)

    # 5. Save to CSV
    save_to_csv(splits, dirs['output_dir'])

    # 6. Visualize
    visualize_split_distribution(splits, dirs['results_dir'])

    # 7. Summary
    print_summary(dirs['output_dir'], dirs['results_dir'])


if __name__ == "__main__":
    main()
