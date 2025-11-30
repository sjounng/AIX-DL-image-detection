"""
PyTorch Dataset and DataLoader Implementation

This module provides functionality for loading and preprocessing the AI image detection dataset.

Main classes:
    - AIImageDataset: Custom PyTorch Dataset
    - get_transforms: Create transform pipeline
    - get_dataloaders: Create Train/Val/Test DataLoaders
"""

import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AIImageDataset(Dataset):
    """
    Custom Dataset class for AI image detection

    Args:
        csv_file (str or Path): CSV file containing image paths and labels
        transform (callable, optional): Image transformation function

    Attributes:
        data (DataFrame): Image paths and labels data
        transform: Image preprocessing function
    """

    def __init__(self, csv_file, transform=None):
        """
        Initialize Dataset

        Args:
            csv_file: CSV file path
            transform: torchvision transforms
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """Return dataset size"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return image and label for the given index

        Args:
            idx (int): Data index

        Returns:
            tuple: (image_tensor, label)
                - image_tensor: Preprocessed image tensor
                - label: Class label (0: FAKE, 1: REAL)
        """
        # Image path and label
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    """
    Create image preprocessing transform pipeline

    Args:
        image_size (int): Image resize dimension (default: 224)

    Returns:
        dict: Dictionary containing 'train' and 'val_test' transforms
    """
    # ImageNet mean and std
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training Transform (with data augmentation)
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

    # Validation & Test Transform (without augmentation)
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
    Create Train/Validation/Test DataLoaders

    Args:
        data_dir (str or Path): Directory containing CSV files (data/processed/)
        batch_size (int): Batch size (default: 32)
        num_workers (int): Number of data loading workers (default: 4)
        image_size (int): Image size (default: 224)

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders
    """
    data_dir = Path(data_dir)

    # Create transforms
    transforms_dict = get_transforms(image_size)

    # Create datasets
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for testing
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
    DataLoader test function

    Verify that the DataLoader is working correctly.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'

    print("="*60)
    print("DataLoader Test")
    print("="*60)

    # Create dataloaders
    dataloaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=16,
        num_workers=2,
        image_size=224
    )

    # Print dataloader info
    for split_name, loader in dataloaders.items():
        print(f"\n{split_name.upper()} Loader:")
        print(f"  Dataset size: {len(loader.dataset):,}")
        print(f"  Batch size: {loader.batch_size}")
        print(f"  Number of batches: {len(loader)}")

    # Get sample batch from train loader
    print("\n" + "="*60)
    print("Sample Batch Check")
    print("="*60)

    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))

    print(f"\nBatch shape:")
    print(f"  Images: {images.shape}")  # [batch_size, 3, 224, 224]
    print(f"  Labels: {labels.shape}")  # [batch_size]

    print(f"\nBatch contents:")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image min/max: {images.min():.3f} / {images.max():.3f}")
    print(f"  Labels: {labels[:8].tolist()}")  # First 8 labels

    # Label distribution
    fake_count = (labels == 0).sum().item()
    real_count = (labels == 1).sum().item()
    print(f"\nLabel distribution in batch:")
    print(f"  FAKE (0): {fake_count} samples")
    print(f"  REAL (1): {real_count} samples")

    # Image visualization
    print("\nVisualizing images...")

    # Denormalization mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Denormalize
            img = images[i] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = img.clip(0, 1)

            ax.imshow(img)
            label_name = 'FAKE' if labels[i] == 0 else 'REAL'
            ax.set_title(f'{label_name}', fontsize=10)
            ax.axis('off')

    plt.suptitle('DataLoader Sample Images', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_path = project_root / 'results' / 'figures' / 'dataloader_test.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved: {save_path}")

    print("\n" + "="*60)
    print("[OK] DataLoader test completed!")
    print("="*60)


if __name__ == "__main__":
    # Run test
    test_dataloader()
