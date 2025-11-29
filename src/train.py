"""
모델 학습 스크립트

이 스크립트는 AI 이미지 판별 모델을 학습합니다.

사용 예시:
    python src/train.py --model resnet50 --epochs 50 --batch-size 32
    python src/train.py --model efficientnet_b0 --lr 0.0001
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models import get_model
from data_loader import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    1 epoch 학습

    Args:
        model: PyTorch 모델
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: 디바이스 (cuda/mps/cpu)

    Returns:
        dict: 학습 loss와 accuracy
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Progress bar
    pbar = tqdm(dataloader, desc='Training')

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # 통계
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Progress bar 업데이트
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Epoch 통계
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(model, dataloader, criterion, device):
    """
    검증

    Args:
        model: PyTorch 모델
        dataloader: Validation DataLoader
        criterion: Loss function
        device: 디바이스

    Returns:
        dict: 검증 loss, accuracy, precision, recall, f1
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 통계
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 평가 지표 계산
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(model, dataloaders, criterion, optimizer, scheduler,
                device, num_epochs, save_dir, model_name):
    """
    모델 학습 메인 함수

    Args:
        model: PyTorch 모델
        dataloaders: Train/Val DataLoader 딕셔너리
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: 디바이스
        num_epochs: 학습 epoch 수
        save_dir: 모델 저장 디렉토리
        model_name: 모델 이름

    Returns:
        dict: 학습 기록
    """
    # 저장 디렉토리 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 학습 기록
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'lr': []
    }

    # Early stopping 설정
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    print("="*60)
    print(f"학습 시작: {model_name}")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Train size: {len(dataloaders['train'].dataset):,}")
    print(f"Val size: {len(dataloaders['val'].dataset):,}")
    print("="*60)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        # 학습
        train_metrics = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )

        # 검증
        val_metrics = validate(
            model, dataloaders['val'], criterion, device
        )

        # Learning rate scheduler step
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # 기록 저장
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)

        # 결과 출력
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | "
              f"Val Recall: {val_metrics['recall']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Best model 저장
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0

            # 모델 저장
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            }
            save_path = save_dir / f'{model_name}_best.pth'
            torch.save(checkpoint, save_path)
            print(f"✓ Best model saved! (Val Loss: {val_metrics['loss']:.4f})")

        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("\n" + "="*60)
                print("Early stopping triggered!")
                print("="*60)
                break

    # 학습 종료
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"총 소요 시간: {elapsed_time/60:.2f} 분")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print("="*60)

    return history


def plot_training_history(history, save_path):
    """
    학습 곡선 시각화

    Args:
        history: 학습 기록
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Metrics (Precision, Recall, F1)
    axes[1, 0].plot(history['val_precision'], label='Precision', marker='o')
    axes[1, 0].plot(history['val_recall'], label='Recall', marker='s')
    axes[1, 0].plot(history['val_f1'], label='F1-Score', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(history['lr'], marker='o', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"학습 곡선 저장: {save_path}")


def save_training_history(history, save_path):
    """
    학습 기록을 CSV로 저장

    Args:
        history: 학습 기록
        save_path: 저장 경로
    """
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"학습 기록 저장: {save_path}")


def main():
    """메인 함수"""
    # Argument parser
    parser = argparse.ArgumentParser(description='AI Image Detection Training')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['simple_cnn', 'resnet50', 'efficientnet_b0', 'vgg16'],
                       help='Model name')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Do not use pretrained weights')

    args = parser.parse_args()

    # 경로 설정
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'
    model_save_dir = project_root / 'models'
    results_dir = project_root / 'results'

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\n사용 디바이스: {device}")

    # DataLoader 생성
    print("\nDataLoader 생성 중...")
    dataloaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 모델 생성
    print(f"\n모델 생성 중: {args.model}")
    model = get_model(
        args.model,
        num_classes=2,
        pretrained=args.pretrained
    )
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # 학습
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=model_save_dir,
        model_name=args.model
    )

    # 결과 저장
    print("\n결과 저장 중...")

    # 학습 곡선 그래프
    plot_path = results_dir / 'figures' / f'{args.model}_training_curves.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, plot_path)

    # 학습 기록 CSV
    csv_path = results_dir / 'metrics' / f'{args.model}_training_history.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_history(history, csv_path)

    print("\n" + "="*60)
    print("모든 작업 완료!")
    print("="*60)
    print(f"모델 저장: models/{args.model}_best.pth")
    print(f"학습 곡선: {plot_path}")
    print(f"학습 기록: {csv_path}")
    print("="*60)


if __name__ == "__main__":
    main()
