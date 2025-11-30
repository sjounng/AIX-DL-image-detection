"""
Model Evaluation Script

This script evaluates the trained AI image detection model on the test set.

Usage:
    python src/evaluate.py --model efficientnet_b0
    python src/evaluate.py --model resnet50 --batch-size 64
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)

from models import get_model
from data_loader import get_dataloaders


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device (cuda/mps/cpu)

    Returns:
        model: Model with loaded weights
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("="*60)
    print("Checkpoint Loaded")
    print("="*60)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Validation F1: {checkpoint['val_f1']:.4f}")
    print("="*60)

    return model, checkpoint


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on test set

    Args:
        model: PyTorch model
        dataloader: Test DataLoader
        criterion: Loss function
        device: Device

    Returns:
        dict: Evaluation results
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    print("\nEvaluating on test set...")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store results
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (REAL)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate metrics
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def plot_confusion_matrix(labels, predictions, save_path):
    """
    Plot confusion matrix

    Args:
        labels: True labels
        predictions: Predicted labels
        save_path: Save path
    """
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['FAKE (0)', 'REAL (1)'],
        yticklabels=['FAKE (0)', 'REAL (1)'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add accuracy for each class
    tn, fp, fn, tp = cm.ravel()
    fake_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    real_acc = tp / (tp + fn) if (tp + fn) > 0 else 0

    plt.text(0.5, -0.15, f'FAKE Accuracy: {fake_acc:.2%}',
             ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.20, f'REAL Accuracy: {real_acc:.2%}',
             ha='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(labels, probabilities, save_path):
    """
    Plot ROC curve

    Args:
        labels: True labels
        probabilities: Predicted probabilities
        save_path: Save path
    """
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve',
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved: {save_path}")

    return roc_auc


def print_classification_report(labels, predictions):
    """
    Print detailed classification report

    Args:
        labels: True labels
        predictions: Predicted labels
    """
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)

    report = classification_report(
        labels,
        predictions,
        target_names=['FAKE', 'REAL'],
        digits=4
    )
    print(report)


def save_results_to_csv(results, save_path, model_name):
    """
    Save evaluation results to CSV

    Args:
        results: Evaluation results dictionary
        save_path: Save path
        model_name: Model name
    """
    df = pd.DataFrame({
        'model': [model_name],
        'test_loss': [results['loss']],
        'test_accuracy': [results['accuracy']],
        'test_precision': [results['precision']],
        'test_recall': [results['recall']],
        'test_f1': [results['f1']],
        'roc_auc': [results.get('roc_auc', 0.0)]
    })

    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")


def main():
    """Main function"""
    # Argument parser
    parser = argparse.ArgumentParser(description='AI Image Detection Model Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       choices=['simple_cnn', 'resnet50', 'efficientnet_b0', 'vgg16'],
                       help='Model name')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Path settings
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'
    model_dir = project_root / 'models'
    results_dir = project_root / 'results'

    # Device settings
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}")

    # Create DataLoader
    print("\nCreating DataLoader...")
    dataloaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_loader = dataloaders['test']
    print(f"Test set size: {len(test_loader.dataset):,}")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, num_classes=2, pretrained=False)

    # Load checkpoint
    checkpoint_path = model_dir / f'{args.model}_best.pth'

    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    model, checkpoint = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    print("\n" + "="*60)
    print(f"Evaluating {args.model} on Test Set")
    print("="*60)

    results = evaluate_model(model, test_loader, criterion, device)

    # Print results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test F1-Score: {results['f1']:.4f}")
    print("="*60)

    # Print classification report
    print_classification_report(results['labels'], results['predictions'])

    # Create output directories
    figures_dir = results_dir / 'figures'
    metrics_dir = results_dir / 'metrics'
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    cm_path = figures_dir / f'{args.model}_confusion_matrix.png'
    plot_confusion_matrix(results['labels'], results['predictions'], cm_path)

    # Plot ROC curve
    roc_path = figures_dir / f'{args.model}_roc_curve.png'
    roc_auc = plot_roc_curve(results['labels'], results['probabilities'], roc_path)
    results['roc_auc'] = roc_auc

    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # Save results to CSV
    csv_path = metrics_dir / f'{args.model}_test_results.csv'
    save_results_to_csv(results, csv_path, args.model)

    print("\n" + "="*60)
    print("Evaluation Completed!")
    print("="*60)
    print(f"Confusion Matrix: {cm_path}")
    print(f"ROC Curve: {roc_path}")
    print(f"Results CSV: {csv_path}")
    print("="*60)


if __name__ == "__main__":
    main()
