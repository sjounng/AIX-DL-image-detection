"""
Ensemble Model Evaluation Script

This script evaluates an ensemble of the top 2 performing models
(EfficientNet-B0 and ResNet50) on the test set.

Usage:
    python src/ensemble.py --method soft
    python src/ensemble.py --method hard --batch-size 64
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
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


class EnsembleModel:
    """
    Ensemble model wrapper for combining multiple models

    Args:
        models: List of PyTorch models
        method: Ensemble method ('soft' or 'hard')
            - soft: Average predicted probabilities (soft voting)
            - hard: Majority voting on predicted classes (hard voting)
    """

    def __init__(self, models, method='soft'):
        self.models = models
        self.method = method

        if method not in ['soft', 'hard']:
            raise ValueError("method must be 'soft' or 'hard'")

    def predict(self, images, device):
        """
        Make predictions using ensemble

        Args:
            images: Input images tensor
            device: Device

        Returns:
            predictions: Predicted class labels
            probabilities: Predicted probabilities for class 1
        """
        all_outputs = []
        all_probs = []

        # Get predictions from each model
        with torch.no_grad():
            for model in self.models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_outputs.append(outputs)
                all_probs.append(probs)

        # Stack predictions
        all_probs = torch.stack(all_probs)  # [num_models, batch_size, num_classes]

        if self.method == 'soft':
            # Soft voting: Average probabilities
            avg_probs = all_probs.mean(dim=0)  # [batch_size, num_classes]
            predictions = avg_probs.argmax(dim=1)
            probabilities = avg_probs[:, 1]  # Probability of class 1

        else:  # hard voting
            # Hard voting: Majority vote
            individual_preds = all_probs.argmax(dim=2)  # [num_models, batch_size]
            predictions = individual_preds.mode(dim=0).values  # Majority vote
            # Use average probability for ROC curve
            avg_probs = all_probs.mean(dim=0)
            probabilities = avg_probs[:, 1]

        return predictions, probabilities


def evaluate_ensemble(ensemble, dataloader, criterion, device):
    """
    Evaluate ensemble model on test set

    Args:
        ensemble: EnsembleModel instance
        dataloader: Test DataLoader
        criterion: Loss function
        device: Device

    Returns:
        dict: Evaluation results
    """
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    print("\nEvaluating ensemble on test set...")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Ensemble prediction
            preds, probs = ensemble.predict(images, device)

            # Calculate loss using average probability
            # (for soft voting, this is the actual ensemble output)
            loss_value = 0.0
            for model in ensemble.models:
                outputs = model(images)
                loss_value += criterion(outputs, labels).item()
            loss_value /= len(ensemble.models)

            # Store results
            running_loss += loss_value * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

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


def plot_confusion_matrix(labels, predictions, save_path, method):
    """Plot confusion matrix"""
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
    plt.title(f'Confusion Matrix - Ensemble ({method.upper()} voting)',
              fontsize=14, fontweight='bold', pad=20)
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


def plot_roc_curve(labels, probabilities, save_path, method):
    """Plot ROC curve"""
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
    plt.title(f'ROC Curve - Ensemble ({method.upper()} voting)',
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved: {save_path}")

    return roc_auc


def print_classification_report(labels, predictions):
    """Print detailed classification report"""
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


def save_results_to_csv(results, save_path, method):
    """Save evaluation results to CSV"""
    df = pd.DataFrame({
        'model': [f'ensemble_{method}'],
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
    parser = argparse.ArgumentParser(
        description='AI Image Detection Ensemble Model Evaluation'
    )
    parser.add_argument('--method', type=str, default='soft',
                       choices=['soft', 'hard'],
                       help='Ensemble method (soft/hard voting)')
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

    print("="*60)
    print("Ensemble Model Evaluation")
    print("="*60)
    print(f"Using device: {device}")
    print(f"Ensemble method: {args.method.upper()} voting")
    print("="*60)

    # Create DataLoader
    print("\nCreating DataLoader...")
    dataloaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_loader = dataloaders['test']
    print(f"Test set size: {len(test_loader.dataset):,}")

    # Models to ensemble (top 2 performers)
    model_names = ['efficientnet_b0', 'resnet50']
    models = []

    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)

    for model_name in model_names:
        print(f"\nLoading {model_name}...")

        # Create model
        model = get_model(model_name, num_classes=2, pretrained=False)

        # Load checkpoint
        checkpoint_path = model_dir / f'{model_name}_best.pth'

        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("Please train the model first using train.py")
            return

        model = load_checkpoint(model, checkpoint_path, device)
        model = model.to(device)
        models.append(model)

        print(f"{model_name} loaded successfully")

    print("\n" + "="*60)
    print(f"Ensemble: {' + '.join(model_names)}")
    print("="*60)

    # Create ensemble
    ensemble = EnsembleModel(models, method=args.method)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate ensemble
    print("\n" + "="*60)
    print(f"Evaluating Ensemble on Test Set ({args.method.upper()} voting)")
    print("="*60)

    results = evaluate_ensemble(ensemble, test_loader, criterion, device)

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
    cm_path = figures_dir / f'ensemble_{args.method}_confusion_matrix.png'
    plot_confusion_matrix(results['labels'], results['predictions'], cm_path, args.method)

    # Plot ROC curve
    roc_path = figures_dir / f'ensemble_{args.method}_roc_curve.png'
    roc_auc = plot_roc_curve(results['labels'], results['probabilities'], roc_path, args.method)
    results['roc_auc'] = roc_auc

    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # Save results to CSV
    csv_path = metrics_dir / f'ensemble_{args.method}_test_results.csv'
    save_results_to_csv(results, csv_path, args.method)

    print("\n" + "="*60)
    print("Evaluation Completed!")
    print("="*60)
    print(f"Confusion Matrix: {cm_path}")
    print(f"ROC Curve: {roc_path}")
    print(f"Results CSV: {csv_path}")
    print("="*60)


if __name__ == "__main__":
    main()
