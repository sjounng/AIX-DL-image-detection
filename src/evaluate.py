import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from dataset import get_dataloaders
from models import get_model

def evaluate(model, dataloader, device):
    model.eval()
    running_corrects = 0
    total = 0
    
    # For confusion matrix and ROC curve
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            # Store predictions and labels for visualization
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = running_corrects.double() / total
    print(f'Validation Accuracy: {acc:.4f}')
    
    return all_labels, all_preds, all_probs

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Confusion matrix를 생성하고 이미지로 저장합니다.
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        output_path: 저장할 이미지 경로
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def plot_roc_curve(y_true, y_probs, class_names, output_path):
    """
    ROC curve를 생성하고 이미지로 저장합니다.
    
    Args:
        y_true: 실제 레이블
        y_probs: 각 클래스에 대한 예측 확률 (shape: [n_samples, n_classes])
        class_names: 클래스 이름 리스트
        output_path: 저장할 이미지 경로
    """
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.2f})')
    

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate AI vs Real Image Classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='simplecnn', 
                        choices=['simplecnn', 'resnet50', 'convnext', 'efficientnet'],
                        help='Model architecture used for training')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, _, class_names = get_dataloaders(args.data_dir, args.batch_size)
    
    model = get_model(args.model_name, num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    print(f"Evaluating model: {args.model_name} from {args.model_path}")
    
    split_to_eval = 'test' if 'test' in dataloaders else 'val'
    print(f"Evaluating on split: {split_to_eval}")
    
    all_labels, all_preds, all_probs = evaluate(model, dataloaders[split_to_eval], device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Save results
    np.save(os.path.join(args.output_dir, 'true_labels.npy'), all_labels)
    np.save(os.path.join(args.output_dir, 'pred_labels.npy'), all_preds)
    np.save(os.path.join(args.output_dir, 'pred_probs.npy'), all_probs)
    
    # Generate and save visualizations
    print(f"\n=== Generating Visualizations ===")
    
    # Confusion Matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    
    # ROC Curve
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(all_labels, all_probs, class_names, roc_path)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Class names: {class_names}")
    print(f"\nSaved evaluation data to '{args.output_dir}/':")
    print(f"\nSaved visualizations:")
    print(f"  - confusion_matrix.png: Confusion Matrix")
    print(f"  - roc_curve.png: ROC Curve")

if __name__ == '__main__':
    main()

