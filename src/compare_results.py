"""
Model Comparison Script

This script compares the performance of all models including ensemble models.

Usage:
    python src/compare_results.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results(metrics_dir):
    """
    Load all model results from CSV files

    Args:
        metrics_dir: Directory containing result CSV files

    Returns:
        DataFrame with all results
    """
    results = []

    # List of all models
    model_files = [
        'simple_cnn_test_results.csv',
        'resnet50_test_results.csv',
        'efficientnet_b0_test_results.csv',
        'vgg16_test_results.csv',
        'ensemble_soft_test_results.csv',
        'ensemble_hard_test_results.csv'
    ]

    for file in model_files:
        file_path = metrics_dir / file
        if file_path.exists():
            df = pd.read_csv(file_path)
            results.append(df)
        else:
            print(f"Warning: {file} not found, skipping...")

    if not results:
        raise FileNotFoundError("No result files found!")

    # Combine all results
    all_results = pd.concat(results, ignore_index=True)

    return all_results


def create_comparison_table(results_df):
    """
    Create and print comparison table

    Args:
        results_df: DataFrame with all results
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)

    # Sort by F1 score (or accuracy)
    results_df = results_df.sort_values('test_f1', ascending=False)

    # Format table
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        model = row['model']
        acc = f"{row['test_accuracy']:.4f}"
        prec = f"{row['test_precision']:.4f}"
        rec = f"{row['test_recall']:.4f}"
        f1 = f"{row['test_f1']:.4f}"
        roc = f"{row['roc_auc']:.4f}"

        # Highlight best model
        if _ == results_df.index[0]:
            print(f"{model:<20} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {roc:<12} [BEST]")
        else:
            print(f"{model:<20} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {roc:<12}")

    print("="*80)

    # Performance improvement
    if len(results_df) > 1:
        best_model = results_df.iloc[0]['model']
        best_f1 = results_df.iloc[0]['test_f1']

        # Find best individual model (not ensemble)
        individual_models = results_df[~results_df['model'].str.contains('ensemble')]
        if len(individual_models) > 0:
            best_individual = individual_models.iloc[0]['model']
            best_individual_f1 = individual_models.iloc[0]['test_f1']

            print(f"\nBest Overall Model: {best_model} (F1: {best_f1:.4f})")
            print(f"Best Individual Model: {best_individual} (F1: {best_individual_f1:.4f})")

            if 'ensemble' in best_model:
                improvement = (best_f1 - best_individual_f1) * 100
                print(f"Ensemble Improvement: {improvement:+.2f}% F1-Score")

    print("="*80)


def plot_model_comparison(results_df, save_dir):
    """
    Create comparison plots

    Args:
        results_df: DataFrame with all results
        save_dir: Directory to save plots
    """
    # Sort by F1 score
    results_df = results_df.sort_values('test_f1', ascending=False)

    # 1. Bar plot of all metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Color: ensemble models in different color
        colors = ['#2ecc71' if 'ensemble' in m else '#3498db'
                  for m in results_df['model']]

        bars = ax.barh(results_df['model'], results_df[metric], color=colors)

        ax.set_xlabel(title, fontsize=11)
        ax.set_xlim([min(results_df[metric]) - 0.01, 1.0])
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, results_df[metric])):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}',
                   va='center', fontsize=9)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    save_path = save_dir / 'model_comparison_all_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved: {save_path}")

    # 2. ROC-AUC comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71' if 'ensemble' in m else '#3498db'
              for m in results_df['model']]

    bars = ax.barh(results_df['model'], results_df['roc_auc'], color=colors)

    ax.set_xlabel('ROC-AUC Score', fontsize=12)
    ax.set_xlim([min(results_df['roc_auc']) - 0.001, 1.0])
    ax.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, results_df['roc_auc']):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}',
               va='center', fontsize=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Individual Models'),
        Patch(facecolor='#2ecc71', label='Ensemble Models')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    # Save
    save_path = save_dir / 'model_comparison_roc_auc.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC-AUC plot saved: {save_path}")


def create_summary_csv(results_df, save_path):
    """
    Create comprehensive summary CSV

    Args:
        results_df: DataFrame with all results
        save_path: Path to save CSV
    """
    # Sort by F1 score
    results_df = results_df.sort_values('test_f1', ascending=False)

    # Add rank column
    results_df['rank'] = range(1, len(results_df) + 1)

    # Reorder columns
    columns = [
        'rank', 'model', 'test_accuracy', 'test_precision',
        'test_recall', 'test_f1', 'roc_auc', 'test_loss'
    ]
    results_df = results_df[columns]

    # Save
    results_df.to_csv(save_path, index=False)
    print(f"\nSummary CSV saved: {save_path}")


def main():
    """Main function"""
    # Path settings
    project_root = Path(__file__).parent.parent
    metrics_dir = project_root / 'results' / 'metrics'
    figures_dir = project_root / 'results' / 'figures'

    print("="*80)
    print("MODEL RESULTS COMPARISON")
    print("="*80)

    # Load all results
    print("\nLoading results...")
    try:
        results_df = load_all_results(metrics_dir)
        print(f"Loaded {len(results_df)} model results")
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Create comparison table
    create_comparison_table(results_df)

    # Create plots
    print("\nCreating comparison plots...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results_df, figures_dir)

    # Create summary CSV
    summary_path = metrics_dir / 'all_models_comparison.csv'
    create_summary_csv(results_df, summary_path)

    print("\n" + "="*80)
    print("COMPARISON COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
