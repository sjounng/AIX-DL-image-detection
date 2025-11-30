"""
Model Inference Script

This script performs inference on new images using trained models.

Usage:
    # Single image
    python src/inference.py --model efficientnet_b0 --image path/to/image.jpg

    # Multiple images
    python src/inference.py --model efficientnet_b0 --image img1.jpg img2.jpg img3.jpg

    # Directory of images
    python src/inference.py --model efficientnet_b0 --image-dir path/to/images/
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm

from models import get_model
from data_loader import get_transforms


def load_model(model_name, checkpoint_path, device):
    """
    Load trained model from checkpoint

    Args:
        model_name: Model name
        checkpoint_path: Path to checkpoint file
        device: Device (cuda/mps/cpu)

    Returns:
        model: Loaded model in eval mode
    """
    model = get_model(model_name, num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("="*60)
    print("Model Loaded Successfully")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Checkpoint Epoch: {checkpoint['epoch']}")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.4f} ({checkpoint['val_acc']*100:.2f}%)")
    print(f"Validation F1: {checkpoint['val_f1']:.4f}")
    print("="*60)

    return model


def predict_image(model, image_path, transform, device):
    """
    Predict single image

    Args:
        model: Trained model
        image_path: Path to image file
        transform: Image transformation
        device: Device

    Returns:
        dict: Prediction results
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    # Get probabilities for both classes
    fake_prob = probabilities[0][0].item()
    real_prob = probabilities[0][1].item()

    return {
        'image_path': str(image_path),
        'prediction': 'REAL' if prediction == 1 else 'FAKE',
        'confidence': confidence,
        'fake_probability': fake_prob,
        'real_probability': real_prob
    }


def predict_batch(model, image_paths, transform, device):
    """
    Predict multiple images

    Args:
        model: Trained model
        image_paths: List of image paths
        transform: Image transformation
        device: Device

    Returns:
        list: List of prediction results
    """
    results = []

    print("\nProcessing images...")
    for img_path in tqdm(image_paths, desc='Predicting'):
        try:
            result = predict_image(model, img_path, transform, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'image_path': str(img_path),
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_probability': 0.0,
                'real_probability': 0.0,
                'error': str(e)
            })

    return results


def print_results(results):
    """
    Print prediction results in a formatted way

    Args:
        results: List of prediction results
    """
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {Path(result['image_path']).name}")
        print("-" * 60)

        if result['prediction'] == 'ERROR':
            print(f"ERROR: {result.get('error', 'Unknown error')}")
            continue

        # Prediction
        prediction = result['prediction']
        confidence = result['confidence']

        # Color coding (won't show in terminal but good for structure)
        if prediction == 'FAKE':
            symbol = "[FAKE]"
        else:
            symbol = "[REAL]"

        print(f"Prediction: {symbol} {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nProbabilities:")
        print(f"  FAKE: {result['fake_probability']:.2%}")
        print(f"  REAL: {result['real_probability']:.2%}")


def save_results_to_csv(results, save_path):
    """
    Save results to CSV file

    Args:
        results: List of prediction results
        save_path: Path to save CSV
    """
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")


def main():
    """Main function"""
    # Argument parser
    parser = argparse.ArgumentParser(description='AI Image Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       choices=['simple_cnn', 'resnet50', 'efficientnet_b0', 'vgg16'],
                       help='Model name')
    parser.add_argument('--image', type=str, nargs='+',
                       help='Path to image file(s)')
    parser.add_argument('--image-dir', type=str,
                       help='Directory containing images')
    parser.add_argument('--output', type=str,
                       help='Output CSV file path (optional)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (currently not used, for future)')

    args = parser.parse_args()

    # Check if either image or image-dir is provided
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be provided")

    # Path settings
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'
    checkpoint_path = model_dir / f'{args.model}_best.pth'

    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    # Device settings
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}\n")

    # Load model
    model = load_model(args.model, checkpoint_path, device)

    # Get transform (use validation transform, no augmentation)
    transforms_dict = get_transforms()
    transform = transforms_dict['val_test']

    # Collect image paths
    image_paths = []

    if args.image:
        # From command line arguments
        image_paths = [Path(img) for img in args.image]

    if args.image_dir:
        # From directory
        img_dir = Path(args.image_dir)
        if not img_dir.exists():
            print(f"Error: Directory not found: {img_dir}")
            return

        # Common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
        for ext in extensions:
            image_paths.extend(img_dir.glob(ext))
            image_paths.extend(img_dir.glob(ext.upper()))

    # Remove duplicates and check if files exist
    image_paths = list(set(image_paths))
    image_paths = [p for p in image_paths if p.exists()]

    if not image_paths:
        print("Error: No valid images found")
        return

    print(f"\nFound {len(image_paths)} image(s) to process\n")

    # Predict
    results = predict_batch(model, image_paths, transform, device)

    # Print results
    print_results(results)

    # Save to CSV if requested
    if args.output:
        save_path = Path(args.output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_to_csv(results, save_path)
    else:
        # Default save location
        results_dir = project_root / 'results' / 'predictions'
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f'{args.model}_predictions.csv'
        save_results_to_csv(results, save_path)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_count = sum(1 for r in results if r['prediction'] == 'REAL')
    error_count = sum(1 for r in results if r['prediction'] == 'ERROR')

    print(f"Total Images: {len(results)}")
    print(f"  FAKE: {fake_count}")
    print(f"  REAL: {real_count}")
    if error_count > 0:
        print(f"  ERRORS: {error_count}")
    print("="*60)


if __name__ == "__main__":
    main()
