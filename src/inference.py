import torch
import argparse
from torchvision import transforms
from PIL import Image
from models import get_model
import os

def load_image(image_path, img_size=224):
    """
    Loads and transforms an image for inference.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # Add batch dimension
    return image

def predict(model, image_tensor, device, class_names):
    """
    Performs inference on the image tensor.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    return class_names[predicted_class.item()], confidence.item()

def main():
    parser = argparse.ArgumentParser(description='Inference for AI vs Real Image Classifier')
    parser.add_argument('image_path', type=str, help='Path to the image for inference')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model weights')
    parser.add_argument('--model_name', type=str, default='simplecnn', 
                        choices=['simplecnn', 'resnet50', 'convnext', 'efficientnet'],
                        help='Model architecture name')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for transforms')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Class names (must match training order)
    class_names = ['FAKE', 'REAL']
    
    # Load Model
    try:
        model = get_model(args.model_name, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and Preprocess Image
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return
        
    try:
        image_tensor = load_image(args.image_path, args.img_size)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # Predict
    predicted_class, confidence = predict(model, image_tensor, device, class_names)
    
    print("-" * 20)
    print(f"Prediction Result:")
    print(f"Image: {args.image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 20)

if __name__ == '__main__':
    main()
