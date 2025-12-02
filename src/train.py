import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import copy
import os
from dataset import get_dataloaders, set_seed
from models import get_model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()    

            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)


                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train AI vs Real Image Classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='simplecnn', 
                        choices=['simplecnn', 'resnet50', 'convnext', 'efficientnet'],
                        help='Model name to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    dataloaders, dataset_sizes, class_names = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    model = get_model(args.model_name, num_classes=len(class_names))
    model = model.to(device)
    

    if torch.cuda.is_available():
        print(f"Model loaded to GPU. Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=args.epochs)

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    main()
