import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Creates DataLoaders for train and val datasets.
    
    Args:
        data_dir (str): Path to the data directory containing 'train' and 'val' subfolders.
        batch_size (int): Batch size for the dataloaders.
        img_size (int): Target image size for resizing.
        
    Returns:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        dataset_sizes (dict): Dictionary containing size of 'train' and 'val' datasets.
        class_names (list): List of class names.
    """
    

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet statistics    
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size + 32), 
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet statistics    
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet statistics    
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=batch_size,
                                 shuffle=True if x == 'train' else False, 
                                 num_workers=0,  
                                 pin_memory=True if torch.cuda.is_available() else False)
                   for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
