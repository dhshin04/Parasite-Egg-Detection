import os
import json
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.fecal_egg_dataset import FecalEggDataset
from custom_transforms import transform
from split import even_train_test_split


def collate_fn(batch):
    # For handling varying-length image-target pair (especially different sized images)
    return tuple(zip(*batch))


def load_datasets(cv_test_split=0.5, device='cpu'):
    '''
    Preprocess Data: put images and annotations into Fecal Egg Dataset

    Arguments:
        cv_test_split (float): Ratio of validation and test set sizes
        device (str): cuda:0 or cpu
    Returns:
        (tuple): training set, validation set, test set in FecalEggDataset format
    '''
    # Load Image and Annotations
    testset_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
    test_images_path = os.path.join(testset_path, 'images')
    test_labels_path = os.path.join(testset_path, 'refined_labels.json')
    with open(test_labels_path, 'r') as refined_test_labels:
        test_annotations = json.load(refined_test_labels)

    # Prepare list of image names -> used by dataset and data loader
    train_images, test_images = even_train_test_split(
        sorted(os.listdir(test_images_path)),
        annotations=test_annotations,
        test_size=0.4,
        random_seed=10,
    )
        
    cv_images, test_images = train_test_split(
        test_images,       # images from test set
        test_size=cv_test_split,      # 50:50 split by default
        random_state=1,     # for consistent results
    )

    # Define Dataset and DataLoader
    train_dataset = FecalEggDataset(test_images_path, train_images, test_annotations, device=device, transforms=transform)
    validation_dataset = FecalEggDataset(test_images_path, cv_images, test_annotations, device=device)
    test_dataset = FecalEggDataset(test_images_path, test_images, test_annotations, device=device)

    return train_dataset, validation_dataset, test_dataset


def get_data_loaders(cv_test_split=0.5, train_batch=8, cv_batch=8, test_batch=8, device='cpu'):
    '''
    Turn Dataset into DataLoader

    Arguments:
        cv_test_split (float): Ratio of validation and test sets
        train_batch (int): Batch size for train loader
        cv_batch (int): Batch size for validation loader
        test_batch (int): Batch size for test loader
        device (str): cuda:0 or cpu
    Returns:
        (tuple): Data Loaders for training, validation, and test sets
    '''

    # Load Dataset
    train_dataset, validation_dataset, test_dataset = load_datasets(cv_test_split, device)

    # Create Data Loaders
    train_loader = DataLoader(      # With Params for GPU Acceleration
        dataset=train_dataset, 
        batch_size=train_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=6,              # To speed up data transfer between CPU and GPU
        persistent_workers=True,    # Keep workers alive for next batch
        pin_memory=True,            # Allocate tensors in page-locked memory for faster transfer
    )
    validation_loader = DataLoader(      
        dataset=validation_dataset, 
        batch_size=cv_batch,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = DataLoader(      
        dataset=test_dataset, 
        batch_size=test_batch,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, validation_loader, test_loader
