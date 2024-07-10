import os
import json
import multiprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.fecal_egg_dataset import FecalEggDataset


def load_datasets(cv_test_split=0.5, device='cpu'):
    # Load Image and Annotations
    trainingset_path = os.path.join(os.path.dirname(__file__), 'data', 'trainingset')
    training_images_path = os.path.join(trainingset_path, 'images')
    training_labels_path = os.path.join(trainingset_path, 'refined_labels.json')
    with open(training_labels_path, 'r') as refined_labels:
        train_annotations = json.load(refined_labels)

    testset_path = os.path.join(os.path.dirname(__file__), 'data', 'testset')
    test_images_path = os.path.join(testset_path, 'images')
    test_labels_path = os.path.join(testset_path, 'refined_test_labels.json')
    with open(test_labels_path, 'r') as refined_test_labels:
        test_annotations = json.load(refined_test_labels)

    # Prepare list of image names -> used by dataset and data loader
    train_images = sorted(os.listdir(training_images_path))     # images from train set
    cv_images, test_images = train_test_split(
        sorted(os.listdir(test_images_path)),       # images from test set
        test_size=cv_test_split,      # 50:50 split by default
        random_state=1,     # for consistent results
    )

    # Define Dataset and DataLoader
    train_dataset = FecalEggDataset(training_images_path, train_images, train_annotations, device=device)
    validation_dataset = FecalEggDataset(test_images_path, cv_images, test_annotations, device=device)
    test_dataset = FecalEggDataset(test_images_path, test_images, test_annotations, device=device)

    return train_dataset, validation_dataset, test_dataset


def get_data_loaders(cv_test_split=0.5, train_batch=15, cv_batch=10, test_batch=10, device='cpu'):
    train_dataset, validation_dataset, test_dataset = load_datasets(cv_test_split=cv_test_split, device=device)

    train_loader = DataLoader(      # With Params for GPU Acceleration
        dataset=train_dataset, 
        batch_size=train_batch,      # Mini-Batch Gradient Descent
        num_workers=multiprocessing.cpu_count() // 2,   # Num_subprocesses to use for data-loading
        persistent_workers=True,            # Keeps worker processes on for next iteration
        pin_memory='cuda' in device,        # Data loaded is added to page-locked memory -> efficient transfer to GPU
        pin_memory_device=device if 'cuda' in device else '',   # Device where data should be loaded -> CUDA if available
        collate_fn=lambda batch: tuple(zip(*batch)),        # To handle varying length image-target pair
        shuffle=True,
    )
    validation_loader = DataLoader(      # With Params for GPU Acceleration
        dataset=validation_dataset, 
        batch_size=cv_batch,      # Mini-Batch Gradient Descent
        num_workers=multiprocessing.cpu_count() // 2,   # Num_subprocesses to use for data-loading
        persistent_workers=True,            # Keeps worker processes on for next iteration
        pin_memory='cuda' in device,        # Data loaded is added to page-locked memory -> efficient transfer to GPU
        pin_memory_device=device if 'cuda' in device else '',   # Device where data should be loaded -> CUDA if available
        collate_fn=lambda batch: tuple(zip(*batch)),        # To handle varying length image-target pair
        shuffle=True,
    )
    test_loader = DataLoader(      # With Params for GPU Acceleration
        dataset=test_dataset, 
        batch_size=test_batch,      # Mini-Batch Gradient Descent
        num_workers=multiprocessing.cpu_count() // 2,   # Num_subprocesses to use for data-loading
        persistent_workers=True,            # Keeps worker processes on for next iteration
        pin_memory='cuda' in device,        # Data loaded is added to page-locked memory -> efficient transfer to GPU
        pin_memory_device=device if 'cuda' in device else '',   # Device where data should be loaded -> CUDA if available
        collate_fn=lambda batch: tuple(zip(*batch)),        # To handle varying length image-target pair
        shuffle=True,
    )

    return train_loader, validation_loader, test_loader
