import os
import json
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from data.fecal_egg_dataset import FecalEggDataset


# For handling varying-length image-target pair (especially different sized images)
def collate_fn(batch):
    return tuple(zip(*batch))


def scale_training(img_list, scale_train_set=1.0):
    new_img_list = []
    for i in range(11):     # 11 categories
        index_list = random.sample(range(i * 1000, (i + 1) * 1000), int(1000 * scale_train_set))
        for index in index_list:
            new_img_list.append(img_list[index])
    return new_img_list


def load_datasets(cv_test_split=0.5, device='cpu', scale_train_set=1.0):
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
    train_images = scale_training(
        sorted(os.listdir(training_images_path)),     # images from train set
        scale_train_set,
    )
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


def get_data_loaders(cv_test_split=0.5, train_batch=5, cv_batch=4, test_batch=4, device='cpu', scale_train_set=1.0):
    train_dataset, validation_dataset, test_dataset = load_datasets(cv_test_split, device , scale_train_set)

    dataloader_args = {
        # 'num_workers': 2,   # Num_subprocesses to use for data-loading
        # 'persistent_workers': True,            # Keeps worker processes on for next iteration
        'collate_fn': collate_fn,        # To handle varying length image-target pair
        # 'pin_memory': device.type == 'cuda',
    }

    train_loader = DataLoader(      # With Params for GPU Acceleration
        dataset=train_dataset, 
        batch_size=train_batch,
        **dataloader_args,
    )
    validation_loader = DataLoader(      
        dataset=validation_dataset, 
        batch_size=cv_batch,
        **dataloader_args,
    )
    test_loader = DataLoader(      
        dataset=test_dataset, 
        batch_size=test_batch,
        **dataloader_args,
    )

    return train_loader, validation_loader, test_loader
