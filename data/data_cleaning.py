import os
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from fecal_egg_dataset import FecalEggDataset


def plot_image_with_bbox(image, annotation):
    fig, ax = plt.subplots(1)
    image = image.permute(1, 2, 0)
    ax.imshow(image)

    for box in annotation['boxes']:
        box = box.numpy()
    
        # Rectangular patch in COCO format
        patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(patch)
    ax.set_title(f'Image ID: {annotation['image_id']}')
    plt.show()


trainingset_path = os.path.join(os.path.dirname(__file__), 'testset')
training_images_path = os.path.join(trainingset_path, 'images')
training_labels_path = os.path.join(trainingset_path, 'refined_test_labels.json')
with open(training_labels_path, 'r') as refined_labels:
    train_annotations = json.load(refined_labels)

train_images = sorted(os.listdir(training_images_path))
train_dataset = FecalEggDataset(training_images_path, train_images, train_annotations)

for idx in range(len(train_dataset)):
    image, target = train_dataset.__getitem__(idx)
    plot_image_with_bbox(image, target)
    