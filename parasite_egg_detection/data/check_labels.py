''' Check whether annotations in labels.json is appropriate '''

import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def show_image_with_bbox(image_path, annotation):
    '''
    Plot image with corresponding bounding boxes

    Arguments:
        image (tensor): Image tensor to plot
        annotation (dict): Dictionary containing annotations for image
    '''

    # Plot Image
    fig, ax = plt.subplots(1)
    image = Image.open(image_path)
    ax.imshow(image)

    # Plot bounding box on image
    bboxes = annotation['boxes']
    for box in bboxes:
        # Rectangular patch in COCO format
        patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(patch)
    ax.set_title(f'Category: {annotation['labels']}')
    plt.show()


dataset_path = os.path.join(os.path.dirname(__file__), 'strongylid_dataset')
images_path = os.path.join(dataset_path, 'images')
annotations_path = os.path.join(dataset_path, 'labels.json')
with open(annotations_path, 'r') as annotations_file:
    annotations = json.load(annotations_file)

    for image_name, annotation in annotations.items():
        try:
            show_image_with_bbox(
                os.path.join(images_path, image_name),
                annotation,
            )
        except:
            print('Verification of general strongylid egg complete')
            break
