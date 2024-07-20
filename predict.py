''' Make FEC Prediction When Given Image '''

import os
import json
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torchvision.io import read_image
from torchvision import tv_tensors, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random

from data.fecal_egg_dataset import normalize
from evaluate import iou

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Hyperparameter for Inference
confidence_threshold = 0.5


def filter(prediction, confidence_threshold=0.5):
    '''
    Filter out low confidence bounding boxes

    Arguments:
        prediction (dict): Dictionary containing bounding boxes, labels, and scores for image
        confidence_threshold (float): Bounding boxes with confidence below this are discarded
    Returns:
        filtered_prediction (dict): Dictionary without low confidence bounding boxes
    '''
    bboxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    filtered_prediction = {
        'boxes': [],
        'labels': [],
        'scores': [],
    }
    for bbox, label, score in zip(bboxes, labels, scores):
        if score.item() > confidence_threshold:     # only include high confidence bounding boxes
            filtered_prediction['boxes'].append(bbox)
            filtered_prediction['labels'].append(label)
            filtered_prediction['scores'].append(score)
    return filtered_prediction


def non_maximum_suppresion(prediction, iou_threshold=0.5):
    '''
    Removes redundant bounding boxes for same object in image

    Arguments:
        prediction (dict): Dictionary of bounding boxes, labels, and confidence scores
        iou_threshold (float): Bounding boxes with IoU higher than this are considered redundant
    Returns:
        new_pred (dict): Dictionary without redundant bounding boxes
    '''
    bboxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    if len(bboxes) <= 0:    # no prediction to apply NMS
        return prediction

    new_pred = {
        'boxes': [],
        'labels': [],
        'scores': [],
    }

    # Bbox discarded if better bbox found for object
    discard = set()     # HashSet for O(1) search + insert & no duplicates
    for i in range(len(bboxes)):
        if i in discard:
            continue
        keep = True
        for j in range(i+1, len(bboxes)):
            if iou(bboxes[i], bboxes[j]) >= iou_threshold:  # 2 bbox point to same object
                if scores[i] < scores[j]:   # bbox i is discarded
                    keep = False
                    break
                else:       # bbox j is discarded
                    discard.add(j)
        if keep:
            new_pred['boxes'].append(bboxes[i])
            new_pred['labels'].append(labels[i])
            new_pred['scores'].append(scores[i])
    return new_pred


def plot_image_with_bbox(image, annotation):
    '''
    Plot image with corresponding bounding boxes

    Arguments:
        image (tensor): Image tensor to plot
        annotation (dict): Dictionary containing annotations for image
    '''

    # Plot Image
    image = image.cpu()
    fig, ax = plt.subplots(1)
    image = image.permute(1, 2, 0)
    ax.imshow(image)

    # Plot bounding box on image
    bboxes = np.array([bbox.cpu().numpy() for bbox in annotation['boxes']])
    for box in bboxes:
        # Rectangular patch in COCO format
        patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(patch)
    ax.set_title(f'Category: {[str(label.item()) for label in annotation["labels"]]}')
    plt.show()


def predict(image_path):
    '''
    Make object detection inference when given image

    Arguments:
        image_path (str): Path to image
    Returns:
        fec (int): Fecal Egg Count for Image
    '''

    # Preprocess Image
    to_tensor = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')   # handle grayscale or RGBA image inputted
    image = to_tensor(image)
    image = normalize(image)
    image = image.unsqueeze(0)     # model requires list of images

    # Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
    model_version = 'general_model_weights.pth'
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_version)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = checkpoint['num_classes']
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    # Make Inference
    model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
    with torch.no_grad():
        image = image.to(DEVICE)
        predictions = model(image)
        for img, prediction in zip(image, predictions):
            # Post-Process Output
            prediction = filter(prediction, confidence_threshold)
            prediction = non_maximum_suppresion(prediction, iou_threshold=0.7)
            
            fec = len(prediction['boxes'])
            plot_image_with_bbox(img, prediction)
    return fec


if __name__ == '__main__':
    # Example Inference Script for Demonstration
    
    trainingset_path = os.path.join(os.path.dirname(__file__), 'data', 'general_test')
    training_images_path = os.path.join(trainingset_path, 'images')
    training_labels_path = os.path.join(trainingset_path, 'refined_labels.json')
    with open(training_labels_path, 'r') as refined_labels:
        train_annotations = json.load(refined_labels)

    train_images = sorted(os.listdir(training_images_path))

    image_idx = random.randint(0, len(train_images) - 1)
    image_name = train_images[image_idx]
    image_path = os.path.join(training_images_path, image_name)

    annotation = train_annotations[image_name]
    print(f'Correct Category: {annotation["labels"]}')
    print(f'Correct FEC: {len(annotation["boxes"])}')
    print(f'Image ID: {annotation["image_id"]}')

    fec = predict(image_path)
    print(f'FEC Prediction: {fec}')

    image_path = os.path.join(os.path.dirname(__file__), 'data', 'strongylid_data', 'images', 'medium_test.jpeg')
    fec = predict(image_path)
    print(f'\nFEC: {fec}')
