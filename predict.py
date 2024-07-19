''' Make FEC Prediction When Given Image '''
# TODO: Make sure to filter out low confidence scores during inference

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
    bboxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    filtered_prediction = {
        'boxes': [],
        'labels': [],
        'scores': [],
    }
    for bbox, label, score in zip(bboxes, labels, scores):
        if score.item() > confidence_threshold:
            filtered_prediction['boxes'].append(bbox)
            filtered_prediction['labels'].append(label)
            filtered_prediction['scores'].append(score)
    return filtered_prediction


def non_maximum_suppresion(prediction, iou_threshold=0.5):
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
    image = image.cpu()
    fig, ax = plt.subplots(1)
    image = image.permute(1, 2, 0)
    ax.imshow(image)

    bboxes = np.array([bbox.cpu().numpy() for bbox in annotation['boxes']])

    for box in bboxes:
        # Rectangular patch in COCO format
        patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(patch)
    ax.set_title(f'Category: {[str(label.item()) for label in annotation["labels"]]}')
    plt.show()


def predict(image_path):
    # Preprocess Image
    to_tensor = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    image = to_tensor(image)
    image = normalize(image)
    image = image.unsqueeze(0)     # model requires list of images

    # Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 12
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    model_version = 'backup.pth'   # or 'fec_model_weights.pth'
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_version)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    model.to(DEVICE)

    # Run Inference for Test Loader
    model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
    with torch.no_grad():
        '''
        List of dictionaries containing labels, scores, masks, and bbox for each object
        detected. N elements in list for N images passed as parameter. 
        '''
        image = image.to(DEVICE)
        predictions = model(image)
        for img, prediction in zip(image, predictions):
            prediction = filter(prediction, confidence_threshold)
            prediction = non_maximum_suppresion(prediction, iou_threshold=0.7)
            fec = len(prediction['boxes'])

            plot_image_with_bbox(img, prediction)
    return fec


if __name__ == '__main__':
    trainingset_path = os.path.join(os.path.dirname(__file__), 'data', 'trainingset')
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

    image_path = os.path.join(os.path.dirname(__file__), 'barber_pole_egg_data', 'images', '0003.jpg')
    fec = predict(image_path)
    print(f'\nFEC: {fec}')
