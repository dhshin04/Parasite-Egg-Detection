''' Make FEC Prediction When Given Image '''

import os
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters for Inference (best case for both general and strongylid models)
confidence_threshold = 0.3
nms_threshold = 0.3


def iou(box1, box2):
    '''
    Copied from evaluate.py instead of importing due to relative import conflicts
    if predict.py called from file outside parasite_egg_detection folder.

    Arguments:
        box1 (torch.Tensor): [4] tensor that stores (x1, y1, x2, y2)
        box2 (torch.Tensor): [4] tensor that stores (x3, y3, x4, y4)
    Returns:
        (float): Intersection over Union value for 2 boxes
    '''

    # Area of Intersection is given as:
    x1 = box1[0].item(); y1 = box1[1].item(); x2 = box1[2].item(); y2 = box1[3].item()
    x3 = box2[0].item(); y3 = box2[1].item(); x4 = box2[2].item(); y4 = box2[3].item()

    distance_x = min(x2, x4) - max(x1, x3) + 1  # +1 includes pixel that xn covers
    distance_y = min(y2, y4) - max(y1, y3) + 1  # +1 includes pixel that yn covers
    if distance_x <= 0 or distance_y <= 0:      # Indicates no overlaps
        intersection = 0
    else:
        intersection = distance_x * distance_y

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union = box1_area + box2_area - intersection

    if union <= 0:
        raise Exception('Logical Error: union cannot be 0 by definition as long as boxes have area')

    return intersection / union     # IoU (intersection over union)


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


def non_maximum_suppresion(prediction, threshold=0.5):
    '''
    Removes redundant bounding boxes for same object in image

    Arguments:
        prediction (dict): Dictionary of bounding boxes, labels, and confidence scores
        threshold (float): Bounding boxes with IoU higher than this are considered redundant
    Returns:
        new_pred (dict): Dictionary without redundant bounding boxes
    '''
    bboxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    new_pred = {
        'boxes': [],
        'labels': [],
        'scores': [],
    }

    if len(bboxes) == 0:    # no prediction to apply NMS
        return new_pred

    # Bbox discarded if better bbox found for object
    discard = set()     # HashSet for O(1) search + insert & no duplicates
    for i in range(len(bboxes)):
        if i in discard:
            continue
        keep = True
        for j in range(i+1, len(bboxes)):
            if iou(bboxes[i], bboxes[j]) >= threshold:  # 2 bbox point to same object
                if scores[i] < scores[j]:   # bbox i is discarded
                    keep = False
                    break
                else:       # bbox j is discarded
                    discard.add(j)
        if keep:
            new_pred['boxes'].append(bboxes[i].tolist())
            new_pred['labels'].append(labels[i].tolist())
            new_pred['scores'].append(scores[i].tolist())
    
    return new_pred


def add_labels(cv2_image, annotation, parasite):
    '''
    Plot image with corresponding bounding boxes

    Arguments:
        image (tensor): Image tensor to plot
        annotation (dict): Dictionary containing annotations for image
        parasite (str): Type of parasite detection model
    Returns:
        cv2_image (np.ndarray): Labeled image in cv2 image format (np.ndarray)
    '''

    CATEGORIES = {
        1: 'Ascaris lumbricoides',
        2: 'Capillaria philippinensis',
        3: 'Enterobius vermicularis',
        4: 'Fasciolopsis buski',
        5: 'Hookworm egg',
        6: 'Hymenolepis diminuta',
        7: 'Hymenolepis nana',
        8: 'Opisthorchis viverrine',
        9: 'Paragonimus spp',
        10: 'Taenia spp. egg',
        11: 'Trichuris trichiura',
    }

    HEIGHT, WIDTH, _ = cv2_image.shape
    THICKNESS_RATIO = 4.5e-3
    MARGIN_RATIO = 1.5e-2

    bboxes = annotation['boxes']
    labels = annotation['labels']
    scores = annotation['scores']

    for box, label, score in zip(bboxes, labels, scores):
        # Add bounding box
        cv2.rectangle(
            cv2_image, 
            (round(box[0]), round(box[1])), 
            (round(box[2]), round(box[3])), 
            color=(255, 0, 0), 
            thickness=max(1, round((HEIGHT + WIDTH) / 2 * THICKNESS_RATIO)),
        )

        # Add label
        margin = round((HEIGHT + WIDTH) / 2 * MARGIN_RATIO)
        if box[1] < margin:
            label_position = (round(box[0]), round(box[3]) + margin)
        else:
            label_position = (round(box[0]), round(box[1]) - margin)

        if parasite == 'strongylid':
            label_text = str(round(score * 100, 1)) + '%'
            fontscale_ratio = 1e-3
        else:
            label_text = f'{CATEGORIES[label]}, {round(score * 100, 1)}%'
            fontscale_ratio = 8e-4

        cv2.putText(
            cv2_image,
            label_text,
            label_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=(HEIGHT + WIDTH) / 2 * fontscale_ratio,
            color=(255, 20, 0),
            thickness=max(1, round((HEIGHT + WIDTH) / 2 * THICKNESS_RATIO)),
        )
    
    return cv2_image


def make_predictions(cv2_images, tensor_images, parasite=None):
    '''
    Make object detection inferences when given images tensor

    Arguments:
        cv2_images (np.ndarray): Images in cv2 format
        tensor_images (torch.tensor): Images in tensor format
        parasite (str): 'general'/None for general model (default) 
                        or 'strongylid' for strongylid model
    Returns:
        (tuple): List of labeled PIL images, Fecal Egg Count for Image 
                 (or average if multiple images provided, rounded to nearest integer)
    '''

    # Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
    if parasite == 'strongylid':
        model_version = 'strongylid_model_weights.pth'
        
    else:
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
        predictions = model(tensor_images)
        avg_fec = 0.
        labeled_images = []

        for cv2_img, prediction in zip(cv2_images, predictions):
            # Post-Process Output
            prediction = filter(prediction, confidence_threshold)
            prediction = non_maximum_suppresion(prediction, threshold=nms_threshold)

            fec = len(prediction['boxes'])      # FEC = # objects found (num bboxes)
            
            # Add labels to image
            labeled_img = add_labels(cv2_img, prediction, parasite)
            labeled_images.append(labeled_img)

            avg_fec += fec
        avg_fec /= len(cv2_images)
    return labeled_images, round(avg_fec)


def predict(cv2_images, parasite='general'):
    '''
    Make object detection inferences, given list of image paths

    Arguments:
        pil_images (np.ndarray): List of cv2 image inputs (as np.ndarray)
        parasite (str): 'general'/None for general model (default) 
                        or 'strongylid' for strongylid model
    Returns:
        (tuple): Fecal Egg Count and Eggs per Gram for image 
                 (or average fec for multiple images)
    '''

    # Preprocess Image
    to_tensor = transforms.ToTensor()

    tensor_images = []  # Images as tensors for making model predictions

    for cv2_image in cv2_images:
        pil_image = Image.fromarray(        # PIL image needed for transforms.ToTensor()
            cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        )

        tensor_image = to_tensor(pil_image)
        if tensor_image.dtype == torch.uint8:      # Normalize image for appropriate model input
            tensor_image = tensor_image.float() / 255
        tensor_image = tensor_image.to(DEVICE)
        tensor_images.append(tensor_image)

    assert len(cv2_images) == len(tensor_images), 'cv2 images and tensor images have different length'

    # Make Prediction
    labeled_images, fec = make_predictions(cv2_images, tensor_images, parasite)
    epg = fec * 50

    return labeled_images, fec, epg
