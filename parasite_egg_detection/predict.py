''' Make FEC Prediction When Given Image '''

import os
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

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


def plot_image_with_bbox(image, annotation, parasite=None):
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
    bboxes = np.array(annotation['boxes'])      # Convert to np.array for matplotlib
    for box in bboxes:
        # Rectangular patch in COCO format
        patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              linewidth=3, edgecolor='b', facecolor='none')
        ax.add_patch(patch)
    
    fec = len(annotation['boxes'])
    if parasite == 'strongylid':
        ax.set_title(f'Fecal Egg Count: {fec}')
    else:
        ax.set_title(f'Category: {annotation["labels"]}')
    plt.show()

    return fec


def make_predictions(images, parasite=None):
    '''
    Make object detection inferences when given images tensor

    Arguments:
        images (str): Images tensor
        parasite (str): 'general'/None for general model (default) 
                        or 'strongylid' for strongylid model
    Returns:
        fec (int): Fecal Egg Count for Image (or average if multiple images provided, 
                   rounded to nearest integer)
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
        predictions = model(images)
        avg_fec = 0.
        num_img = 0
        for img, prediction in zip(images, predictions):
            # Post-Process Output
            prediction = filter(prediction, confidence_threshold)
            prediction = non_maximum_suppresion(prediction, threshold=nms_threshold)
            
            # TODO: create new image instead of plotting if deploying model using Flask
            fec = plot_image_with_bbox(img, prediction, parasite)

            avg_fec += fec
            num_img += 1
        avg_fec /= num_img
    return round(avg_fec)


def predict(image_paths, parasite='general'):
    '''
    Make object detection inferences, given list of image paths

    Arguments:
        image_paths: List of image paths
        parasite (str): 'general'/None for general model (default) 
                        or 'strongylid' for strongylid model
    Returns:
        (tuple): Fecal Egg Count and Eggs per Gram for image 
                 (or average fec for multiple images)
    '''

    to_tensor = transforms.ToTensor()
    images = []
    for image_path in image_paths:
        # Preprocess Image - same process as FecalEggDataset to ensure consistency
        image = Image.open(image_path).convert('RGB')   # handle grayscale or RGBA image inputted
        image = to_tensor(image)
        if image.dtype == torch.uint8:      # Normalize image for appropriate model input
            image = image.float() / 255
        image = image.to(DEVICE)
        images.append(image)

    fec = make_predictions(images, parasite)
    epg = fec * 50

    return fec, epg
