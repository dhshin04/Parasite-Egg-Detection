import copy
import torch


def iou(box1, box2):
    '''
    Arguments:
        box1 (torch.Tensor): [4] tensor that stores (x1, y1, x2, y2)
        box2 (torch.Tensor): [4] tensor that stores (x3, y3, x4, y4)
    
        Returns:
            (float): Intersection over Union value for 2 boxes
    '''

    # Area of Intersection is given as:
    x1 = box1[0].item(); y1 = box1[1].item(); x2 = box1[2].item(); y2 = box1[3].item()
    x3 = box2[0].item(); y3 = box2[1].item(); x4 = box2[2].item(); y4 = box2[3].item()

    distance_x = min(x2, x4) - max(x1, x3)
    distance_y = min(y2, y4) - max(y1, y3)
    if distance_x <= 0 or distance_y <= 0:      # Indicates no overlaps
        intersection = 0
    else:
        intersection = distance_x * distance_y

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection

    if union <= 0:
        raise Exception('Logical Error: union cannot be 0 by definition as long as boxes have area')

    return intersection / union     # IoU (intersection over union)


def match_pred_to_target(prediction, target, iou_threshold=0.5):
    '''
    Arguments:
        prediction (dict): Prediction dictionary for single image
        target (dict): Target dictionary for single image

    Returns:
        (int[]): Matching prediction box (index) and target box (value at index).
                 Value of -1 represents no match available.
    '''
    prediction_boxes = prediction['boxes']
    target_boxes = copy.deepcopy(target['boxes'])

    potential_true_positives = 0
    match_list = []
    for p_box in prediction_boxes:
        max_iou = 0         # Minimum IoU value
        max_index = -1      # When all available target boxes have IoU of 0
        for i, t_box in enumerate(target_boxes, 0):
            computed_iou = iou(p_box, t_box)

            # If the computed IoU is above threshold and max IoU, it is considered
            # potential true positive. 
            if computed_iou >= iou_threshold and computed_iou > max_iou:
                max_iou = computed_iou
                max_index = i
        # Max IoU target box now belongs to this p_box alone -> setting it to 0 means future IoU is also 0
        if max_iou > 0:     # Some matching target box found
            target_boxes[max_index] = torch.FloatTensor([0, 0, 0, 0])
            potential_true_positives += 1   # Still potential positive since label needs to be compared
        match_list.append(max_index)
    
    total_pred = len(prediction_boxes)  # Total Prediction Labels (Boxes generated)
    total_gt = len(target_boxes)        # Total Ground-Truth Labels (Boxes)

    return match_list, total_pred, total_gt


def compare_labels(prediction, target, match_list, confidence_threshold=0.5):
    p_label = prediction['labels']
    t_label = target['labels']
    true_positives = 0
    for i, match in enumerate(match_list, 0):   # Match list has length of predictions
        # No bbox match or too low confidence
        if match < 0 or prediction['scores'][i] < confidence_threshold:
            continue
        if p_label[i] == t_label[match]:    # Matching bbox and labels
            true_positives += 1
    return true_positives


def evaluate(predictions, targets, iou_threshold=0.5, confidence_threshold=0.5):
    if len(predictions) != len(targets):
        raise ValueError('Logical Error: predictions tensor and targets tensor are different lengths')
    
    num_images = len(predictions)

    # Average precision and recall in given batch of images
    avg_precision = 0.
    avg_recall = 0.

    for prediction, target in zip(predictions, targets):
        match_list, total_pred, total_gt = match_pred_to_target(prediction, target, iou_threshold)
        
        # Matching bbox and labels
        true_positives = compare_labels(prediction, target, match_list, confidence_threshold)
        
        false_positives = total_pred - true_positives
        false_negatives = total_gt - true_positives

        # Metrics
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        avg_precision += precision
        avg_recall += recall
    
    avg_precision /= num_images
    avg_recall /= num_images

    return avg_precision, avg_recall
