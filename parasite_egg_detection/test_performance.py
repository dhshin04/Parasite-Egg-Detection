''' Make Inference on Test Set to Estimate Generalization '''

import os
import time
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import load_data
from evaluate import evaluate
from predict import non_maximum_suppresion

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters for Inference
iou_threshold = 0.5
confidence_threshold = 0.5      # Best: 0.3 for general, 0.5 for strongylid
nms_threshold = 0.25


def pred_to_tensor(prediction):
    # Convert a non-tensor dict prediction to tensor dict

    prediction['boxes'] = torch.tensor(prediction['boxes'], dtype=torch.float32).to(DEVICE)
    prediction['labels'] = torch.tensor(prediction['labels'], dtype=torch.int64).to(DEVICE)
    prediction['scores'] = torch.tensor(prediction['scores'], dtype=torch.float32).to(DEVICE)

    return prediction


def without_low_confidence(prediction):
    # Fecal Egg Count without Low Confidence Predictions

    total_fec = len(prediction['boxes'])
    scores = prediction['scores']

    for score in scores:
        # Handle both low confidence
        if score < confidence_threshold:
            total_fec -= 1
    
    return total_fec


def test_fec_accuracy(predictions, targets):
    # FEC Accuracy using Percent Error - return average accuracy across prediction batch
    
    if len(predictions) == 0:       # Handle empty predictions
        print('Given predictions dictionary is empty')
        return -1

    avg_accuracy = 0.       # Average accuracy in prediction batch

    for prediction, target in zip(predictions, targets):
        actual_fec = without_low_confidence(prediction)
        expected_fec = len(target['boxes'])

        if expected_fec == 0:       # No object in image
            if actual_fec == 0:
                accuracy = 1.
            else:
                accuracy = 0.
        else:       # There is object in image
            # Accuracy = 1 - percent error
            accuracy = 1 - abs(actual_fec - expected_fec) / expected_fec
            if accuracy < 0:    # Handle potential negative cases
                accuracy = 0.
        avg_accuracy += accuracy

    avg_accuracy /= len(predictions)
    return avg_accuracy


def test_performance(model, data_loader, iou_threshold, confidence_threshold, nms_threshold):
    '''
    Evaluate model's precision, recall, and mean average precision metrics

    Arguments:
        data_loader (torch.utils.data.DataLoader): train loader, validation loader, or test loader
        iou_threshold (float): To retrieve precision and recall
        confidence_threshold (float): To retrieve precision and recall
    '''

    model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
    with torch.no_grad():
        metric = MeanAveragePrecision(iou_type='bbox')

        avg_precision = 0.
        avg_recall = 0.
        avg_accuracy = 0.
        num_batch = 0

        start = time.time()
        for x_test, y_test in data_loader:

            x_test = [image.to(DEVICE) for image in x_test]
            y_test = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y_test]
            predictions = model(x_test)
            for i in range(len(predictions)):
                predictions[i] = non_maximum_suppresion(predictions[i], threshold=nms_threshold)
                predictions[i] = pred_to_tensor(predictions[i])

            # Handle potential faulty predictions that have label of 0 (0 is only for background)
            for prediction in predictions:
                for label in prediction['labels']:
                    if label.item() == 0:
                        raise ValueError('Label cannot be 0')

            # Custom Precision and Recall
            precision, recall = evaluate(predictions, y_test, iou_threshold, confidence_threshold)
            avg_precision += precision
            avg_recall += recall
            avg_accuracy += test_fec_accuracy(predictions, y_test)
            num_batch += 1

            # mAP
            preds = [
                {
                    'boxes': prediction['boxes'], 
                    'labels': prediction['labels'], 
                    'scores': prediction['scores'],
                } 
                for prediction in predictions
            ]
            targets = [
                {
                    'boxes': target['boxes'], 
                    'labels': target['labels'], 
                } 
                for target in y_test
            ]
            metric.update(preds, targets)

        avg_precision /= num_batch
        avg_recall /= num_batch
        avg_accuracy /= num_batch
        end = time.time()

        print(f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Elapsed Time: {end - start:.2f}s')
        print(f'Fecal Egg Count Accuracy: {avg_accuracy * 100:.2f}%\n')

        results = metric.compute()
        print('mAP Results:')
        print(f'mAP@0.5: {results["map_50"].item():.4f}')
        print(f'mAP@0.5-0.95: {results["map"].item():.4f}')


def main(parasite=None):
    '''
    Test model performance (validation and test)

    Arguments:
        parasite (str): 'general'/None for general model (default) 
                        or 'strongylid' for strongylid model
    '''

    # Load Test Data For Inference
    _, validation_loader, test_loader = load_data.get_data_loaders(
        cv_test_split=0.5,
        cv_batch=8,
        test_batch=8,
        device=DEVICE,
        data_type=parasite,
    )

    # Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    if parasite == 'strongylid':
        model_version = 'strongylid_model_weights.pth'
        confidence_threshold = 0.5
    else:
        model_version = 'general_model_weights.pth'
        confidence_threshold = 0.3

    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_version)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Fine-Tune Box Predictor (Classifier + Object Detection)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
    num_classes = checkpoint['num_classes']        # 11 classes + background
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)

    print('Validation Performance')
    test_performance(model, validation_loader, iou_threshold, confidence_threshold, nms_threshold)

    print('\nTest Performance')
    test_performance(model, test_loader, iou_threshold, confidence_threshold, nms_threshold)


if __name__ == '__main__':
    main(parasite='strongylid')
