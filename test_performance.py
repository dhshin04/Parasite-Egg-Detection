''' Make Inference on Test Set to Estimate Generalization '''

import os
import time
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import load_data
from evaluate import evaluate
from model import export_hyperparameters

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_performance(data_loader, iou_threshold, confidence_threshold):
    model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
    with torch.no_grad():
        metric = MeanAveragePrecision(iou_type='bbox')

        avg_precision = 0.
        avg_recall = 0.
        num_batch = 0

        start = time.time()
        for x_test, y_test in data_loader:
            '''
            List of dictionaries containing labels, scores, masks, and bbox for each object
            detected. N elements in list for N images passed as parameter. 
            '''
            x_test = [image.to(DEVICE) for image in x_test]
            y_test = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y_test]
            predictions = model(x_test)

            # Custom Precision and Recall
            precision, recall = evaluate(predictions, y_test, iou_threshold, confidence_threshold)
            avg_precision += precision
            avg_recall += recall
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
        end = time.time()

        print(f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Elapsed Time: {end - start:.2f}s\n')

        results = metric.compute()
        print('mAP Results:')
        print(f'mAP@0.5: {results["map_50"].item():.4f}')
        print(f'mAP@0.5-0.95: {results["map"].item():.4f}')


if __name__ == '__main__':
    # Load Test Data For Inference
    train_loader, validation_loader, test_loader = load_data.get_data_loaders(
        cv_test_split=0.5,
        cv_batch=8,
        test_batch=8,
        device=DEVICE,
    )

    # Load Hyperparameters
    iou_threshold, confidence_threshold = export_hyperparameters()
    # iou_threshold = 0.75
    # confidence_threshold = 0.75

    # Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    # Fine-Tune Box Predictor (Classifier + Object Detection)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
    num_classes = 12        # 11 classes + background
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    model_version = 'backup.pth'   # or 'fec_model_weights.pth'
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_version)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    model.to(DEVICE)

    print('Validation Performance')
    test_performance(validation_loader, iou_threshold, confidence_threshold)

    print('\nTest Performance')
    test_performance(test_loader, iou_threshold, confidence_threshold)
