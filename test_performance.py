''' Make Inference on Test Set to Estimate Generalization '''

import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import load_data
from evaluate import evaluate
from general_model import export_hyperparameters

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load Test Data For Inference
test_loader = load_data.get_data_loaders(
    test_batch=10,
    device=DEVICE,
)[2]

# Load Hyperparameters
iou_threshold, confidence_threshold = export_hyperparameters()

# Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
model = fasterrcnn_resnet50_fpn_v2(weights=None)

model_version = 'general_FEC_weights.pth'   # or 'fec_model_weights.pth'
checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_version)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

model.to(DEVICE)

# Run Inference for Test Loader
model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
with torch.no_grad():
    avg_precision = 0.
    avg_recall = 0.
    avg_accuracy = 0.
    num_batch = 0
    for images, targets in test_loader:
        '''
        List of dictionaries containing labels, scores, masks, and bbox for each object
        detected. N elements in list for N images passed as parameter. 
        '''
        predictions = model(images)
        precision, recall, accuracy = evaluate(predictions, targets, iou_threshold, confidence_threshold)
        
        avg_precision += precision
        avg_recall += recall
        avg_accuracy += accuracy
        num_batch += 1
    avg_precision /= num_batch
    avg_recall /= num_batch
    avg_accuracy /= num_batch

    print(f'\nPrecision: {avg_precision}, Recall: {avg_recall}, Accuracy: {avg_accuracy}')
