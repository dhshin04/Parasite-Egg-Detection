''' Make FEC Prediction When Given Image '''

import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import load_data
from evaluate import evaluate
from fec_model import export_hyperparameters

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict(image):
    # TODO: Preprocess image

    # Load Hyperparameters
    iou_threshold, confidence_threshold = export_hyperparameters()

    # Load Pre-trained Mask R-CNN Model with Custom-Trained Parameters
    model = fasterrcnn_resnet50_fpn_v2(weights=None)

    model_version = 'fec_model_weights.pth'
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
        prediction = model(image)
        bbox = prediction['boxes']
        fec = len(bbox)
            
        print(f'\nFecal Egg Count in Image: {fec}')
