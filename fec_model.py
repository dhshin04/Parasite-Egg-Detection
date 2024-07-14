''' Computes Fecal Egg Count of Strongylid Eggs '''

# TODO: Custom efficient evaluate (only need to compare bbox, not labels as well), 
#       dataset

import os
import torch
from torch import optim
import torch.multiprocessing as mp
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, faster_rcnn
torch.manual_seed(1234)

# Custom Modules
import load_data
from train import train_model

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Structure of Torchvision's Faster R-CNN ResNet50-FPN Model:

    Feature extraction - ResNet (Bottleneck/Residual Blocks)
        - model.backbone.body
            .parameters() for just param OR named_parameters() to access layer names
    Consistency in feature maps - FPN (for objects of varying sizes)
        - model.backbone.fpn
    Region proposals of where objects likely are based on features - RPN
        - model.rpn
    Classification & bounding box predictions - ROI Heads
        - model.roi_heads
'''

# Hyperparameters
train_batch = 3         # Train Loader Batch Size
cv_batch = 3            # Validation Loader Batch Size
learning_rate = 0.01            # For Training
epochs = 20                     # For Training
iou_threshold = 0.5             # For Evaluation
confidence_threshold = 0.5      # For Evaluation


# Used for making predictions
def export_hyperparameters():
    return iou_threshold, confidence_threshold


def main():
    ''' Data Loading '''
    print('Data Loading...')
    train_loader, validation_loader, _ = load_data.get_data_loaders(
        cv_test_split=0.5,
        train_batch=train_batch,
        cv_batch=cv_batch,
        device=DEVICE,
    )


    ''' Model Fine-Tuning '''
    print('Model Fine-Tuning...')
    # Load Pre-trained Faster R-CNN Model with Pretrained Parameters
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    general_model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'general_FEC_weights.pth')
    model.load_state_dict(torch.load(general_model_path))   # trained on general egg dataset

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Fine-Tune Box Predictor (Classifier + Object Detection)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
    num_classes = 2        # Strongylid Egg Present or Not
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    # Unfreeze just FC layer(s)
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    model.to(DEVICE)

    ''' Model Training and Evaluation '''
    print('Prepare Training...')

    # Compile Model
    optimizer = optim.Adam(
        [parameters for parameters in model.parameters() if parameters.requires_grad],  # only alter params of unfreezed layers
        lr=learning_rate,
    )

    # Train and Evaluate Model
    train_model(        # Stored in train.py
        model=model, 
        device=DEVICE, 
        train_loader=train_loader, 
        validation_loader=validation_loader,
        optimizer=optimizer,
        epochs=epochs,
        iou_threshold=iou_threshold,                    
        confidence_threshold=confidence_threshold,      
    )

    # Save Model
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'fec_model_weights.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)    # safe method for creating new subprocesses - used for data loaders
    main()
