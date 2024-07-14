''' Classifies 11 different fecal eggs '''

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
train_batch = 8        # Train Loader Batch Size
cv_batch = 8            # Validation Loader Batch Size
learning_rate = 0.1             # For Training
epochs = 20                     # For Training
iou_threshold = 0.5             # For Evaluation
confidence_threshold = 0.1      # For Evaluation


# Used for making predictions
def export_hyperparameters():
    return iou_threshold, confidence_threshold


def main():
    ''' Data Loading '''
    train_loader, validation_loader, _ = load_data.get_data_loaders(
        cv_test_split=0.5,
        train_batch=train_batch,
        cv_batch=cv_batch,
        device=DEVICE,
        scale_train_set=0.1,       # 0.1 for simple hyperparam tuning, 0.2 for complex, 1.0 for final training
    )


    ''' Model Fine-Tuning '''
    # Load Pre-trained Faster R-CNN Model with Pretrained Parameters
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # Freeze parameters of early layers, while unfreezing later layers for fine-tuning
    for name, param in model.backbone.body.named_parameters():
        if 'layer4' in name:                # Unfreeze layer 4
            param.requires_grad = False
        else:
            param.requires_grad = False     # Freeze layers 1~3 in ResNet layers

    for param in model.backbone.fpn.parameters():
        param.requires_grad = False         # Freeze fpn layers (objects are mostly similar in size)

    for param in model.rpn.parameters():
        param.requires_grad = False         # Unfreeze rpn layers

    # Fine-Tune Box Predictor (Classifier + Object Detection)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
    num_classes = 12        # 11 classes + background
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    for param in model.roi_heads.parameters():
        param.requires_grad = True          # Unfreeze fc layers

    model.to(DEVICE)

    ''' Model Training and Evaluation '''
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
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'general_FEC_weights.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
