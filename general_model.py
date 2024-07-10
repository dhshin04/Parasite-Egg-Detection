''' Classifies 11 different fecal eggs '''

import os
import json
import multiprocessing
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, faster_rcnn, mask_rcnn
from sklearn.model_selection import train_test_split
import load_data
from train import train_model
torch.manual_seed(1234)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Structure of Torchvision's Mask R-CNN ResNet50-FPN Model:

    Feature extraction - ResNet (Bottleneck/Residual Blocks)
        - model.backbone.body
            .paramters() for just param OR named_parameters() to access layer names
    Consistency in feature maps - FPN (for objects of varying sizes)
        - model.backbone.fpn
    Region proposals of where objects likely are based on features - RPN
        - model.rpn
    Classification & bounding box/mask predictions - ROI Heads
        - model.roi_heads
'''

''' Data Loading '''
# Hyperparameters (1/2)
train_batch = 15        # Train Loader Batch Size
cv_batch = 10           # Validation Loader Batch Size

# Load Train and Validation Data for Training and Evaluation
train_loader, validation_loader, _ = load_data.get_data_loaders(
    cv_test_split=0.5,
    train_batch=train_batch,
    cv_batch=cv_batch,
    device=DEVICE,
)


''' Model Fine-Tuning '''
# Load Pre-trained Mask R-CNN Model with Pretrained Parameters
model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT').to(DEVICE)

# Freeze parameters of early layers, while unfreezing later layers for fine-tuning
for name, param in model.backbone.body.named_parameters():
    if 'layer4' in name:                # Unfreeze layer 4
        param.requires_grad = True
    else:
        param.requires_grad = False     # Freeze layers 1~3 in ResNet layers

for param in model.backbone.fpn.parameters():
    param.requires_grad = False         # Freeze fpn layers (objects are mostly similar in size)

for param in model.rpn.parameters():
    param.requires_grad = True          # Unfreeze rpn layers

for param in model.roi_heads.parameters():
    param.requires_grad = True          # Unfreeze fc layers

# Fine-Tune Box and Mask Predictors (Classifier + Object Detection)
in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
num_classes = 12        # 11 classes + background
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layers = 256
model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_channels_mask, hidden_layers, num_classes)


''' Model Training and Evaluation '''
# Hyperparameters (2/2)
learning_rate = 0.01            # For Training
epochs = 20                     # For Training
iou_threshold = 0.5             # For Evaluation
confidence_threshold = 0.5      # For Evaluation


def export_hyperparameters():
    return iou_threshold, confidence_threshold


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
# torch.save(model.state_dict(), checkpoint_path)
