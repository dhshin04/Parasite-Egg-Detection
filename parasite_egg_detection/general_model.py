''' Classifies 11 different fecal eggs '''

import os
import load_data
from train import train_model

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.multiprocessing as mp
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torch.cuda.amp import GradScaler
torch.manual_seed(1234)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
train_batch = 8                 # Train Loader Batch Size
cv_batch = 8                    # Validation Loader Batch Size
accumulation_size = 1           # For Gradient Accumulation
learning_rate = 1e-4            # For Training
epochs = 20                     # For Training
warmup_step = 10                # For LambdaLR Warmup
weight_decay = 0.0005           # For AdamW 
T_max = epochs                  # For CosineAnnealingLR
eta_min = learning_rate / 30    # For CosineAnnealingLR

iou_threshold = 0.5             # For Evaluation
confidence_threshold = 0.5      # For Evaluation


def warmup(epoch):
    '''
    For LambdaLR. Warms up learning rate for smoother convergence.

    Arguments:
        epoch (int): Current epoch 
    Returns:
        (float): Scaled learning rate
    '''
    if epoch < warmup_step:
        return float(epoch) / warmup_step
    return 1.
    

def main():
    ''' Data Loading '''
    train_loader, validation_loader, _ = load_data.get_data_loaders(
        cv_test_split=0.5,
        train_batch=train_batch,
        cv_batch=cv_batch,
        device=DEVICE,
        data_type='general',
    )


    ''' Model Fine-Tuning '''
    # Load Pre-trained Faster R-CNN Model with Pretrained Weights
    model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

    # Fine-Tune Box Predictor (Classifier + Bounding Box Predictor)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
    num_classes = 12        # 11 classes + background
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    model.to(DEVICE)

    ''' Model Training and Evaluation '''
    # Compile Model
    optimizer = optim.AdamW(
        [parameters for parameters in model.parameters() if parameters.requires_grad],  # only alter params of unfreezed layers
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)                     # LR Warmup to Complement AdamW
    cos_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)   # LR Scheduler to Complement AdamW
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[warmup_step])
    
    if torch.cuda.is_available():
        scaler = GradScaler()   # Mixed Precision for faster training
    else:
        scaler = None

    # Train and Evaluate Model
    train_model(        # Stored in train.py
        model=model, 
        device=DEVICE, 
        train_loader=train_loader, 
        validation_loader=validation_loader,
        accumulation_size=accumulation_size,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epochs=epochs,
        iou_threshold=iou_threshold,                    
        confidence_threshold=confidence_threshold,      
    )

    # Save Model
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'general_model_weights.pth')
    torch.save({
        'model_type': 'fasterrcnn_mobilenet_v3_large_fpn',
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
    }, checkpoint_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)        # for compatibility with Windows multiprocessing
    main()
