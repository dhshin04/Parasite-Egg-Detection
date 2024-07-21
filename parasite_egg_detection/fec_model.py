''' Computes Fecal Egg Count of Strongylid Eggs '''

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
train_batch = 2                 # Train Loader Batch Size
cv_batch = 8                    # Validation Loader Batch Size
accumulation_size = 1           # For Gradient Accumulation
learning_rate = 1e-4            # For Training - best: 1e-4
epochs = 50                     # For Training
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
        data_type='strongylid',
    )


    ''' Model Fine-Tuning '''
    # Load Pre-trained Faster R-CNN Model with Custom-Trained Parameters
    model_version = 'general_model_weights.pth'
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_version)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Fine-Tune Model
    #for param in model.backbone.parameters():    # Freeze MobileNet & FPN networks
    #    param.requires_grad = False
    
    for name, param in model.backbone.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
            break
        else:
            param.requires_grad = False
    print(model)
    
    for param in model.rpn.parameters():         # Freeze RPN network
        param.requires_grad = True
        
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2     # Binary Classification: Strongylid eggs present or not
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    # for param in model.roi_heads.parameters():   # Ensures ROI Head is not frozen 
    #     param.requires_grad = True

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
    
    scaler = GradScaler()   # Mixed Precision for faster training

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
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'fec_model_weights.pth')
    torch.save({
        'model_type': 'fasterrcnn_mobilenet_v3_large_fpn',
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
    }, checkpoint_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)        # for compatibility with Windows multiprocessing
    main()
