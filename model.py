''' Classifies 11 different fecal eggs '''

import os
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.multiprocessing as mp
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
from torch.cuda.amp import GradScaler
torch.manual_seed(1234)

# Custom Modules
import load_data
from train import train_model

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Structure of Torchvision's Faster R-CNN MobileNet-Large-FPN Model:

    Feature extraction - MobileNet (for faster computation)
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
train_batch = 8                 # Train Loader Batch Size
cv_batch = 8                    # Validation Loader Batch Size
accumulation_size = 1           # For Gradient Accumulation
learning_rate = 1e-4            # For Training - best: 5e-5; potential: 1e-4, 5e-4, 1e-6
epochs = 20                     # For Training
warmup_step = 10                # For LR Warmup
scale_train_set = 1.0

iou_threshold = 0.5             # For Evaluation
confidence_threshold = 0.5      # For Evaluation


# Used for making predictions
def export_hyperparameters():
    return iou_threshold, confidence_threshold


def warmup(epoch):       # Learning Rate Warmup (Lambda)
    # Warm-Up over step_size epochs
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
        scale_train_set=scale_train_set,
    )


    ''' Model Fine-Tuning '''
    # Load Pre-trained Faster R-CNN Model with Pretrained Parameters
    model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

    '''
    # Freeze parameters of early layers, while unfreezing later layers for fine-tuning
    for param in model.backbone.body.parameters():
        param.requires_grad = False         # Freeze backbone layers

    for param in model.backbone.fpn.parameters():
        param.requires_grad = False         # Freeze fpn layers (objects are mostly similar in size)

    for param in model.rpn.parameters():
        param.requires_grad = False         # Freeze rpn layers
    '''

    # Fine-Tune Box Predictor (Classifier + Object Detection)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features    # Original input to box predictor fc layers
    num_classes = 12        # 11 classes + background
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    '''
    for param in model.roi_heads.parameters():
        param.requires_grad = True          # Unfreeze fc layers
    '''

    model.to(DEVICE)

    ''' Model Training and Evaluation '''
    # Compile Model
    optimizer = optim.AdamW(
        [parameters for parameters in model.parameters() if parameters.requires_grad],  # only alter params of unfreezed layers
        lr=learning_rate,
        # momentum=0.9,
        weight_decay=0.0005,
    )
    cos_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=(learning_rate/30))   # LR Warmup to Complement Adam
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[5])
    scaler = GradScaler()

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
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'general_FEC_weights.pth')
    torch.save({
        'model_type': 'fasterrcnn_mobilenet_v3_large_fpn',
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
    }, checkpoint_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)        # for multiprocessing
    main()
