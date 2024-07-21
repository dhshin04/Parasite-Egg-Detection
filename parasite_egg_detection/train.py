import torch
import numpy as np
import time
from evaluate import evaluate

# Enable TF32 mode for matrix multiplications - ensure speed + accuracy for Mixed Precision
torch.backends.cuda.matmul.allow_tf32 = True


def train_model(model, device, train_loader, validation_loader, accumulation_size, optimizer, scheduler, scaler, epochs, iou_threshold=0.5, confidence_threshold=0.5):
    # Train and Evaluate Model

    print('Training...')
    for epoch in range(epochs):
        # Train
        model.train()               # Train Mode: requires_grad=True, Batch Norm on
        train_loss_list = []

        start = time.time()
        for batch_index, (x, y) in enumerate(train_loader):   # Mini-Batch Gradient Descent
            # Move tensors to GPU
            model.train()
            x = [image.to(device) for image in x]
            y = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y]
            
            if scaler is not None:
                with torch.autocast(device_type='cuda', dtype=torch.float16):   # Mixed Precision
                    losses = model(x, y)        # Returns dictionary of losses
                    loss = sum([loss for loss in losses.values()]) / accumulation_size      # Total loss = sum of all losses; normalize
            else:
                losses = model(x, y)
                loss = sum([loss for loss in losses.values()]) / accumulation_size
            
            train_loss_list.append(loss.data.item())
            
            if scaler is not None:
                scaler.scale(loss).backward()         # Compute gradient of loss
            else:
                loss.backward()

            if (batch_index + 1) % accumulation_size == 0 or (batch_index + 1) == len(train_loader):    # Gradient Accumulation (if applicable)
                if scaler is not None:
                    scaler.step(optimizer)     # Update parameters
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()      # clear old gradient before new gradient calculation
            
        # Validation Loss
        with torch.no_grad():
            cv_loss_list = []
            for x_test, y_test in validation_loader:
                x_test = [image.to(device) for image in x_test]
                y_test = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y_test]
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):   # Mixed Precision
                    losses = model(x_test, y_test)
                    loss = sum([loss for loss in losses.values()])
                cv_loss_list.append(loss.data.item())
        
        end = time.time()
        print(f'Epoch: {(epoch + 1)}/{epochs}, Training Loss: {np.mean(train_loss_list):.3f}, Validation Loss: {np.mean(cv_loss_list):.3f}, Elapsed Time: {end - start:.1f}s')
        
        scheduler.step()            # Next step for warmup
        torch.cuda.empty_cache()    # Clear cache to free up some memory

        # Evaluate performance on validation set every 5 epoch
        model.eval()
        if epoch >= 4 and (epoch + 1) % 5 == 0:     # Small subset of training set every 5 epoch
            with torch.no_grad():
                avg_precision = 0.
                avg_recall = 0.
                num_batch = 0
                for x, y in validation_loader:
                    x = [image.to(device) for image in x]
                    y = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y]

                    predictions = model(x)

                    precision, recall = evaluate(predictions, y, iou_threshold, confidence_threshold)
                    avg_precision += precision
                    avg_recall += recall
                    num_batch += 1
                avg_precision /= num_batch
                avg_recall /= num_batch
                
                print(f'Validation Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}\n')
