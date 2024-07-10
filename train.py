import torch
import numpy as np
from evaluate import evaluate


# Train and Evaluate Model
def train_model(model, device, train_loader, validation_loader, optimizer, epochs, iou_threshold=0.5, confidence_threshold=0.5):
    for epoch in range(epochs):
        # Train
        model.train()               # Train Mode: requires_grad=True, Batch Norm on
        loss_sublist = []
        for x, y in train_loader:   # Mini-Batch Gradient Descent
            optimizer.zero_grad()   # clear old gradient before new gradient calculation
            losses = model(x, y)    # Returns dictionary of losses
            loss = sum([loss for loss in losses.values()])      # Total loss = sum of all losses
            loss_sublist.append(loss.data.item())
            loss.backward()         # Compute gradient of loss
            optimizer.step()        # Update parameters
        
        # Evaluate
        model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
        with torch.no_grad():
            avg_precision = 0.
            avg_recall = 0.
            num_batch = 0
            for x_test, y_test in validation_loader:
                '''
                List of dictionaries containing labels, scores, masks, and bbox for each object
                detected. N elements in list for N images passed as parameter. 
                '''
                predictions = model(x_test)
                precision, recall = evaluate(predictions, y_test, iou_threshold, confidence_threshold)
                
                avg_precision += precision
                avg_recall += recall
                num_batch += 1
            avg_precision /= num_batch
            avg_recall /= num_batch

        print(f'Training Loss: {np.mean(loss_sublist)}. Precision: {avg_precision}, Recall: {avg_recall}')
