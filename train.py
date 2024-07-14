import torch
import numpy as np
import time
from evaluate import evaluate


# Train and Evaluate Model
def train_model(model, device, train_loader, validation_loader, optimizer, epochs, iou_threshold=0.5, confidence_threshold=0.5):
    avg_precision = 0.
    avg_recall = 0.
    avg_accuracy = 0.

    accumulation_size = 3
    print('Training...')
    for epoch in range(epochs):
        # Train
        model.train()               # Train Mode: requires_grad=True, Batch Norm on
        loss_sublist = []
        loss_box_list = []
        loss_labels_list = []

        start = time.time()
        for batch_index, (x, y) in enumerate(train_loader):   # Mini-Batch Gradient Descent
            # Move tensors to GPU
            model.train()
            x = [image.to(device) for image in x]
            y = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y]
            
            losses = model(x, y)    # Returns dictionary of losses
            loss_box = losses['loss_box_reg'] / accumulation_size
            loss_labels = losses['loss_classifier'] / accumulation_size
            loss = sum([loss for loss in losses.values()]) / accumulation_size      # Total loss = sum of all losses; normalize
            loss_sublist.append(loss.data.item())
            loss_box_list.append(loss_box.data.item())
            loss_labels_list.append(loss_labels.data.item())
            loss.backward()         # Compute gradient of loss

            if (batch_index + 1) % accumulation_size == 0 or (batch_index + 1) == len(train_loader):
                optimizer.step()        # Update parameters
                optimizer.zero_grad()   # clear old gradient before new gradient calculation
        end = time.time()

        print(f'Epoch: {(epoch + 1)}/{epochs}, Training Loss: {np.mean(loss_sublist):.4f}, Loss Box: {np.mean(loss_box_list):.4f}, Loss Labels: {np.mean(loss_labels_list):.4f}, Elapsed Time: {end - start:.2f}s')

    # Evaluate
    model.eval()                # Eval Mode: requires_grad=False, Batch Norm off
    with torch.no_grad():
        avg_precision = 0.
        avg_recall = 0.
        avg_accuracy = 0.
        num_batch = 0

        start = time.time()
        for x_test, y_test in validation_loader:
            '''
            List of dictionaries containing labels, scores, masks, and bbox for each object
            detected. N elements in list for N images passed as parameter. 
            '''
            x_test = [image.to(device) for image in x_test]
            y_test = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()} for target in y_test]
            predictions = model(x_test)
            precision, recall, accuracy = evaluate(predictions, y_test, iou_threshold, confidence_threshold)
            
            avg_precision += precision
            avg_recall += recall
            avg_accuracy += accuracy
            num_batch += 1
        avg_precision /= num_batch
        avg_recall /= num_batch
        avg_accuracy /= num_batch
        end = time.time()

        print(f'\nPrecision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Accuracy: {avg_accuracy * 100:.2f}%, Elapsed Time: {end - start:.2f}s')
