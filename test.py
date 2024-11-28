from model.inference import load_model

import os
import yaml
import torch
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader.load_data import create_datasets, create_data_loaders
from model.inference import load_model
from visualizer.visualize import visualize_data
import time 

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_segmentation_metrics(predictions, targets, threshold=0.5):
    """
    Calculate precision, recall, IoU, and accuracy for binary segmentation with thresholding.
    """
    # Convert model output probabilities to binary class predictions
    #predicted_probs = torch.softmax(predictions, dim=1)
    #predicted_masks = (predicted_probs[:, 1] > threshold).float()
    predicted_masks = []
    for prediction in predictions:
        mask = torch.argmax(prediction, dim=0)
        predicted_masks.append(mask)
    
    predicted_masks = torch.stack(predicted_masks)
      
    # Flatten predictions and targets for pixel-wise comparison
    targets = targets.view(-1) #torch.argmax(targets, dim=1) #.view(-1)
    predicted_masks = predicted_masks.view(-1)

    # Calculate TP, FP, TN, FN
    true_positive = (predicted_masks == 1) & (targets == 1)
    false_positive = (predicted_masks == 1) & (targets == 0)
    false_negative = (predicted_masks == 0) & (targets == 1)
    true_negative = (predicted_masks == 0) & (targets == 0)

    # Metrics
    precision = true_positive.sum().float() / (true_positive.sum() + false_positive.sum() + 1e-6)
    recall = true_positive.sum().float() / (true_positive.sum() + false_negative.sum() + 1e-6)
    iou = true_positive.sum().float() / (true_positive.sum() + false_positive.sum() + false_negative.sum() + 1e-6)
    accuracy = (true_positive.sum().float() + true_negative.sum().float()) / (targets.numel() + 1e-6)
   
    return precision.item(), recall.item(), iou.item(), accuracy.item()

    
config_path = 'trainer_config.yaml'
config = load_config(config_path)

data_loader_config = load_config(config['DATA_LOADER_CONFIG_PATH'])
train_dataset, val_dataset, test_dataset = create_datasets(data_loader_config)
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, 
                                                            data_loader_config['data']['batch_size'])
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model = load_model(config['MODEL_CONFIG_PATH'], device)
model = model.to(device)

start = time.time()
model.eval()
val_loss, total_val_precision, total_val_recall, total_val_iou, total_val_accuracy = 0, 0, 0, 0, 0

with torch.no_grad():
    for val_inputs, val_labels in test_loader:
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
        val_labels = val_labels.squeeze(1)
        val_labels = val_labels.long()
        
        val_outputs = model(val_inputs)

        # Validation loss and metrics
        #val_loss += loss_criterion(val_outputs, val_labels).item()
        precision, recall, iou, accuracy = calculate_segmentation_metrics(val_outputs, val_labels)
        total_val_precision += precision
        total_val_recall += recall
        total_val_iou += iou
        total_val_accuracy += accuracy
        
end = time.time()
print(end - start)
#avg_val_loss = val_loss / len(val_loader)
avg_val_precision = total_val_precision / len(val_loader)
avg_val_recall = total_val_recall / len(val_loader)
avg_val_iou = total_val_iou / len(val_loader)
avg_val_accuracy = total_val_accuracy / len(val_loader)

#print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(f"Average Test Precision: {avg_val_precision:.4f}")
print(f"Average Test Recall: {avg_val_recall:.4f}")
print(f"Average Test IoU: {avg_val_iou:.4f}")
print(f"Average Test Accuracy: {avg_val_accuracy:.4f}")

exit()
if config['TEST_VISUALIZATION']:
    visualize_data(config['TEST_IMAGE_PATH'], config['TEST_VISUALIZATION_PATH'], 
                       model, config['RESIZE'], device)
