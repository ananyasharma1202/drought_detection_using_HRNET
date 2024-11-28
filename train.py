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

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_directory_exists(directory_path):
    """Creates the directory if it does not already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

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

# Main code
config_path = 'trainer_config.yaml'
config = load_config(config_path)

output_folder_path = config['OUTPUT_FOLDER_PATH']
ensure_directory_exists(output_folder_path)

# Prepare additional folders
folders_to_create = ['visualizations', 'weights', 'metrics']
for folder_name in folders_to_create:
    ensure_directory_exists(os.path.join(output_folder_path, folder_name))

# Load datasets
data_loader_config = load_config(config['DATA_LOADER_CONFIG_PATH'])
train_dataset, val_dataset, test_dataset = create_datasets(data_loader_config)
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, 
                                                            data_loader_config['data']['batch_size'])

# Load model
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = load_model(config['MODEL_CONFIG_PATH'], device)
model = model.to(device)

# Loss and optimizer
loss_criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=config['LEARNING_RATE'],
    momentum=config['MOMENTUM'],
    weight_decay=config['WEIGHT_DECAY'],
    nesterov=config['NESTEROV']
)

# Create weight and metric directories for checkpoints
weights_dir = os.path.join(output_folder_path, 'weights')
metrics_dir = os.path.join(output_folder_path, 'metrics')
best_weights_dir = os.path.join(weights_dir, 'best')
epoch_weights_dir = os.path.join(weights_dir, 'epochs')

for folder in [best_weights_dir, epoch_weights_dir, metrics_dir]:
    ensure_directory_exists(folder)
    

# Initialize best IoU tracker
#best_iou_path = os.path.join(best_weights_dir, 'best_iou.txt')
#best_iou = float(open(best_iou_path).read()) if os.path.exists(best_iou_path) else 0.0

best_iou = 0.0

# Training and evaluation loop
for epoch in tqdm(range(config['EPOCHS']), desc="Training"):
    model.train()
    train_loss, total_train_precision, total_train_recall, total_train_iou, total_train_accuracy = 0, 0, 0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels.squeeze(1)
        labels = labels.long()
        optimizer.zero_grad()

        # Forward pass and loss
        outputs = model(inputs)
        
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate metrics
        precision, recall, iou, accuracy = calculate_segmentation_metrics(outputs, labels)
        train_loss += loss.item()
        total_train_precision += precision
        total_train_recall += recall
        total_train_iou += iou
        total_train_accuracy += accuracy
        
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_precision = total_train_precision / len(train_loader)
    avg_train_recall = total_train_recall / len(train_loader)
    avg_train_iou = total_train_iou / len(train_loader)
    avg_train_accuracy = total_train_accuracy / len(train_loader)

    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}")

    # Validation Phase
    '''
    model.eval()
    val_loss, total_val_precision, total_val_recall, total_val_iou, total_val_accuracy = 0, 0, 0, 0, 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_labels = val_labels.squeeze(1)
            val_labels = val_labels.long()
            val_outputs = model(val_inputs)

            # Validation loss and metrics
            val_loss += loss_criterion(val_outputs, val_labels).item()
            precision, recall, iou, accuracy = calculate_segmentation_metrics(val_outputs, val_labels)
            total_val_precision += precision
            total_val_recall += recall
            total_val_iou += iou
            total_val_accuracy += accuracy
            
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_precision = total_val_precision / len(val_loader)
    avg_val_recall = total_val_recall / len(val_loader)
    avg_val_iou = total_val_iou / len(val_loader)
    avg_val_accuracy = total_val_accuracy / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}")
    '''
    # Save model and best IoU
    torch.save(model.state_dict(), os.path.join(epoch_weights_dir, f"model_epoch_last.pth"))
    if avg_train_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), os.path.join(best_weights_dir, 'best_model.pth'))
        with open(best_iou_path, 'w') as f:
            f.write(f"{best_iou:.4f}")

    # Save metrics
    metrics = {
        'train_loss': avg_train_loss,
        'train_iou': avg_train_iou,
        #'val_loss': avg_val_loss,
        #'val_iou': avg_val_iou,
    }
    with open(os.path.join(metrics_dir, 'metrics_epoch_' + str(config['START_EPOCH'] + epoch+1 ) + '.json'), 'w') as f:
        json.dump(metrics, f)

    # Visualization
    #if config['VISUALIZATION_FREQUENCY'] and (config['START_EPOCH'] + epoch+1) % config['VISUALIZATION_FREQUENCY'] == 0:
    #    visualize_data(config['VAL_IMAGE_PATH'], os.path.join(output_folder_path, 'visualizations', 'epoch_' + str(config['START_EPOCH'] + epoch+1 )), 
    #                   model, config['RESIZE'], device)
    
print("Training completed.")
