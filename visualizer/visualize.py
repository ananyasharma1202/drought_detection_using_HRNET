import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import logging
import shutil

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_image(image_path, resize):
    """Preprocesses the image for model input."""
    try:
        img = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        return preprocess(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def save_mask(mask, outfolder, filename):
    """Saves the predicted mask to the output folder."""
    try:
        outfile = os.path.join(outfolder, filename)
        Image.fromarray(mask, mode='L').save(outfile)
        logging.info(f"Saved mask to {outfile}")
    except Exception as e:
        logging.error(f"Error saving mask {filename}: {e}")

def visualize_data(image_path, outfolder, model, resize, device="cpu"):
    """Visualizes data by predicting and saving segmentation masks."""
    
    model.eval()
    
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)

    os.makedirs(outfolder)
    logging.info(f"Created output directory: {outfolder}")

    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        img_tensor = preprocess_image(file_path, resize)

        
        if img_tensor is None:
            continue  # Skip if image preprocessing failed
        
        img_tensor = img_tensor.to(device)

        # Predict the segmentation mask
        try:
            with torch.no_grad():
                predictions = model(img_tensor)  # Shape: [1, 1, height, width]
                #predicted_mask = (torch.sigmoid(outputs) > 0.5).float()  # Binarize the output
            
            if predictions.shape[1] == 2:  # Assuming binary segmentation with 2 channels
                predictions = torch.softmax(predictions, dim=1)

            # Get the predicted class (binary mask) by thresholding on class 1 probability
            threshold = 0.5
            predicted_mask = (predictions[:, 1, :, :] > threshold).long()
           
            predicted_mask = predicted_mask.squeeze().cpu().numpy() * 255  # Shape: [height, width]
            predicted_mask = predicted_mask.astype(np.uint8)  # Convert to uint8 type
            
            # Save the predicted mask
            save_mask(predicted_mask, outfolder, filename)
            


        except Exception as e:
            logging.error(f"Error during prediction for {filename}: {e}")


