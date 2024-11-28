import torch
import logging
import sys
from .hrnet import get_seg_model
from model.model_config.config_loader import yaml_to_dotdict  

# Setup logging for production-level code
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(config_path, device):
    """
    Load the model from the configuration file.
    
    Args:
        config_path (str): Path to the model configuration file.

    Returns:
        torch.nn.Module: The loaded segmentation model.
    """
    logging.info(f"Loading model configuration from {config_path}")
    try:
        config = yaml_to_dotdict(config_path)
        model = get_seg_model(config, device)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)


def main(config_path):
    """
    Main function to load the model, move it to the GPU if available, and run inference.

    Args:
        config_path (str): Path to the model configuration file.
    """
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load the model
    model = load_model(config_path, device)
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Create a random input tensor and move it to the correct device
    random_tensor = torch.rand((1, 3, 400, 200), dtype=torch.float32)
    random_tensor = random_tensor.to(device)
    
    logging.info("Running inference...")
    out_vector = model(random_tensor)
    logging.info(f"Inference completed. Output shape: {out_vector.shape}, Output device: {out_vector.device}")
   
'''
if __name__ == "__main__":
    
    config_path = '/home/remote/u7669839/drought_detection/drought_detection/model/model_config/model_config.py'
    main(config_path)
'''