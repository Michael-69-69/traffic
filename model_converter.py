import os
import json
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_standalone_keras_to_tf(input_path, output_path=None):
    """
    Convert a standalone Keras model to TensorFlow SavedModel format
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_tf"
    
    logger.info(f"Model conversion started: {input_path} -> {output_path}")
    
    # Load the Keras model
    try:
        keras_model = tf.keras.models.load_model(input_path, compile=False)
        logger.info(f"Loaded Keras model from {input_path}")
    except Exception as e:
        logger.error(f"Failed to load Keras model: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save as SavedModel format
    try:
        tf.keras.models.save_model(keras_model, output_path, save_format="tf")
        logger.info(f"Successfully converted model to SavedModel format at {output_path}")
        
        # Optional: Create metadata file
        with open(os.path.join(output_path, "conversion_metadata.json"), "w") as f:
            json.dump({
                "source_model": input_path,
                "conversion_type": "saved_model",
                "version": "1.0"
            }, f)
        logger.info(f"Created metadata at {output_path}/conversion_metadata.json")
        
        return True
    except Exception as e:
        logger.error(f"Failed to convert model to SavedModel format: {e}")
        return False

if __name__ == "__main__":
    # Define input model paths
    road_model_path = "unet_road_segmentation.keras"
    vehicle_model_path = "unet_multi_classV1.keras"
    
    # Define output model paths
    road_model_output = "unet_road_segmentation_tf"
    vehicle_model_output = "unet_multi_classV1_tf"
    
    # Convert models
    success1 = convert_standalone_keras_to_tf(road_model_path, road_model_output)
    success2 = convert_standalone_keras_to_tf(vehicle_model_path, vehicle_model_output)
    
    if success1 and success2:
        logger.info("Models successfully converted to SavedModel format")
    else:
        logger.warning("Model conversion incomplete")