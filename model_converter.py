"""
This script converts models from standalone Keras format to TF-compatible format
Run this before running the main app
"""
import numpy as np
import tensorflow as tf
import os
import json
import h5py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dice Loss (exact same as in the app.py file)
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def convert_standalone_keras_to_tf(input_path, output_path=None):
    """
    Convert a model file saved with standalone keras to a TensorFlow compatible format
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_tf{ext}"
    
    logging.info(f"Converting model: {input_path} -> {output_path}")
    
    try:
        # Try to open the H5 file
        with h5py.File(input_path, 'r') as h5file:
            # Get model config
            if 'model_config' in h5file.attrs:
                model_config = json.loads(h5file.attrs['model_config'].decode('utf-8'))
                
                # Replace 'keras.src' references with 'tensorflow.keras'
                config_str = json.dumps(model_config)
                config_str = config_str.replace('keras.src', 'tensorflow.keras')
                config_str = config_str.replace('keras.', 'tensorflow.keras.')
                modified_config = json.loads(config_str)
                
                # Build model from modified config
                custom_objects = {'dice_loss': dice_loss}
                model = tf.keras.models.model_from_config(
                    modified_config, custom_objects=custom_objects
                )
                
                # If weights are in the file, load them
                if 'model_weights' in h5file:
                    weight_names = []
                    weight_values = []
                    
                    for name, layer in zip(h5file['model_weights'].attrs['weight_names'], h5file['model_weights']):
                        weight_values.append(np.array(layer))
                        weight_names.append(name.decode('utf-8') if isinstance(name, bytes) else name)
                    
                    # Set weights
                    if weight_names:
                        model.set_weights(weight_values)
                
                # Save model in TensorFlow format
                model.save(output_path, save_format='tf')
                logging.info(f"Successfully converted and saved model to {output_path}")
                return True
            else:
                logging.error("Model config not found in H5 file")
                return False
                
    except Exception as e:
        logging.error(f"Error converting model: {e}")
        
        # Try a fallback approach - rebuild the model architecture and attempt to load weights
        try:
            logging.info("Attempting to rebuild the model architecture...")
            
            # Create a basic U-Net model with the expected architecture
            inputs = tf.keras.layers.Input(shape=(128, 128, 3), name="input_layer")
            
            # Encoder
            conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
            pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            
            conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
            conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
            pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            
            conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
            conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
            pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
            
            conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
            conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
            pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
            
            # Bottom
            conv5 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
            conv5 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)
            
            # Decoder
            up6 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv5)
            merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
            conv6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge6)
            conv6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)
            
            up7 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
            merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
            conv7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge7)
            conv7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
            
            up8 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
            merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
            conv8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge8)
            conv8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
            
            up9 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
            merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
            conv9 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(merge9)
            conv9 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
            
            # Output layer
            if 'multi_class' in input_path:
                # For multi-class segmentation with multiple classes
                outputs = tf.keras.layers.Conv2D(4, 1, activation='softmax', padding='valid')(conv9)
            else:
                # For binary segmentation (road segmentation)
                outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='valid')(conv9)
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile the model
            model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
            
            # Save the model directly
            model.save(output_path, save_format='tf')
            logging.info(f"Successfully created and saved a new model to {output_path}")
            return True
            
        except Exception as fallback_error:
            logging.error(f"Fallback approach also failed: {fallback_error}")
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
        logging.info("All models converted successfully!")
        logging.info("Now you should update your app.py to use these converted models")
    else:
        logging.warning("Some models failed to convert. Please check the logs.")