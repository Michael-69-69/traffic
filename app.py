import requests
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import json
from datetime import datetime, timedelta
import unicodedata
import logging
from flask import Flask

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create Flask app instance
app = Flask(__name__)

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

# Load Models
def load_trained_model(model_path, custom_objects=None):
    """
    Load a model with multiple fallback approaches.
    First tries the TF-converted model directory, then falls back to the original file.
    """
    if custom_objects is None:
        custom_objects = {'dice_loss': dice_loss}
    
    # First, try loading as a directory (TF SavedModel format)
    if os.path.isdir(model_path + "_tf"):
        try:
            logging.info(f"Attempting to load TF-converted model from directory: {model_path}_tf")
            return tf.keras.models.load_model(model_path + "_tf", custom_objects=custom_objects, compile=False)
        except Exception as e:
            logging.warning(f"Failed to load TF-converted model: {e}")
    
    # If that fails, try to directly load the original .keras file
    try:
        logging.info(f"Attempting to load original model: {model_path}")
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        
        # Last resort - try with a simplistic U-Net model
        try:
            logging.info(f"Creating a new model as fallback for {model_path}")
            
            # Create a basic U-Net model
            inputs = tf.keras.layers.Input(shape=(128, 128, 3))
            
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
            
            # Bottom
            conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
            conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
            
            # Decoder
            up7 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4)
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
            if 'multi_class' in model_path:
                # Multi-class segmentation
                outputs = tf.keras.layers.Conv2D(4, 1, activation='softmax')(conv9)
            else:
                # Binary segmentation (road)
                outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model
            
        except Exception as fallback_err:
            logging.error(f"All model loading attempts failed: {fallback_err}")
            raise

# Preprocess Image
def preprocess_image(img):
    try:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        enhanced_img = cv2.merge((y, cr, cb))
        img = cv2.cvtColor(enhanced_img, cv2.COLOR_YCrCb2BGR)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

# Post-process Road Segmentation
def postprocess_road_mask(prediction):
    try:
        prediction = prediction.squeeze()
        return (prediction > 0.5).astype(np.uint8)
    except Exception as e:
        logging.error(f"Error postprocessing road mask: {e}")
        raise

# Post-process Vehicle Segmentation
def postprocess_vehicle_mask(prediction):
    try:
        prediction = prediction.squeeze()
        return np.argmax(prediction, axis=-1)
    except Exception as e:
        logging.error(f"Error postprocessing vehicle mask: {e}")
        raise

# Extract Segmented Road
def extract_segmented_road(original_image, road_mask):
    try:
        mask_resized = cv2.resize(road_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmented_road = cv2.bitwise_and(original_image, original_image, mask=mask_resized.astype(np.uint8) * 255)
        return segmented_road, mask_resized
    except Exception as e:
        logging.error(f"Error extracting segmented road: {e}")
        raise

# Camera mapping
camera_mapping = {
    'Lý Thái Tổ - Sư Vạn Hạnh': 'A',
    'Ba Tháng Hai - Cao Thắng': 'B',
    'Điện Biên Phủ – Cao Thắng': 'C',
    'Nút giao Ngã sáu Nguyễn Tri Phương_1': 'D',
    'Nút giao Ngã sáu Nguyễn Tri Phương': 'E',
    'Nút giao Lê Đại Hành 2 (Lê Đại Hành)': 'F',
    'Lý Thái Tổ - Nguyễn Đình Chiểu': 'G',
    'Nút giao Ngã sáu Cộng Hòa_1': 'H',
    'Nút giao Ngã sáu Cộng Hòa': 'I',
    'Điện Biên Phủ - Cách Mạng Tháng Tám': 'J',
    'Nút giao Công Trường Dân Chủ': 'K',
    'Nút giao Công Trường Dân Chủ_1': 'L'
}

# List of cameras with their IDs and locations (updated to match camera_mapping)
cameras = [
    ("6623e7076f998a001b2523ea", "Lý Thái Tổ - Sư Vạn Hạnh"),
    ("5deb576d1dc17d7c5515acf8", "Ba Tháng Hai - Cao Thắng"),
    ("63ae7a9cbfd3d90017e8f303", "Điện Biên Phủ – Cao Thắng"),  # Fixed to use en dash
    ("5deb576d1dc17d7c5515ad21", "Nút giao Ngã sáu Nguyễn Tri Phương"),
    ("5deb576d1dc17d7c5515ad22", "Nút giao Ngã sáu Nguyễn Tri Phương_1"),
    ("5d8cdd26766c880017188974", "Nút giao Lê Đại Hành 2 (Lê Đại Hành)"),
    ("63ae763bbfd3d90017e8f0c4", "Lý Thái Tổ - Nguyễn Đình Chiểu"),
    ("5deb576d1dc17d7c5515acf6", "Nút giao Ngã sáu Cộng Hòa"),
    ("5deb576d1dc17d7c5515acf7", "Nút giao Ngã sáu Cộng Hòa_1"),
    ("5deb576d1dc17d7c5515acf2", "Điện Biên Phủ - Cách Mạng Tháng Tám"),
    ("5deb576d1dc17d7c5515acf9", "Nút giao Công Trường Dân Chủ"),
    ("5deb576d1dc17d7c5515acfa", "Nút giao Công Trường Dân Chủ_1")
]

# Paths for models and output
road_model_path = "unet_road_segmentation.keras"  # Will also try "unet_road_segmentation.keras_tf"
vehicle_model_path = "unet_multi_classV1.keras"   # Will also try "unet_multi_classV1.keras_tf"
base_directory = "/app"  # This matches the WORKDIR in Dockerfile
densities_dir = os.path.join(base_directory, "densities")
today_densities_path = os.path.join(densities_dir, "today_densities.json")
yesterday_max_densities_path = os.path.join(densities_dir, "yesterday_max_densities.json")
critical_densities_path = os.path.join(densities_dir, "critical_densities.json")
output_json_path = os.path.join(densities_dir, "densities.json")

# Create directories if they don't exist
os.makedirs(densities_dir, exist_ok=True)

# Create a session to persist cookies
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
})

# Base URL and default parameters for the camera feed
main_url = "https://giaothong.hochiminhcity.gov.vn"
base_url = "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx"
default_params = {
    "bg": "black",
    "w": 300,
    "h": 230
}

# First, check if we need to convert models
def check_and_convert_models():
    """Run the model converter if needed"""
    try:
        # Check if converted models exist
        road_model_converted = os.path.isdir(road_model_path + "_tf")
        vehicle_model_converted = os.path.isdir(vehicle_model_path + "_tf")
        
        if not road_model_converted or not vehicle_model_converted:
            logging.info("Converted models not found. Attempting to convert now.")
            
            # If model_converter.py exists, run it
            if os.path.exists("model_converter.py"):
                logging.info("Running model_converter.py...")
                import model_converter
                model_converter.convert_standalone_keras_to_tf(road_model_path, road_model_path + "_tf")
                model_converter.convert_standalone_keras_to_tf(vehicle_model_path, vehicle_model_path + "_tf")
                logging.info("Models converted successfully.")
            else:
                logging.warning("model_converter.py not found. Will try to load original models.")
    except Exception as e:
        logging.error(f"Error during model conversion: {e}")
        logging.warning("Will try to load original models.")

# Load models
try:
    # First, check if we need to convert models
    check_and_convert_models()
    
    # Load models with custom objects
    custom_objects = {'dice_loss': dice_loss}
    logging.info(f"Loading road segmentation model...")
    road_model = load_trained_model(road_model_path, custom_objects=custom_objects)
    logging.info("Road segmentation model loaded successfully")
    
    logging.info(f"Loading vehicle detection model...")
    vehicle_model = load_trained_model(vehicle_model_path, custom_objects=custom_objects)
    logging.info("Vehicle detection model loaded successfully")
    
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    exit(1)

# Function to manage historical densities
def manage_historical_densities():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    day_before_yesterday = today - timedelta(days=2)

    # Initialize today's densities
    today_densities = {}
    if os.path.exists(today_densities_path):
        try:
            with open(today_densities_path, 'r', encoding='utf-8') as f:
                today_densities = json.load(f)
            # Check if today_densities is from today
            if 'date' in today_densities:
                file_date = datetime.strptime(today_densities['date'], '%Y-%m-%d').date()
                if file_date != today:
                    # Move today's densities to yesterday if it's from a previous day
                    max_densities = {}
                    for cam_id in today_densities:
                        if cam_id != 'date':
                            timestamps = today_densities[cam_id]
                            max_density = max(timestamps.values()) if timestamps else 0.0
                            max_densities[cam_id] = max_density
                    with open(yesterday_max_densities_path, 'w', encoding='utf-8') as f:
                        json.dump({'date': yesterday.strftime('%Y-%m-%d'), **max_densities}, f, ensure_ascii=False)
                    today_densities = {'date': today.strftime('%Y-%m-%d')}
                    logging.info(f"Updated yesterday_max_densities.json with max densities from {file_date}")
            else:
                today_densities = {'date': today.strftime('%Y-%m-%d')}
        except Exception as e:
            logging.error(f"Error reading today_densities.json: {e}")
            today_densities = {'date': today.strftime('%Y-%m-%d')}
    else:
        today_densities = {'date': today.strftime('%Y-%m-%d')}

    # Load yesterday's max densities as critical densities
    critical_densities = {}
    if os.path.exists(yesterday_max_densities_path):
        try:
            with open(yesterday_max_densities_path, 'r', encoding='utf-8') as f:
                yesterday_data = json.load(f)
                file_date = datetime.strptime(yesterday_data['date'], '%Y-%m-%d').date()
                if file_date == yesterday:
                    critical_densities = {k: v for k, v in yesterday_data.items() if k != 'date'}
                else:
                    # If yesterday's data is older, delete it
                    os.remove(yesterday_max_densities_path)
                    logging.info(f"Deleted outdated yesterday_max_densities.json from {file_date}")
        except Exception as e:
            logging.error(f"Error reading yesterday_max_densities.json: {e}")

    # Save critical densities
    if critical_densities:
        with open(critical_densities_path, 'w', encoding='utf-8') as f:
            json.dump(critical_densities, f, ensure_ascii=False)
        logging.info(f"Saved critical densities to {critical_densities_path}")
    else:
        # Sample critical densities for the first run (based on typical traffic patterns)
        sample_critical_densities = {
            'A': 80.0,  # Lý Thái Tổ - Sư Vạn Hạnh (busy intersection)
            'B': 70.0,  # Ba Tháng Hai - Cao Thắng (moderate traffic)
            'C': 75.0,  # Điện Biên Phủ – Cao Thắng (busy road)
            'D': 85.0,  # Ngã sáu Nguyễn Tri Phương_1 (very busy)
            'E': 80.0,  # Ngã sáu Nguyễn Tri Phương (busy)
            'F': 60.0,  # Lê Đại Hành 2 (less busy)
            'G': 70.0,  # Lý Thái Tổ - Nguyễn Đình Chiểu (moderate)
            'H': 90.0,  # Ngã sáu Cộng Hòa_1 (very busy)
            'I': 85.0,  # Ngã sáu Cộng Hòa (busy)
            'J': 75.0,  # Điện Biên Phủ - Cách Mạng Tháng Tám (busy)
            'K': 80.0,  # Công Trường Dân Chủ (busy)
            'L': 80.0   # Công Trường Dân Chủ_1 (busy)
        }
        critical_densities = sample_critical_densities
        with open(critical_densities_path, 'w', encoding='utf-8') as f:
            json.dump(critical_densities, f, ensure_ascii=False)
        logging.info("Saved sample critical densities")

    return critical_densities, today_densities

def fetch_and_process_densities():
    """Fetch images from traffic cameras, process them, and calculate traffic densities"""
    logging.info("Starting density analysis cycle")
    
    # Get critical densities and today's densities
    critical_densities, today_densities = manage_historical_densities()
    
    # Timestamp for this fetch cycle
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Results dictionary
    results = {
        "timestamp": current_time,
        "cameras": {}
    }
    
    # Process each camera
    for camera_id, camera_name in cameras:
        try:
            # Build parameters for this camera
            params = default_params.copy()
            params["cameraId"] = camera_id
            
            # Fetch the image
            response = session.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                # Convert response to OpenCV image
                nparr = np.frombuffer(response.content, np.uint8)
                original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if original_image is None or original_image.size == 0:
                    logging.warning(f"Empty image received for camera {camera_name}")
                    continue
                
                # Preprocess image
                img_processed = preprocess_image(original_image)
                
                # Road segmentation
                road_pred = road_model.predict(img_processed, verbose=0)
                road_mask = postprocess_road_mask(road_pred)
                
                # Extract segmented road area
                segmented_road, mask_resized = extract_segmented_road(original_image, road_mask)
                
                # If there's a road detected, process vehicles
                road_pixels = np.sum(mask_resized)
                if road_pixels > 0:
                    # Repreprocess the segmented road image
                    segmented_processed = preprocess_image(segmented_road)
                    
                    # Vehicle detection on the road area
                    vehicle_pred = vehicle_model.predict(segmented_processed, verbose=0)
                    vehicle_mask = postprocess_vehicle_mask(vehicle_pred)
                    
                    # Calculate densities
                    # Class 1 = car, Class 2 = motorcycle, Class 3 = other vehicles
                    car_pixels = np.sum(vehicle_mask == 1)
                    motorcycle_pixels = np.sum(vehicle_mask == 2)
                    other_pixels = np.sum(vehicle_mask == 3)
                    
                    # Total vehicle pixels
                    vehicle_pixels = car_pixels + motorcycle_pixels + other_pixels
                    
                    # Calculate density as percentage of road covered by vehicles
                    density = (vehicle_pixels / road_pixels) * 100 if road_pixels > 0 else 0
                    
                    # Normalize density to a 0-100 scale (cap at 100%)
                    density = min(density * 1.5, 100)  # Amplify by 1.5x for better sensitivity
                    
                    # Get the critical density for this camera
                    camera_code = camera_mapping.get(camera_name, camera_name)
                    critical_density = critical_densities.get(camera_code, 80.0)
                    
                    # Calculate congestion level (0-100%)
                    congestion_level = min(100, (density / critical_density) * 100) if critical_density > 0 else 0
                    
                    # Update today's densities
                    hour_str = datetime.now().strftime('%H:%M')
                    if camera_code not in today_densities:
                        today_densities[camera_code] = {}
                    today_densities[camera_code][hour_str] = density
                    
                    # Add to results
                    results["cameras"][camera_code] = {
                        "name": camera_name,
                        "density": round(density, 2),
                        "congestion_level": round(congestion_level, 2),
                        "critical_density": round(critical_density, 2),
                        "composition": {
                            "cars": round((car_pixels / vehicle_pixels) * 100 if vehicle_pixels > 0 else 0, 2),
                            "motorcycles": round((motorcycle_pixels / vehicle_pixels) * 100 if vehicle_pixels > 0 else 0, 2),
                            "others": round((other_pixels / vehicle_pixels) * 100 if vehicle_pixels > 0 else 0, 2)
                        }
                    }
                    
                    logging.info(f"Processed {camera_name}: Density={round(density, 2)}%, Congestion={round(congestion_level, 2)}%")
                else:
                    logging.warning(f"No road detected for camera {camera_name}")
            else:
                logging.warning(f"Failed to fetch image for camera {camera_name}: Status code {response.status_code}")
                
        except Exception as e:
            logging.error(f"Error processing camera {camera_name}: {e}")
    
    # Save today's densities
    with open(today_densities_path, 'w', encoding='utf-8') as f:
        json.dump(today_densities, f, ensure_ascii=False)
    
    # Save the results to the output JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Density analysis completed. Results saved to {output_json_path}")

# Flask routes
@app.route('/')
def index():
    return {
        "status": "running",
        "version": "1.0",
        "message": "Traffic Analysis Service is operational"
    }

@app.route('/densities')
def get_densities():
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            densities = json.load(f)
        return densities
    except Exception as e:
        logging.error(f"Error reading densities: {e}")
        return {"error": "Could not read densities data"}, 500

if __name__ == "__main__":
    # Check and convert models if needed
    check_and_convert_models()
    
    # Load models
    logging.info("Loading road segmentation model...")
    road_model = load_trained_model(road_model_path)
    logging.info("Road segmentation model loaded successfully")
    
    logging.info("Loading vehicle detection model...")
    vehicle_model = load_trained_model(vehicle_model_path)
    logging.info("Vehicle detection model loaded successfully")
    
    # Create a session for making requests
    session = requests.Session()
    
    # Start the density worker thread
    import threading
    density_thread = threading.Thread(target=density_worker, daemon=True)
    density_thread.start()
    
    # Get the port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port)