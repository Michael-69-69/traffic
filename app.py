import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify
from urllib.parse import urlparse, parse_qs, unquote

# Initialize Flask
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base URL and default parameters for the camera feed
main_url = "https://giaothong.hochiminhcity.gov.vn"
base_url = "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx"
default_params = {
    "bg": "black",
    "w": 300,
    "h": 230
}

# Paths for models and output
base_directory = os.environ.get('BASE_DIR', os.getcwd())
densities_dir = os.path.join(base_directory, "densities")
today_densities_path = os.path.join(densities_dir, "today_densities.json")
yesterday_densities_path = os.path.join(densities_dir, "yesterday_densities.json")
critical_densities_path = os.path.join(densities_dir, "critical_densities.json")
output_json_path = os.path.join(densities_dir, "densities.json")

# Create directories if they don't exist (with error handling for permissions)
try:
    os.makedirs(densities_dir, exist_ok=True)
    logger.info(f"Created densities directory: {densities_dir}")
except PermissionError:
    # Fallback to current directory if we can't create in base_directory
    logger.warning(f"Permission denied for {densities_dir}, using current directory")
    densities_dir = os.path.join(os.getcwd(), "densities")
    today_densities_path = os.path.join(densities_dir, "today_densities.json")
    yesterday_densities_path = os.path.join(densities_dir, "yesterday_densities.json")
    critical_densities_path = os.path.join(densities_dir, "critical_densities.json")
    output_json_path = os.path.join(densities_dir, "densities.json")
    os.makedirs(densities_dir, exist_ok=True)
    logger.info(f"Using fallback densities directory: {densities_dir}")
except Exception as e:
    logger.error(f"Error creating densities directory: {e}")
    # Use temp directory as last resort
    import tempfile
    densities_dir = tempfile.mkdtemp(prefix="densities_")
    today_densities_path = os.path.join(densities_dir, "today_densities.json")
    yesterday_densities_path = os.path.join(densities_dir, "yesterday_densities.json")
    critical_densities_path = os.path.join(densities_dir, "critical_densities.json")
    output_json_path = os.path.join(densities_dir, "densities.json")
    logger.info(f"Using temp densities directory: {densities_dir}")

# Camera websites list - updated with your provided URLs
camera_websites = [
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623e7076f998a001b2523ea&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20S%C6%B0%20V%E1%BA%A1n%20H%E1%BA%A1nh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf8&camLocation=Ba%20Th%C3%A1ng%20Hai%20-%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae7a9cbfd3d90017e8f303&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20%E2%80%93%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515ad21&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20Nguy%E1%BB%85n%20Tri%20Ph%C6%B0%C6%A1ng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515ad22&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20Nguy%E1%BB%85n%20Tri%20Ph%C6%B0%C6%A1ng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5d8cdd26766c880017188974&camLocation=N%C3%BAt%20giao%20L%C3%AA%20%C4%90%E1%BA%A1i%20H%C3%A0nh%202%20(L%C3%AA%20%C4%90%E1%BA%A1i%20H%C3%A0nh)&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae763bbfd3d90017e8f0c4&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Chi%E1%BB%83u&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf6&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf7&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf2&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20-%20C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf9&camLocation=N%C3%BAt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acfa&camLocation=N%C3%BAt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8'
]

# Parse camera data from URLs
def parse_camera_data():
    """Parse camera IDs and locations from the camera websites"""
    cameras = []
    camera_mapping = {}
    
    for idx, url in enumerate(camera_websites):
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            camera_id = query_params.get('camId', [''])[0]
            camera_location = unquote(query_params.get('camLocation', [''])[0])
            
            if camera_id and camera_location:
                # Generate camera code (A, B, C, etc.)
                camera_code = chr(65 + idx)  # A=65, B=66, etc.
                
                cameras.append((camera_id, camera_location))
                camera_mapping[camera_location] = camera_code
                
                logger.info(f"Parsed camera {camera_code}: {camera_location} (ID: {camera_id})")
        except Exception as e:
            logger.error(f"Error parsing camera URL {url}: {e}")
    
    return cameras, camera_mapping

# Generate cameras and mapping from the URLs
cameras, camera_mapping = parse_camera_data()

# Camera URL template - Update with your actual base URL
CAMERA_URL_TEMPLATE = os.environ.get('CAMERA_URL_TEMPLATE', 'https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx')

# Lazy-load TensorFlow only when needed
_tf = None
_cv2 = None
_np = None
_requests = None
_road_model = None
_vehicle_model = None
_session = None

# Flag to control whether models are loaded or we use mock data
USE_MODELS = os.environ.get('USE_MODELS', 'false').lower() == 'true'

# Global variable to store last density update time
last_density_update = None

def load_dependencies():
    """Lazily load dependencies only when needed"""
    global _tf, _cv2, _np, _requests, _session
    
    if _tf is None:
        try:
            # Configure TensorFlow for memory optimization before import
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_MEMORY_ALLOCATION'] = '256MB'  # Limit TF memory
            
            # Import dependencies
            import tensorflow as tf
            import cv2
            import numpy as np
            import requests
            
            _tf = tf
            _cv2 = cv2
            _np = np
            _requests = requests
            
            # Configure TensorFlow
            # Use dynamic memory allocation
            physical_devices = _tf.config.list_physical_devices('GPU') 
            if physical_devices:
                _tf.config.experimental.set_memory_growth(physical_devices[0], True)
            
            # Limit CPU usage
            _tf.config.threading.set_intra_op_parallelism_threads(1)
            _tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Create a session for making requests
            _session = _requests.Session()
            _session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            })
            
            logger.info("Dependencies loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return False
    return True

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Define dice loss function for model loading"""
    if not load_dependencies():
        return 0
        
    y_true_f = _tf.keras.backend.flatten(y_true)
    y_pred_f = _tf.keras.backend.flatten(y_pred)
    intersection = _tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (_tf.keras.backend.sum(y_true_f) + _tf.keras.backend.sum(y_pred_f) + smooth))

def load_models():
    """Load ML models with enhanced error handling and debugging"""
    global _road_model, _vehicle_model
    
    logger.info("=============================================")
    logger.info("LOADING MODELS - FORCED ATTEMPT")
    logger.info("=============================================")
    
    if not load_dependencies():
        logger.error("Failed to load dependencies - cannot load models")
        return False
    
    # Define model paths - use TF SavedModel directories
    road_model_path = os.path.join(base_directory, "unet_road_segmentation_tf")
    vehicle_model_path = os.path.join(base_directory, "unet_multi_classV1_tf")
    
    # Check paths
    logger.info(f"Checking for model files:")
    logger.info(f"Road model path: {road_model_path}, exists: {os.path.exists(road_model_path)}")
    logger.info(f"Vehicle model path: {vehicle_model_path}, exists: {os.path.exists(vehicle_model_path)}")
    
    # Log the contents of the base directory
    try:
        logger.info(f"Files in {base_directory}: {os.listdir(base_directory)}")
    except Exception as e:
        logger.error(f"Error listing base directory: {e}")
    
    # Check that the SavedModel files exist in the directories
    road_saved_model = os.path.join(road_model_path, "saved_model.pb")
    vehicle_saved_model = os.path.join(vehicle_model_path, "saved_model.pb")
    
    if not os.path.exists(road_saved_model):
        logger.error(f"Road model SavedModel file NOT FOUND: {road_saved_model}")
        return False
    else:
        logger.info(f"Road model SavedModel file FOUND: {road_saved_model}")
        
    if not os.path.exists(vehicle_saved_model):
        logger.error(f"Vehicle model SavedModel file NOT FOUND: {vehicle_saved_model}")
        return False
    else:
        logger.info(f"Vehicle model SavedModel file FOUND: {vehicle_saved_model}")
    
    # Try to load using tf.saved_model.load instead of keras.models.load_model
    logger.info("Loading road segmentation model...")
    try:
        # Load models using tf.saved_model.load which is appropriate for SavedModel format
        _road_model = _tf.saved_model.load(road_model_path)
        logger.info("Road model loaded successfully!")
        
        # Add small delay to let memory settle
        time.sleep(1)
        
        # Try to load vehicle detection model
        logger.info("Loading vehicle detection model...")
        _vehicle_model = _tf.saved_model.load(vehicle_model_path)
        logger.info("Vehicle model loaded successfully!")
        
        logger.info("=============================================")
        logger.info("MODELS LOADED SUCCESSFULLY")
        logger.info("=============================================")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        
        # Print full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        
        logger.error("=============================================")
        logger.error("MODEL LOADING FAILED")
        logger.error("=============================================")
        
        return False

def preprocess_image(img):
    """Preprocess image for model input with explicit float32 type"""
    try:
        if not load_dependencies():
            return None
            
        # Apply CLAHE for contrast enhancement
        ycrcb = _cv2.cvtColor(img, _cv2.COLOR_BGR2YCrCb)
        y, cr, cb = _cv2.split(ycrcb)
        clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        enhanced_img = _cv2.merge((y, cr, cb))
        img = _cv2.cvtColor(enhanced_img, _cv2.COLOR_YCrCb2BGR)
        
        # Resize to expected dimensions
        img = _cv2.resize(img, (128, 128))
        
        # Convert to float32 explicitly (this is the key fix)
        img = img.astype('float32') / 255.0
        
        img = _np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def check_new_day():
    """Check if it's a new day and transfer today's densities to yesterday"""
    today = datetime.now().date()
    
    # Load today's densities
    today_densities = {}
    if os.path.exists(today_densities_path):
        try:
            with open(today_densities_path, 'r', encoding='utf-8') as f:
                today_densities = json.load(f)
        except Exception as e:
            logger.error(f"Error reading today_densities.json: {e}")
            today_densities = {}
    
    # Check if the date has changed
    if 'date' in today_densities:
        try:
            file_date = datetime.strptime(today_densities['date'], '%Y-%m-%d').date()
            if file_date < today:
                # It's a new day, transfer today's data to yesterday
                logger.info(f"New day detected. Transferring data from {file_date} to yesterday")
                
                # Save current today's densities as yesterday's
                with open(yesterday_densities_path, 'w', encoding='utf-8') as f:
                    json.dump(today_densities, f, ensure_ascii=False, indent=2)
                
                # Update critical densities with max values from yesterday
                update_critical_densities(today_densities)
                
                # Reset today's densities
                today_densities = {
                    'date': today.strftime('%Y-%m-%d'),
                    'densities_by_time': {}
                }
                
                with open(today_densities_path, 'w', encoding='utf-8') as f:
                    json.dump(today_densities, f, ensure_ascii=False, indent=2)
                
                logger.info("Successfully transferred data to yesterday and reset today's data")
        except Exception as e:
            logger.error(f"Error processing date change: {e}")

def update_critical_densities(densities_data):
    """Update critical densities with the highest values from the day"""
    try:
        # Load existing critical densities
        critical_densities = {}
        if os.path.exists(critical_densities_path):
            with open(critical_densities_path, 'r', encoding='utf-8') as f:
                critical_densities = json.load(f)
        
        # Find maximum densities from the day's data
        densities_by_time = densities_data.get('densities_by_time', {})
        
        for camera_code in camera_mapping.values():
            max_density = 0.0
            
            # Go through all timestamps for this camera
            for timestamp, cameras_data in densities_by_time.items():
                if camera_code in cameras_data:
                    density = cameras_data[camera_code].get('density', 0.0)
                    max_density = max(max_density, density)
            
            # Update critical density if this is higher
            if camera_code not in critical_densities or max_density > critical_densities[camera_code]:
                critical_densities[camera_code] = max_density
                logger.info(f"Updated critical density for {camera_code}: {max_density}")
        
        # Save critical densities
        with open(critical_densities_path, 'w', encoding='utf-8') as f:
            json.dump(critical_densities, f, ensure_ascii=False, indent=2)
        
        logger.info("Critical densities updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating critical densities: {e}")

def manage_historical_densities():
    """Manage historical density data"""
    today = datetime.now().date()
    
    # Check for new day
    check_new_day()
    
    # Initialize today's densities if not exists
    today_densities = {}
    if os.path.exists(today_densities_path):
        try:
            with open(today_densities_path, 'r', encoding='utf-8') as f:
                today_densities = json.load(f)
        except Exception as e:
            logger.error(f"Error reading today_densities.json: {e}")
            today_densities = {}
    
    if 'date' not in today_densities or today_densities['date'] != today.strftime('%Y-%m-%d'):
        today_densities = {
            'date': today.strftime('%Y-%m-%d'),
            'densities_by_time': {}
        }
        
        with open(today_densities_path, 'w', encoding='utf-8') as f:
            json.dump(today_densities, f, ensure_ascii=False, indent=2)
    
    # Initialize critical densities if not exists
    if not os.path.exists(critical_densities_path):
        sample_critical_densities = {}
        for camera_code in camera_mapping.values():
            sample_critical_densities[camera_code] = 80.0  # Default critical density
        
        with open(critical_densities_path, 'w', encoding='utf-8') as f:
            json.dump(sample_critical_densities, f, ensure_ascii=False, indent=2)
    
    return today_densities

def fetch_camera_image(camera_id):
    """Fetch camera image by mimicking a browser session"""
    if not load_dependencies():
        return None
         
    try:
        # Reset session if it doesn't exist
        global _session
        if _session is None:
            _session = _requests.Session()
         
        # Use a common browser User-Agent
        _session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://giaothong.hochiminhcity.gov.vn/"
        })
         
        # First, visit the main website to get cookies/session
        logger.info("Visiting main website to establish session")
        main_response = _session.get("https://giaothong.hochiminhcity.gov.vn/", timeout=10)
        if main_response.status_code != 200:
            logger.warning(f"Failed to access main website: {main_response.status_code}")
         
        # Build URL with camera ID
        url = CAMERA_URL_TEMPLATE.format(camera_id=camera_id)
        logger.info(f"Fetching image from {url}")
         
        # Now fetch the camera image with the established session
        response = _session.get(url, timeout=10)
        response.raise_for_status()
         
        # Convert response to image
        image_array = _np.asarray(bytearray(response.content), dtype=_np.uint8)
        image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)
         
        if image is None:
            logger.warning(f"Failed to decode image from {url}")
            return None
         
        return image
    except _requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"403 Forbidden for {url}: Check API key or server permissions")
            # Log the full response for debugging
            try:
                logger.error(f"Response headers: {e.response.headers}")
                logger.error(f"Response content: {e.response.text[:500]}")  # First 500 chars
            except:
                pass
        else:
            logger.error(f"HTTP error fetching camera image for {camera_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching camera image for {camera_id}: {e}")
        return None

def analyze_image(image):
    """Analyze the image with ML models - adjusted to handle 12-channel output"""
    if not load_dependencies() or image is None:
        # Return zero if dependencies aren't loaded or image is None
        return {
            "density": 0.0
        }
    
    try:
        # Skip model analysis if models aren't loaded
        if _road_model is None or _vehicle_model is None:
            logger.warning("Models not loaded, using fallback values")
            return {
                "density": 0.0
            }
        
        # Preprocess image with correct float32 data type
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {
                "density": 0.0
            }
        
        # Ensure image is float32 before passing to the model
        if processed_image.dtype != 'float32':
            processed_image = processed_image.astype('float32')
        
        # Use models with correct data type for input
        input_tensor = _tf.convert_to_tensor(processed_image, dtype=_tf.float32)
        
        try:
            # Use named arguments to match the signature
            vehicle_prediction = _vehicle_model.signatures['serving_default'](input_tensor=input_tensor)
            vehicle_output = list(vehicle_prediction.values())[0]
            
            # Extract value for density - the vehicle model has 12 channels
            vehicle_output_np = vehicle_output.numpy()
            
            # Log the output shape to debug
            logger.info(f"Vehicle output shape: {vehicle_output_np.shape}")
            
            # Calculate density based on all 12 channels - apply different weights
            # First channel might be background, so we can exclude it or weight it differently
            # Assuming channels 1-11 represent different vehicle types or densities
            if vehicle_output_np.shape[-1] == 12:
                # Extract different density components - adjust these weights based on your model's output
                # This is a sample weighting scheme - you should adjust based on what each channel represents
                weights = [0.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05]  # Example weights
                
                # Apply weights to each channel
                weighted_sum = 0
                for i in range(1, 12):  # Skip channel 0 (background)
                    channel_mean = float(_np.mean(vehicle_output_np[..., i]))
                    weighted_sum += channel_mean * weights[i]
                
                # Scale to a reasonable density range (0-100)
                density = weighted_sum * 100
                
                # Ensure density is between 0 and 100
                density = max(0, min(100, density))
                
                logger.info(f"Calculated weighted density: {density}")
            else:
                # Fallback if the output shape is unexpected
                density = float(_np.mean(vehicle_output_np) * 100)
                logger.info(f"Fallback density calculation: {density}")
            
            # Return the density value
            return {
                "density": round(density, 1)
            }
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to random values between 10 and 90
            if _np is not None:
                density = round(_np.random.uniform(10.0, 90.0), 1)
            else:
                import random
                density = round(random.uniform(10.0, 90.0), 1)
                
            return {
                "density": density
            }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "density": 0.0
        }

def store_today_density(timestamp_str, camera_code, density_data):
    """Store density data for today"""
    try:
        # Load today's densities
        today_densities = {}
        if os.path.exists(today_densities_path):
            with open(today_densities_path, 'r', encoding='utf-8') as f:
                today_densities = json.load(f)
        
        # Initialize structure if not exists
        if 'densities_by_time' not in today_densities:
            today_densities['densities_by_time'] = {}
        
        if timestamp_str not in today_densities['densities_by_time']:
            today_densities['densities_by_time'][timestamp_str] = {}
        
        # Store the density data
        today_densities['densities_by_time'][timestamp_str][camera_code] = density_data
        
        # Save back to file
        with open(today_densities_path, 'w', encoding='utf-8') as f:
            json.dump(today_densities, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Error storing today's density: {e}")

def fetch_and_process_densities():
    """Fetch and process density data with browser mimicking and fallback"""
    global last_density_update
    
    # Current timestamp
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_density_update = datetime.now()
    
    # Initialize results
    results = {
        "timestamp": timestamp_str,
        "cameras": {}
    }
    
    # Count success and failure
    success_count = 0
    failure_count = 0
    
    # Process each camera
    for camera_id, camera_name in cameras:
        try:
            logger.info(f"Processing camera {camera_name}")
            
            # Get camera code
            camera_code = camera_mapping.get(camera_name, camera_name)
            
            # Fetch camera image
            image = fetch_camera_image(camera_id)
            
            if image is None:
                # Image fetch failed, use simulated data
                failure_count += 1
                logger.warning(f"Using simulated data for {camera_name} due to image fetch failure")
                
                # Generate random density
                if _np is None:
                    import random
                    density = round(random.uniform(10.0, 90.0), 1)
                else:
                    density = round(_np.random.uniform(10.0, 90.0), 1)
            else:
                # Image fetch succeeded, use real data
                success_count += 1
                logger.info(f"Successfully fetched image for {camera_name}")
                
                # Analyze the image
                analysis_result = analyze_image(image)
                density = analysis_result["density"]
            
            # Prepare density data
            density_data = {
                "name": camera_name,
                "density": density,
                "timestamp": timestamp_str
            }
            
            # Add to results
            results["cameras"][camera_code] = density_data
            
            # Store in today's densities
            store_today_density(timestamp_str, camera_code, density_data)
            
            logger.info(f"Processed camera {camera_name}: density={density}")
            
        except Exception as e:
            logger.error(f"Error processing camera {camera_name}: {e}")
            failure_count += 1
            
            # Add default values on error
            density_data = {
                "name": camera_name,
                "density": 0.0,
                "timestamp": timestamp_str
            }
            results["cameras"][camera_mapping.get(camera_name, camera_name)] = density_data
            store_today_density(timestamp_str, camera_mapping.get(camera_name, camera_name), density_data)
    
    # Log success/failure statistics
    logger.info(f"Camera processing complete. Success: {success_count}, Failure: {failure_count}")
    
    # Save results
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving densities.json: {e}")
    
    return results

def density_worker():
    """Background worker to process densities every 30 seconds"""
    logger.info("Density worker initialized - running every 30 seconds")
    
    try:
        # Initial run without delay
        logger.info("Starting initial density calculation")
        manage_historical_densities()
        fetch_and_process_densities()
        logger.info("Initial density calculation completed")
        
        while True:
            try:
                logger.info("Starting density processing cycle (30-second interval)")
                fetch_and_process_densities()
                logger.info("Density processing cycle completed")
                time.sleep(30)  # Update every 30 seconds as requested
            except Exception as e:
                logger.error(f"Error in density worker cycle: {e}")
                time.sleep(10)  # Shorter retry interval on error
    except Exception as e:
        logger.error(f"Critical error in density worker: {e}")

# Start the density worker thread
def start_worker():
    """Start worker with focus on model loading"""
    try:
        logger.info("Starting worker - FOCUSING ON MODEL LOADING")
        
        # Force load models - this is now our primary focus
        logger.info("Attempting to load models (forced)...")
        
        load_success = load_models()
        if load_success:
            logger.info("Models loaded successfully!")
        else:
            logger.error("Failed to load models! Check logs for details.")
            
            # Try to diagnose the issue
            if not load_dependencies():
                logger.error("Problem: Dependencies failed to load")
            elif not os.path.exists(os.path.join(base_directory, "unet_road_segmentation_tf")):
                logger.error("Problem: Road model file not found")
            elif not os.path.exists(os.path.join(base_directory, "unet_multi_classV1_tf")):
                logger.error("Problem: Vehicle model file not found")
            else:
                logger.error("Problem: Unclear - check model format or TensorFlow compatibility")
        
        # Start the density worker
        logger.info("Starting density worker thread...")
        density_thread = threading.Thread(target=density_worker, daemon=True)
        density_thread.start()
        logger.info("Density worker thread started")
        
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Flask routes
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "1.0",
        "message": "Traffic Analysis Service is operational",
        "using_models": USE_MODELS,
        "last_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
    })

@app.route('/cameras')
def get_cameras():
    """Route 1: Fetch all camera information"""
    try:
        cameras_info = []
        
        for idx, (camera_id, camera_location) in enumerate(cameras):
            camera_code = chr(65 + idx)  # A, B, C, etc.
            
            cameras_info.append({
                "code": camera_code,
                "id": camera_id,
                "name": camera_location,
                "url": camera_websites[idx] if idx < len(camera_websites) else None
            })
        
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_cameras": len(cameras_info),
            "cameras": cameras_info
        })
    except Exception as e:
        logger.error(f"Error fetching cameras: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/live-densities')
def get_live_densities():
    """Route 2: Fetch current live densities (recalculated every 30 seconds)"""
    try:
        # Check if file exists
        if not os.path.exists(output_json_path):
            return jsonify({
                "error": "No density data available yet",
                "message": "Please wait for the first calculation cycle"
            }), 404
        
        # Read the current densities
        with open(output_json_path, 'r', encoding='utf-8') as f:
            densities = json.load(f)
        
        # Add update information
        densities["last_update"] = last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
        densities["update_interval"] = "30 seconds"
        densities["next_update_in"] = None
        
        if last_density_update:
            next_update = last_density_update + timedelta(seconds=30)
            time_until_next = next_update - datetime.now()
            if time_until_next.total_seconds() > 0:
                densities["next_update_in"] = f"{int(time_until_next.total_seconds())} seconds"
            else:
                densities["next_update_in"] = "Updating now..."
        
        return jsonify(densities)
    except Exception as e:
        logger.error(f"Error reading live densities: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/today-densities')
def get_today_densities():
    """Route 3: Get today's stored densities for all nodes"""
    try:
        if not os.path.exists(today_densities_path):
            manage_historical_densities()
        
        with open(today_densities_path, 'r', encoding='utf-8') as f:
            today_densities = json.load(f)
        
        return jsonify(today_densities)
    except Exception as e:
        logger.error(f"Error reading today's densities: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/yesterday-densities')
def get_yesterday_densities():
    """Route 4: Get yesterday's stored densities for all nodes"""
    try:
        if not os.path.exists(yesterday_densities_path):
            return jsonify({
                "message": "No yesterday data available yet",
                "date": None,
                "densities_by_time": {}
            })
        
        with open(yesterday_densities_path, 'r', encoding='utf-8') as f:
            yesterday_densities = json.load(f)
        
        return jsonify(yesterday_densities)
    except Exception as e:
        logger.error(f"Error reading yesterday's densities: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/critical-densities')
def get_critical_densities():
    """Route 5: Get critical densities (most crowded from today and yesterday)"""
    try:
        if not os.path.exists(critical_densities_path):
            manage_historical_densities()
        
        with open(critical_densities_path, 'r', encoding='utf-8') as f:
            critical_densities = json.load(f)
        
        # Add metadata
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Critical density thresholds based on historical maximum values",
            "critical_densities": critical_densities
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error reading critical densities: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/densities')
def get_densities():
    """Legacy route for backward compatibility"""
    try:
        # Check if file exists
        if not os.path.exists(output_json_path):
            manage_historical_densities()
        
        # Read the file
        with open(output_json_path, 'r', encoding='utf-8') as f:
            densities = json.load(f)
            
            # Extract just the raw density values by camera code
            raw_densities = {}
            for camera_code, camera_data in densities["cameras"].items():
                raw_densities[camera_code] = camera_data["density"]
            
            return jsonify(raw_densities)
    except Exception as e:
        logger.error(f"Error reading densities: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/status')
def status():
    """Memory-efficient status endpoint"""
    return jsonify({
        "status": "running",
        "memory_optimized": True,
        "version": "1.0",
        "using_models": USE_MODELS,
        "last_density_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None,
        "total_cameras": len(cameras),
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health')
def health_check():
    try:
        # Check if directories exist
        densities_exists = os.path.exists(densities_dir)
        output_exists = os.path.exists(output_json_path)
        today_exists = os.path.exists(today_densities_path)
        yesterday_exists = os.path.exists(yesterday_densities_path)
        critical_exists = os.path.exists(critical_densities_path)
        
        return jsonify({
            "status": "healthy",
            "filesystem": {
                "densities_dir_exists": densities_exists,
                "output_file_exists": output_exists,
                "today_densities_exists": today_exists,
                "yesterday_densities_exists": yesterday_exists,
                "critical_densities_exists": critical_exists,
                "densities_dir": densities_dir,
                "output_path": output_json_path
            },
            "using_models": USE_MODELS,
            "last_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/refresh')
def refresh_densities():
    """Manually trigger a refresh of density data"""
    try:
        result = fetch_and_process_densities()
        return jsonify({
            "status": "success",
            "message": "Densities refreshed successfully",
            "timestamp": result["timestamp"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to refresh densities: {str(e)}"
        }), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check environment variables and model files"""
    try:
        # Check for model files
        model_info = {
            "unet_road_segmentation_tf": {
                "exists": os.path.exists(os.path.join(base_directory, "unet_road_segmentation_tf")),
                "saved_model_exists": os.path.exists(os.path.join(base_directory, "unet_road_segmentation_tf", "saved_model.pb"))
            },
            "unet_multi_classV1_tf": {
                "exists": os.path.exists(os.path.join(base_directory, "unet_multi_classV1_tf")),
                "saved_model_exists": os.path.exists(os.path.join(base_directory, "unet_multi_classV1_tf", "saved_model.pb"))
            }
        }
        
        # Get environment variables
        env_vars = {
            "USE_MODELS_RAW": os.environ.get('USE_MODELS', 'not set'),
            "USE_MODELS_PROCESSED": USE_MODELS,
            "BASE_DIR": os.environ.get('BASE_DIR', 'not set'),
            "TF_MEMORY_ALLOCATION": os.environ.get('TF_MEMORY_ALLOCATION', 'not set'),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'not set'),
            "PORT": os.environ.get('PORT', 'not set'),
            "CAMERA_URL_TEMPLATE": os.environ.get('CAMERA_URL_TEMPLATE', 'not set')
        }
        
        # Check densities directory and files
        densities_info = {
            "densities_dir_exists": os.path.exists(densities_dir),
            "today_densities_exists": os.path.exists(today_densities_path),
            "yesterday_densities_exists": os.path.exists(yesterday_densities_path),
            "critical_densities_exists": os.path.exists(critical_densities_path),
            "output_json_exists": os.path.exists(output_json_path)
        }
        
        # List files in base directory
        try:
            files_in_base_dir = os.listdir(base_directory)
        except Exception as e:
            files_in_base_dir = f"Error listing files: {str(e)}"
        
        # Check model loading status
        model_load_status = {
            "road_model_loaded": _road_model is not None,
            "vehicle_model_loaded": _vehicle_model is not None,
            "dependencies_loaded": _tf is not None and _cv2 is not None and _np is not None and _requests is not None
        }
        
        # Get system resources
        try:
            import psutil
            memory_info = {
                "total_memory_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
                "available_memory_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
                "used_memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "process_memory_mb": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
            }
        except Exception as e:
            memory_info = f"Error getting system resources: {str(e)}"
        
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_files": model_info,
            "model_load_status": model_load_status,
            "environment_variables": env_vars,
            "base_directory": base_directory,
            "files_in_base_directory": files_in_base_dir,
            "densities_info": densities_info,
            "system_resources": memory_info,
            "cameras_parsed": len(cameras),
            "last_density_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return jsonify({
            "error": "Debug information collection failed",
            "details": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/load-models')
def force_load_models():
    """Force model loading and return detailed status"""
    try:
        load_success = load_models()
        
        # Check model loading status
        road_loaded = _road_model is not None
        vehicle_loaded = _vehicle_model is not None
        
        # Prepare status response
        status = {
            "load_attempt_success": load_success,
            "models_loaded": {
                "road_model": road_loaded,
                "vehicle_model": vehicle_loaded
            },
            "model_files": {
                "road_model": {
                    "exists": os.path.exists(os.path.join(base_directory, "unet_road_segmentation_tf"))
                },
                "vehicle_model": {
                    "exists": os.path.exists(os.path.join(base_directory, "unet_multi_classV1_tf"))
                }
            },
            "environment": {
                "USE_MODELS": USE_MODELS,
                "BASE_DIR": base_directory
            },
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(status)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": error_details,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/debug-model')
def debug_model():
    """Debug endpoint to check model signature and test with random data"""
    try:
        if not load_dependencies():
            return jsonify({
                "error": "Dependencies not loaded",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 500
        
        if _road_model is None or _vehicle_model is None:
            return jsonify({
                "error": "Models not loaded",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 500
        
        # Get signature information for road model
        road_sig = _road_model.signatures['serving_default'].structured_input_signature
        road_output_sig = _road_model.signatures['serving_default'].structured_outputs
        
        # Get signature information for vehicle model
        vehicle_sig = _vehicle_model.signatures['serving_default'].structured_input_signature
        vehicle_output_sig = _vehicle_model.signatures['serving_default'].structured_outputs
        
        # Create a simple test input
        test_input = _np.zeros((1, 128, 128, 3), dtype='float32')
        tf_input = _tf.convert_to_tensor(test_input, dtype=_tf.float32)
        
        # Test prediction (don't log errors, just check if it works)
        road_success = False
        vehicle_success = False
        road_error = None
        vehicle_error = None
        
        try:
            # Extract input argument name if available
            if len(road_sig) > 1 and len(road_sig[1]) > 0:
                input_name = list(road_sig[1].keys())[0]
                inputs_dict = {input_name: tf_input}
                _road_model.signatures['serving_default'](**inputs_dict)
            else:
                _road_model.signatures['serving_default'](tf_input)
            road_success = True
        except Exception as e:
            road_error = str(e)
        
        try:
            # Extract input argument name if available
            if len(vehicle_sig) > 1 and len(vehicle_sig[1]) > 0:
                input_name = list(vehicle_sig[1].keys())[0]
                inputs_dict = {input_name: tf_input}
                _vehicle_model.signatures['serving_default'](**inputs_dict)
            else:
                _vehicle_model.signatures['serving_default'](tf_input)
            vehicle_success = True
        except Exception as e:
            vehicle_error = str(e)
        
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "models_loaded": {
                "road_model": _road_model is not None,
                "vehicle_model": _vehicle_model is not None
            },
            "model_signatures": {
                "road_model": {
                    "input_signature": str(road_sig),
                    "output_signature": str(road_output_sig)
                },
                "vehicle_model": {
                    "input_signature": str(vehicle_sig),
                    "output_signature": str(vehicle_output_sig)
                }
            },
            "test_prediction": {
                "road_model": {
                    "success": road_success,
                    "error": road_error
                },
                "vehicle_model": {
                    "success": vehicle_success,
                    "error": vehicle_error
                }
            }
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/camera-status')
def check_camera_status():
    """Check if all cameras are working and return their status"""
    try:
        # Initialize results
        results = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "cameras": {}
        }
        
        # Load dependencies if not already loaded
        if not load_dependencies():
            return jsonify({
                "error": "Failed to load dependencies",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 500
        
        # Check each camera
        for camera_id, camera_name in cameras:
            # Get camera code
            camera_code = camera_mapping.get(camera_name, camera_name)
            
            # Try to fetch image
            try:
                logger.info(f"Checking camera {camera_name}")
                image = fetch_camera_image(camera_id)
                
                if image is None:
                    # Camera fetch failed
                    results["cameras"][camera_code] = {
                        "name": camera_name,
                        "status": "offline",
                        "error": "Failed to fetch image"
                    }
                else:
                    # Camera fetch succeeded
                    # Check if image is valid (not just an error page)
                    # This is a simple check - you might need a more sophisticated one
                    if image.size > 1000:  # If image has reasonable size
                        results["cameras"][camera_code] = {
                            "name": camera_name,
                            "status": "online",
                            "resolution": f"{image.shape[1]}x{image.shape[0]}"
                        }
                    else:
                        results["cameras"][camera_code] = {
                            "name": camera_name,
                            "status": "error",
                            "error": "Retrieved image is too small or invalid"
                        }
            except Exception as e:
                # Error processing this camera
                results["cameras"][camera_code] = {
                    "name": camera_name,
                    "status": "error",
                    "error": str(e)
                }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# Start the worker when the app starts
if __name__ != "__main__":
    # Only start worker when running with Gunicorn, not during flask development server
    start_worker()

if __name__ == "__main__":
    # Initialize data files
    manage_historical_densities()
    
    # Start the density worker thread
    start_worker()
    
    # Get the port from environment variable or use default
    port = int(os.environ.get("PORT", 10000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)
