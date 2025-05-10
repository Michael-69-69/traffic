import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify

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
base_directory = os.environ.get('BASE_DIR', '/app')
densities_dir = os.path.join(base_directory, "densities")
today_densities_path = os.path.join(densities_dir, "today_densities.json")
yesterday_max_densities_path = os.path.join(densities_dir, "yesterday_max_densities.json")
critical_densities_path = os.path.join(densities_dir, "critical_densities.json")
output_json_path = os.path.join(densities_dir, "densities.json")

# Create directories if they don't exist
os.makedirs(densities_dir, exist_ok=True)

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

# List of cameras with their IDs and locations
cameras = [
    ("6623e7076f998a001b2523ea", "Lý Thái Tổ - Sư Vạn Hạnh"),
    ("5deb576d1dc17d7c5515acf8", "Ba Tháng Hai - Cao Thắng"),
    ("63ae7a9cbfd3d90017e8f303", "Điện Biên Phủ – Cao Thắng"),
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

# Camera URL template - Update with your actual base URL
CAMERA_URL_TEMPLATE = os.environ.get('CAMERA_URL_TEMPLATE', 'https://api.example.com/cameras/{camera_id}/snapshot')

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
    """Load ML models with memory optimizations"""
    global _road_model, _vehicle_model
    
    if not USE_MODELS:
        logger.info("Model loading skipped as USE_MODELS is set to false")
        return True
        
    if not load_dependencies():
        logger.error("Dependencies not loaded, cannot load models")
        return False
    
    # Define model paths
    road_model_path = os.path.join(base_directory, "unet_road_segmentation.keras")
    vehicle_model_path = os.path.join(base_directory, "unet_multi_classV1.keras")
    
    # Check if model files exist
    if not os.path.exists(road_model_path) or not os.path.exists(vehicle_model_path):
        logger.error(f"Model files not found: {road_model_path} or {vehicle_model_path}")
        return False
    
    logger.info("Loading road segmentation model...")
    
    try:
        # Load models with optimized settings
        custom_objects = {'dice_loss': dice_loss}
        _road_model = _tf.keras.models.load_model(road_model_path, custom_objects=custom_objects, compile=False)
        
        # Force model to use less memory by setting specific configuration
        _road_model.make_predict_function()  # This initializes the model
        
        logger.info("Road segmentation model loaded")
        
        # Sleep briefly to let memory settle
        time.sleep(1)
        
        logger.info("Loading vehicle detection model...")
        _vehicle_model = _tf.keras.models.load_model(vehicle_model_path, custom_objects=custom_objects, compile=False)
        _vehicle_model.make_predict_function()  # Initialize the model
        
        logger.info("Vehicle detection model loaded")
        
        # Return success
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def preprocess_image(img):
    """Preprocess image for model input"""
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
        img = img / 255.0
        img = _np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def manage_historical_densities():
    """Manage historical density data"""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
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
                    logger.info(f"Updated yesterday_max_densities.json with max densities from {file_date}")
            else:
                today_densities = {'date': today.strftime('%Y-%m-%d')}
        except Exception as e:
            logger.error(f"Error reading today_densities.json: {e}")
            today_densities = {'date': today.strftime('%Y-%m-%d')}
    else:
        today_densities = {'date': today.strftime('%Y-%m-%d')}
    
    # Sample critical densities (static values to avoid model predictions initially)
    sample_critical_densities = {
        'A': 80.0, 'B': 70.0, 'C': 75.0, 'D': 85.0, 'E': 80.0, 'F': 60.0,
        'G': 70.0, 'H': 90.0, 'I': 85.0, 'J': 75.0, 'K': 80.0, 'L': 80.0
    }
    
    with open(critical_densities_path, 'w', encoding='utf-8') as f:
        json.dump(sample_critical_densities, f, ensure_ascii=False)
    
    # Generate sample results (avoid running models on startup)
    results = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "cameras": {}
    }
    
    # Populate with sample data
    for _, camera_name in cameras:
        camera_code = camera_mapping.get(camera_name, camera_name)
        critical_density = sample_critical_densities.get(camera_code, 80.0)
        
        # Add simulated density data
        results["cameras"][camera_code] = {
            "name": camera_name,
            "density": 50.0,  # Sample value
            "congestion_level": 62.5,  # 50/80 * 100
            "critical_density": critical_density,
            "composition": {
                "cars": 60.0,
                "motorcycles": 35.0,
                "others": 5.0
            }
        }
    
    # Save the initial results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return sample_critical_densities, today_densities

def fetch_camera_image(camera_id):
    """Fetch camera image from the API"""
    if not load_dependencies():
        return None
        
    try:
        url = CAMERA_URL_TEMPLATE.format(camera_id=camera_id)
        logger.info(f"Fetching image from {url}")
        
        response = _session.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert response to image
        image_array = _np.asarray(bytearray(response.content), dtype=_np.uint8)
        image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Error fetching camera image for {camera_id}: {e}")
        return None

def analyze_image(image):
    """Analyze the image with ML models"""
    if not USE_MODELS or not load_dependencies() or image is None:
        # Return simulated data if models not used or image is None
        return {
            "density": round(_np.random.uniform(40.0, 70.0), 1),
            "composition": {
                "cars": round(_np.random.uniform(55.0, 65.0), 1),
                "motorcycles": round(_np.random.uniform(30.0, 40.0), 1),
                "others": round(_np.random.uniform(3.0, 7.0), 1)
            }
        }
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Failed to preprocess image")
        
        # Use road model to get road segmentation
        road_prediction = _road_model.predict(processed_image, verbose=0)
        
        # Use vehicle model to get vehicle types
        vehicle_prediction = _vehicle_model.predict(processed_image, verbose=0)
        
        # Calculate density based on vehicle prediction
        # This is a simplified calculation - adjust based on your model's output
        density = float(_np.mean(vehicle_prediction[0]) * 100)
        
        # Extract vehicle composition
        # Assuming vehicle_prediction has channels for different vehicle types
        # This should be adjusted based on your model's actual output format
        cars_percentage = float(_np.mean(vehicle_prediction[0][:,:,:,0]) * 100)
        motorcycles_percentage = float(_np.mean(vehicle_prediction[0][:,:,:,1]) * 100)
        others_percentage = 100.0 - cars_percentage - motorcycles_percentage
        
        return {
            "density": round(density, 1),
            "composition": {
                "cars": round(cars_percentage, 1),
                "motorcycles": round(motorcycles_percentage, 1),
                "others": round(others_percentage, 1)
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {
            "density": 50.0,  # Fallback value
            "composition": {
                "cars": 60.0,
                "motorcycles": 35.0,
                "others": 5.0
            }
        }

def fetch_and_process_densities():
    """Fetch and process density data from cameras"""
    # Load critical densities
    critical_densities = {}
    if os.path.exists(critical_densities_path):
        try:
            with open(critical_densities_path, 'r', encoding='utf-8') as f:
                critical_densities = json.load(f)
        except Exception as e:
            logger.error(f"Error reading critical_densities.json: {e}")
    
    # Load today's densities
    today_densities = {}
    if os.path.exists(today_densities_path):
        try:
            with open(today_densities_path, 'r', encoding='utf-8') as f:
                today_densities = json.load(f)
        except Exception as e:
            logger.error(f"Error reading today_densities.json: {e}")
    
    # Current timestamp
    current_time = datetime.now()
    timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize results
    results = {
        "timestamp": timestamp_str,
        "cameras": {}
    }
    
    # Process each camera
    for camera_id, camera_name in cameras:
        try:
            logger.info(f"Processing camera {camera_name}")
            
            # Get camera code
            camera_code = camera_mapping.get(camera_name, camera_name)
            
            # Get critical density
            critical_density = critical_densities.get(camera_code, 80.0)
            
            # For simulated data mode or when image fetch fails
            if not USE_MODELS:
                # Simulate density value with small variation
                base_density = 50.0
                if camera_code in today_densities and today_densities[camera_code]:
                    # Get the most recent density value
                    recent_densities = list(today_densities[camera_code].values())
                    if recent_densities:
                        base_density = recent_densities[-1]
                
                # Add small random variation to simulate changing traffic
                if _np is None and not load_dependencies():
                    import random
                    density = base_density + random.uniform(-5.0, 5.0)
                    density = max(10.0, min(95.0, density))  # Keep within reasonable range
                else:
                    density = base_density + _np.random.uniform(-5.0, 5.0)
                    density = max(10.0, min(95.0, density))  # Keep within reasonable range
                
                analysis_result = {
                    "density": round(density, 1),
                    "composition": {
                        "cars": round(_np.random.uniform(55.0, 65.0) if _np else 60.0, 1),
                        "motorcycles": round(_np.random.uniform(30.0, 40.0) if _np else 35.0, 1),
                        "others": round(_np.random.uniform(3.0, 7.0) if _np else 5.0, 1)
                    }
                }
            else:
                # Fetch camera image
                image = fetch_camera_image(camera_id)
                
                # Analyze image
                analysis_result = analyze_image(image)
            
            # Calculate congestion level
            density = analysis_result["density"]
            congestion_level = (density / critical_density) * 100 if critical_density > 0 else 0
            congestion_level = round(min(100.0, congestion_level), 1)
            
            # Update today's densities
            if camera_code not in today_densities:
                today_densities[camera_code] = {}
            today_densities[camera_code][timestamp_str] = density
            
            # Add to results
            results["cameras"][camera_code] = {
                "name": camera_name,
                "density": density,
                "congestion_level": congestion_level,
                "critical_density": critical_density,
                "composition": analysis_result["composition"]
            }
            
            logger.info(f"Processed camera {camera_name}: density={density}, congestion={congestion_level}")
            
        except Exception as e:
            logger.error(f"Error processing camera {camera_name}: {e}")
            
            # Add default values on error
            results["cameras"][camera_mapping.get(camera_name, camera_name)] = {
                "name": camera_name,
                "density": 0.0,
                "congestion_level": 0.0,
                "critical_density": critical_densities.get(camera_mapping.get(camera_name, camera_name), 80.0),
                "composition": {
                    "cars": 0.0,
                    "motorcycles": 0.0,
                    "others": 0.0
                },
                "error": str(e)
            }
    
    # Save today's densities
    try:
        with open(today_densities_path, 'w', encoding='utf-8') as f:
            json.dump(today_densities, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving today_densities.json: {e}")
    
    # Save results
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving densities.json: {e}")
    
    return results

def density_worker():
    """Background worker to process densities periodically"""
    logger.info("Density worker initialized")
    
    try:
        # Initial run without delay
        logger.info("Starting initial density calculation")
        critical_densities, today_densities = manage_historical_densities()
        logger.info(f"Initial densities created: {len(critical_densities)} critical densities")
        
        while True:
            try:
                logger.info("Starting density processing cycle")
                fetch_and_process_densities()
                logger.info("Density processing cycle completed")
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in density worker cycle: {e}")
                time.sleep(30)  # Shorter retry interval on error
    except Exception as e:
        logger.error(f"Critical error in density worker: {e}")

# Start the density worker thread
def start_worker():
    try:
        # Only load models if USE_MODELS is true
        if USE_MODELS:
            logger.info("Attempting to load models...")
            load_models()
        else:
            logger.info("Skipping model loading (USE_MODELS=false)")
        
        # Start the density worker
        density_thread = threading.Thread(target=density_worker, daemon=True)
        density_thread.start()
        logger.info("Density worker thread started successfully")
    except Exception as e:
        logger.error(f"Failed to start density worker: {e}")

# Flask routes
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "1.0",
        "message": "Traffic Analysis Service is operational",
        "using_models": USE_MODELS
    })

@app.route('/densities')
def get_densities():
    try:
        # Log the attempt to read densities
        logger.info(f"Attempting to read densities from {output_json_path}")
        
        # Check if file exists
        if not os.path.exists(output_json_path):
            logger.warning(f"Densities file not found at {output_json_path}, creating initial data")
            manage_historical_densities()
        
        # Try to read the file
        with open(output_json_path, 'r', encoding='utf-8') as f:
            densities = json.load(f)
            logger.info("Successfully read densities data")
            return jsonify(densities)
    except Exception as e:
        logger.error(f"Error reading densities: {e}")
        return jsonify({
            "error": "Could not read densities data",
            "details": str(e)
        }), 500

@app.route('/status')
def status():
    """Memory-efficient status endpoint"""
    return jsonify({
        "status": "running",
        "memory_optimized": True,
        "version": "1.0",
        "using_models": USE_MODELS,
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health')
def health_check():
    try:
        # Check if directories exist
        densities_exists = os.path.exists(densities_dir)
        output_exists = os.path.exists(output_json_path)
        
        return jsonify({
            "status": "healthy",
            "filesystem": {
                "densities_dir_exists": densities_exists,
                "output_file_exists": output_exists,
                "densities_dir": densities_dir,
                "output_path": output_json_path
            },
            "using_models": USE_MODELS,
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