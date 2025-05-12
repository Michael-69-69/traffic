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
    """Fetch camera image from the API with enhanced error handling"""
    if not load_dependencies():
        return None
        
    try:
        url = CAMERA_URL_TEMPLATE.format(camera_id=camera_id)
        logger.info(f"Fetching image from {url}")
        
        response = _session.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
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
        else:
            logger.error(f"HTTP error fetching camera image for {camera_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching camera image for {camera_id}: {e}")
        return None

def analyze_image(image):
    """Analyze the image with ML models - returns only density value"""
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
        
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {
                "density": 0.0
            }
        
        # Use models with the correct signature for SavedModel format
        # SavedModel format uses the 'serving_default' signature
        road_prediction = _road_model.signatures['serving_default'](_tf.constant(processed_image))
        
        vehicle_prediction = _vehicle_model.signatures['serving_default'](_tf.constant(processed_image))
        vehicle_output = list(vehicle_prediction.values())[0]
        
        # Extract value for density - simplified to return raw output
        # Convert to numpy array for easier manipulation
        vehicle_output_np = vehicle_output.numpy()
        density = float(_np.mean(vehicle_output_np) * 100)
        
        # Just return the density value
        return {
            "density": round(density, 1)
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "density": 0.0
        }

def fetch_and_process_densities():
    """Fetch and process density data from cameras - simplified version"""
    # Current timestamp
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
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
            
            # For simulated data mode or when image fetch fails
            if not USE_MODELS:
                # Simulate density value
                density = _np.random.uniform(0.0, 100.0) if _np else 50.0
            else:
                # Fetch camera image
                image = fetch_camera_image(camera_id)
                
                # Analyze image
                analysis_result = analyze_image(image)
                
                # Get density value
                density = analysis_result["density"]
            
            # Add to results - just the density value and name for internal reference
            results["cameras"][camera_code] = {
                "name": camera_name,
                "density": density
            }
            
            logger.info(f"Processed camera {camera_name}: density={density}")
            
        except Exception as e:
            logger.error(f"Error processing camera {camera_name}: {e}")
            
            # Add default values on error
            results["cameras"][camera_mapping.get(camera_name, camera_name)] = {
                "name": camera_name,
                "density": 0.0
            }
    
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
                time.sleep(300)  # Increase to 5 minutes to avoid rate limiting
            except Exception as e:
                logger.error(f"Error in density worker cycle: {e}")
                time.sleep(30)  # Shorter retry interval on error
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
            elif not os.path.exists(os.path.join(base_directory, "unet_road_segmentation.keras")):
                logger.error("Problem: Road model file not found")
            elif not os.path.exists(os.path.join(base_directory, "unet_multi_classV1.keras")):
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
        "using_models": USE_MODELS
    })

@app.route('/densities')
def get_densities():
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


@app.route('/debug')
def debug():
    """Debug endpoint to check environment variables and model files"""
    try:
        # Check for model files
        model_info = {
            "unet_road_segmentation.keras": {
                "exists": os.path.exists(os.path.join(base_directory, "unet_road_segmentation.keras")),
                "size_mb": round(os.path.getsize(os.path.join(base_directory, "unet_road_segmentation.keras")) / (1024 * 1024), 2) if os.path.exists(os.path.join(base_directory, "unet_road_segmentation.keras")) else None,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(os.path.join(base_directory, "unet_road_segmentation.keras"))).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(os.path.join(base_directory, "unet_road_segmentation.keras")) else None,
                "tf_converted_exists": os.path.exists(os.path.join(base_directory, "unet_road_segmentation.keras_tf"))
            },
            "unet_multi_classV1.keras": {
                "exists": os.path.exists(os.path.join(base_directory, "unet_multi_classV1.keras")),
                "size_mb": round(os.path.getsize(os.path.join(base_directory, "unet_multi_classV1.keras")) / (1024 * 1024), 2) if os.path.exists(os.path.join(base_directory, "unet_multi_classV1.keras")) else None,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(os.path.join(base_directory, "unet_multi_classV1.keras"))).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(os.path.join(base_directory, "unet_multi_classV1.keras")) else None,
                "tf_converted_exists": os.path.exists(os.path.join(base_directory, "unet_multi_classV1.keras_tf"))
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
            "yesterday_max_densities_exists": os.path.exists(yesterday_max_densities_path),
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
        import psutil
        try:
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
            "system_resources": memory_info
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
                    "exists": os.path.exists(os.path.join(base_directory, "unet_road_segmentation.keras"))
                },
                "vehicle_model": {
                    "exists": os.path.exists(os.path.join(base_directory, "unet_multi_classV1.keras"))
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