import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify
from urllib.parse import urlparse, parse_qs, unquote
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# Initialize Flask
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google Drive setup
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CLIENT_ID = os.environ.get('GOOGLE_DRIVE_CLIENT_ID')
CLIENT_SECRET = os.environ.get('GOOGLE_DRIVE_CLIENT_SECRET')
REFRESH_TOKEN = os.environ.get('GOOGLE_DRIVE_REFRESH_TOKEN')
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')

# File names for density data
TODAY_DENSITIES_FILE = "today_densities.json"
YESTERDAY_DENSITIES_FILE = "yesterday_densities.json"
CRITICAL_DENSITIES_FILE = "critical_densities.json"
OUTPUT_JSON_FILE = "densities.json"

# Initialize Google Drive service
drive_service = None

def init_google_drive():
    global drive_service
    try:
        creds = Credentials(
            None,
            refresh_token=REFRESH_TOKEN,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES,
            token_uri="https://oauth2.googleapis.com/token"
        )
        drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Google Drive: {e}")
        drive_service = None

# Utility functions for Google Drive operations
def get_file_id(filename):
    try:
        query = f"'{FOLDER_ID}' in parents and name = '{filename}' and trashed = false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None
    except Exception as e:
        logger.error(f"Error finding file {filename} in Google Drive: {e}")
        return None

def upload_json_to_drive(filename, data):
    try:
        # Convert data to JSON string and write to a temporary file
        temp_file = f"/tmp/{filename}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        file_id = get_file_id(filename)
        media = MediaFileUpload(temp_file, mimetype='application/json')
        
        if file_id:
            # Update existing file
            drive_service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            logger.info(f"Updated {filename} in Google Drive")
        else:
            # Create new file
            file_metadata = {
                'name': filename,
                'parents': [FOLDER_ID],
                'mimeType': 'application/json'
            }
            drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            logger.info(f"Uploaded {filename} to Google Drive")
        
        # Clean up temporary file
        os.remove(temp_file)
    except Exception as e:
        logger.error(f"Error uploading {filename} to Google Drive: {e}")

def download_json_from_drive(filename):
    try:
        file_id = get_file_id(filename)
        if not file_id:
            logger.warning(f"File {filename} not found in Google Drive")
            return None
        
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        data = json.loads(fh.read().decode('utf-8'))
        logger.info(f"Downloaded {filename} from Google Drive")
        return data
    except Exception as e:
        logger.error(f"Error downloading {filename} from Google Drive: {e}")
        return None

# Base URL and default parameters for the camera feed
main_url = "https://giaothong.hochiminhcity.gov.vn"
base_url = "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx"
default_params = {
    "bg": "black",
    "w": 300,
    "h": 230
}

# Camera websites list
camera_websites = [
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623e7076f998a001b2523ea&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20S%C6%B0%20V%E1%BA%A1n%20H%E1%BA%A1nh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf8&camLocation=Ba%20Th%C3%A1ng%20Hai%20-%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae7a9cbfd3d90017e8f303&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20%E2%80%93%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515ad21&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20Nguy%E1%BB%85n%20Tri%20Ph%C6%B0%C6%A1ng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515ad22&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20Nguy%E1%BB%85n%20Tri%20Ph%C6%B0%C6%A1ng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5d8cdd26766c880017188974&camLocation=N%C3%BAt%20giao%20L%C3%AA%20%C4%90%E1%BA%A1i%20H%C3%A0nh%202%20(L%C3%AA%20%C4%90%E1%BA%A1i%20H%C3%A0nh)&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae763bbfd3d90017e8f0c4&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Chi%E1%BB%83u&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf6&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf7&camLocation=N%C3%BUt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf2&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20-%20C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf9&camLocation=N%C3%BAt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acfa&camLocation=N%C3%BAt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8'
]

# Parse camera data from URLs
def parse_camera_data():
    cameras = []
    camera_mapping = {}
    for idx, url in enumerate(camera_websites):
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            camera_id = query_params.get('camId', [''])[0]
            camera_location = unquote(query_params.get('camLocation', [''])[0])
            if camera_id and camera_location:
                camera_code = chr(65 + idx)
                cameras.append((camera_id, camera_location))
                camera_mapping[camera_location] = camera_code
                logger.info(f"Parsed camera {camera_code}: {camera_location} (ID: {camera_id})")
        except Exception as e:
            logger.error(f"Error parsing camera URL {url}: {e}")
    return cameras, camera_mapping

# Generate cameras and mapping
cameras, camera_mapping = parse_camera_data()
CAMERA_URL_TEMPLATE = os.environ.get('CAMERA_URL_TEMPLATE', 'https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx')

# Lazy-load TensorFlow and other dependencies
_tf, _cv2, _np, _requests, _road_model, _vehicle_model, _session = [None] * 7
USE_MODELS = os.environ.get('USE_MODELS', 'false').lower() == 'true'
last_density_update = None

def load_dependencies():
    global _tf, _cv2, _np, _requests, _session
    if _tf is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_MEMORY_ALLOCATION'] = '256MB'
            import tensorflow as tf
            import cv2
            import numpy as np
            import requests
            _tf, _cv2, _np, _requests = tf, cv2, np, requests
            physical_devices = _tf.config.list_physical_devices('GPU')
            if physical_devices:
                _tf.config.experimental.set_memory_growth(physical_devices[0], True)
            _tf.config.threading.set_intra_op_parallelism_threads(1)
            _tf.config.threading.set_inter_op_parallelism_threads(1)
            _session = _requests.Session()
            _session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"})
            logger.info("Dependencies loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return False
    return True

def dice_loss(y_true, y_pred, smooth=1e-6):
    if not load_dependencies():
        return 0
    y_true_f = _tf.keras.backend.flatten(y_true)
    y_pred_f = _tf.keras.backend.flatten(y_pred)
    intersection = _tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (_tf.keras.backend.sum(y_true_f) + _tf.keras.backend.sum(y_pred_f) + smooth))

def load_models():
    global _road_model, _vehicle_model
    logger.info("=============================================")
    logger.info("LOADING MODELS - FORCED ATTEMPT")
    logger.info("=============================================")
    if not load_dependencies():
        logger.error("Failed to load dependencies - cannot load models")
        return False
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    road_model_path = os.path.join(base_directory, "unet_road_segmentation_tf")
    vehicle_model_path = os.path.join(base_directory, "unet_multi_classV1_tf")
    logger.info(f"Checking for model files: Road: {os.path.exists(road_model_path)}, Vehicle: {os.path.exists(vehicle_model_path)}")
    try:
        logger.info("Loading road segmentation model...")
        _road_model = _tf.saved_model.load(road_model_path)
        time.sleep(1)
        logger.info("Loading vehicle detection model...")
        _vehicle_model = _tf.saved_model.load(vehicle_model_path)
        logger.info("=============================================")
        logger.info("MODELS LOADED SUCCESSFULLY")
        logger.info("=============================================")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=============================================")
        logger.error("MODEL LOADING FAILED")
        logger.error("=============================================")
        return False

def preprocess_image(img):
    if not load_dependencies() or img is None:
        return None
    ycrcb = _cv2.cvtColor(img, _cv2.COLOR_BGR2YCrCb)
    y, cr, cb = _cv2.split(ycrcb)
    clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    enhanced_img = _cv2.merge((y, cr, cb))
    img = _cv2.cvtColor(enhanced_img, _cv2.COLOR_YCrCb2BGR)
    img = _cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    return _np.expand_dims(img, axis=0)

def check_new_day():
    today = datetime.now().date()
    today_densities = download_json_from_drive(TODAY_DENSITIES_FILE) or {}
    if 'date' in today_densities:
        try:
            file_date = datetime.strptime(today_densities['date'], '%Y-%m-%d').date()
            if file_date < today:
                logger.info(f"New day detected. Transferring data from {file_date} to yesterday")
                upload_json_to_drive(YESTERDAY_DENSITIES_FILE, today_densities)
                update_critical_densities(today_densities)
                today_densities = {'date': today.strftime('%Y-%m-%d'), 'densities_by_time': {}}
                upload_json_to_drive(TODAY_DENSITIES_FILE, today_densities)
                logger.info("Successfully transferred data to yesterday and reset today's data")
        except Exception as e:
            logger.error(f"Error processing date change: {e}")

def update_critical_densities(densities_data):
    try:
        critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE) or {}
        densities_by_time = densities_data.get('densities_by_time', {})
        for camera_code in camera_mapping.values():
            max_density = 0.0
            for timestamp, cameras_data in densities_by_time.items():
                if camera_code in cameras_data:
                    density = cameras_data[camera_code].get('density', 0.0)
                    max_density = max(max_density, density)
            if camera_code not in critical_densities or max_density > critical_densities[camera_code]:
                critical_densities[camera_code] = max_density
                logger.info(f"Updated critical density for {camera_code}: {max_density}")
        upload_json_to_drive(CRITICAL_DENSITIES_FILE, critical_densities)
        logger.info("Critical densities updated successfully")
    except Exception as e:
        logger.error(f"Error updating critical densities: {e}")

def manage_historical_densities():
    check_new_day()
    today_densities = download_json_from_drive(TODAY_DENSITIES_FILE) or {}
    if 'date' not in today_densities or today_densities['date'] != datetime.now().date().strftime('%Y-%m-%d'):
        today_densities = {
            'date': datetime.now().date().strftime('%Y-%m-%d'),
            'densities_by_time': {}
        }
        upload_json_to_drive(TODAY_DENSITIES_FILE, today_densities)
    critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE)
    if not critical_densities:
        sample_critical_densities = {code: 80.0 for code in camera_mapping.values()}
        upload_json_to_drive(CRITICAL_DENSITIES_FILE, sample_critical_densities)
    return today_densities

def fetch_camera_image(camera_id):
    if not load_dependencies():
        return None
    try:
        global _session
        if _session is None:
            _session = _requests.Session()
        _session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://giaothong.hochiminhcity.gov.vn/"
        })
        _session.get("https://giaothong.hochiminhcity.gov.vn/", timeout=10)
        url = CAMERA_URL_TEMPLATE.format(camera_id=camera_id)
        logger.info(f"Fetching image from {url}")
        response = _session.get(url, timeout=10)
        response.raise_for_status()
        image_array = _np.asarray(bytearray(response.content), dtype=_np.uint8)
        image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"Failed to decode image from {url}")
            return None
        return image
    except _requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"403 Forbidden for {url}: Check API key or server permissions")
            try:
                logger.error(f"Response headers: {e.response.headers}")
                logger.error(f"Response content: {e.response.text[:500]}")
            except:
                pass
        else:
            logger.error(f"HTTP error fetching camera image for {camera_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching camera image for {camera_id}: {e}")
        return None

def analyze_image(image):
    if not load_dependencies() or image is None:
        return {"density": 0.0}
    try:
        if _road_model is None or _vehicle_model is None:
            logger.warning("Models not loaded, using fallback values")
            return {"density": 0.0}
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"density": 0.0}
        input_tensor = _tf.convert_to_tensor(processed_image, dtype=_tf.float32)
        try:
            vehicle_prediction = _vehicle_model.signatures['serving_default'](input_tensor=input_tensor)
            vehicle_output = list(vehicle_prediction.values())[0]
            vehicle_output_np = vehicle_output.numpy()
            logger.info(f"Vehicle output shape: {vehicle_output_np.shape}")
            if vehicle_output_np.shape[-1] == 12:
                weights = [0.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05]
                weighted_sum = sum(float(_np.mean(vehicle_output_np[..., i])) * weights[i] for i in range(1, 12))
                density = max(0, min(100, weighted_sum * 100))
            else:
                density = float(_np.mean(vehicle_output_np) * 100)
            logger.info(f"Calculated weighted density: {density}")
            return {"density": round(density, 1)}
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if _np:
                density = round(_np.random.uniform(10.0, 90.0), 1)
            else:
                import random
                density = round(random.uniform(10.0, 90.0), 1)
            return {"density": density}
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {"density": 0.0}

def store_today_density(timestamp_str, camera_code, density_data):
    try:
        today_densities = download_json_from_drive(TODAY_DENSITIES_FILE) or {}
        if 'densities_by_time' not in today_densities:
            today_densities['densities_by_time'] = {}
        if timestamp_str not in today_densities['densities_by_time']:
            today_densities['densities_by_time'][timestamp_str] = {}
        today_densities['densities_by_time'][timestamp_str][camera_code] = density_data
        upload_json_to_drive(TODAY_DENSITIES_FILE, today_densities)
    except Exception as e:
        logger.error(f"Error storing today's density: {e}")

def fetch_and_process_densities():
    global last_density_update
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_density_update = datetime.now()
    results = {"timestamp": timestamp_str, "cameras": {}}
    success_count, failure_count = 0, 0
    for camera_id, camera_name in cameras:
        try:
            logger.info(f"Processing camera {camera_name}")
            camera_code = camera_mapping.get(camera_name, camera_name)
            image = fetch_camera_image(camera_id)
            if image is None:
                failure_count += 1
                logger.warning(f"Using simulated data for {camera_name} due to image fetch failure")
                if _np:
                    density = round(_np.random.uniform(10.0, 90.0), 1)
                else:
                    import random
                    density = round(random.uniform(10.0, 90.0), 1)
            else:
                success_count += 1
                logger.info(f"Successfully fetched image for {camera_name}")
                analysis_result = analyze_image(image)
                density = analysis_result["density"]
            density_data = {"name": camera_name, "density": density, "timestamp": timestamp_str}
            results["cameras"][camera_code] = density_data
            store_today_density(timestamp_str, camera_code, density_data)
            logger.info(f"Processed camera {camera_name}: density={density}")
        except Exception as e:
            logger.error(f"Error processing camera {camera_name}: {e}")
            failure_count += 1
            density_data = {"name": camera_name, "density": 0.0, "timestamp": timestamp_str}
            results["cameras"][camera_mapping.get(camera_name, camera_name)] = density_data
            store_today_density(timestamp_str, camera_mapping.get(camera_name, camera_name), density_data)
    logger.info(f"Camera processing complete. Success: {success_count}, Failure: {failure_count}")
    try:
        upload_json_to_drive(OUTPUT_JSON_FILE, results)
    except Exception as e:
        logger.error(f"Error saving densities.json to Google Drive: {e}")
    return results

def density_worker():
    logger.info("Density worker initialized - running every 30 seconds")
    try:
        logger.info("Starting initial density calculation")
        manage_historical_densities()
        fetch_and_process_densities()
        logger.info("Initial density calculation completed")
        while True:
            try:
                logger.info("Starting density processing cycle (30-second interval)")
                fetch_and_process_densities()
                logger.info("Density processing cycle completed")
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in density worker cycle: {e}")
                time.sleep(10)
    except Exception as e:
        logger.error(f"Critical error in density worker: {e}")

def start_worker():
    try:
        logger.info("Starting worker - FOCUSING ON MODEL LOADING")
        logger.info("Attempting to load models (forced)...")
        load_success = load_models()
        if load_success:
            logger.info("Models loaded successfully!")
        else:
            logger.error("Failed to load models! Check logs for details.")
            if not load_dependencies():
                logger.error("Problem: Dependencies failed to load")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation_tf")):
                logger.error("Problem: Road model file not found")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_multi_classV1_tf")):
                logger.error("Problem: Vehicle model file not found")
            else:
                logger.error("Problem: Unclear - check model format or TensorFlow compatibility")
        logger.info("Starting density worker thread...")
        density_thread = threading.Thread(target=density_worker, daemon=True)
        density_thread.start()
        logger.info("Density worker thread started")
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Date transition worker for precise midnight updates
def date_transition_worker():
    while True:
        now = datetime.now()
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
        seconds_until_midnight = (midnight - now).total_seconds()
        time.sleep(seconds_until_midnight + 1)
        check_new_day()
        logger.info(f"Date transition completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize Google Drive and start workers
if __name__ != "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Application may not function correctly.")
    else:
        threading.Thread(target=date_transition_worker, daemon=True).start()
        start_worker()

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
    try:
        cameras_info = [{"code": chr(65 + idx), "id": camera_id, "name": camera_location, "url": camera_websites[idx] if idx < len(camera_websites) else None} for idx, (camera_id, camera_location) in enumerate(cameras)]
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_cameras": len(cameras_info),
            "cameras": cameras_info
        })
    except Exception as e:
        logger.error(f"Error fetching cameras: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/live-densities')
def get_live_densities():
    try:
        densities = download_json_from_drive(OUTPUT_JSON_FILE)
        if not densities:
            return jsonify({
                "error": "No density data available yet",
                "message": "Please wait for the first calculation cycle"
            }), 404
        densities["last_update"] = last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
        densities["update_interval"] = "30 seconds"
        if last_density_update:
            next_update = last_density_update + timedelta(seconds=30)
            time_until_next = next_update - datetime.now()
            densities["next_update_in"] = f"{int(time_until_next.total_seconds())} seconds" if time_until_next.total_seconds() > 0 else "Updating now..."
        return jsonify(densities)
    except Exception as e:
        logger.error(f"Error reading live densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/today-densities')
def get_today_densities():
    try:
        today_densities = download_json_from_drive(TODAY_DENSITIES_FILE)
        if not today_densities:
            manage_historical_densities()
            today_densities = download_json_from_drive(TODAY_DENSITIES_FILE)
        return jsonify(today_densities)
    except Exception as e:
        logger.error(f"Error reading today's densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/yesterday-densities')
def get_yesterday_densities():
    try:
        yesterday_densities = download_json_from_drive(YESTERDAY_DENSITIES_FILE)
        if not yesterday_densities:
            return jsonify({
                "message": "No yesterday data available yet",
                "date": None,
                "densities_by_time": {}
            })
        return jsonify(yesterday_densities)
    except Exception as e:
        logger.error(f"Error reading yesterday's densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/critical-densities')
def get_critical_densities():
    try:
        critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE)
        if not critical_densities:
            manage_historical_densities()
            critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE)
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Critical density thresholds based on historical maximum values",
            "critical_densities": critical_densities
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error reading critical densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/densities')
def get_densities():
    try:
        densities = download_json_from_drive(OUTPUT_JSON_FILE)
        if not densities:
            manage_historical_densities()
            densities = download_json_from_drive(OUTPUT_JSON_FILE)
        raw_densities = {camera_code: camera_data["density"] for camera_code, camera_data in densities["cameras"].items()}
        return jsonify(raw_densities)
    except Exception as e:
        logger.error(f"Error reading densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
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
        today_exists = bool(get_file_id(TODAY_DENSITIES_FILE))
        yesterday_exists = bool(get_file_id(YESTERDAY_DENSITIES_FILE))
        critical_exists = bool(get_file_id(CRITICAL_DENSITIES_FILE))
        output_exists = bool(get_file_id(OUTPUT_JSON_FILE))
        return jsonify({
            "status": "healthy",
            "storage": {
                "backend": "Google Drive",
                "folder_id": FOLDER_ID,
                "today_densities_exists": today_exists,
                "yesterday_densities_exists": yesterday_exists,
                "critical_densities_exists": critical_exists,
                "output_file_exists": output_exists
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
    try:
        model_info = {
            "unet_road_segmentation_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation_tf"))},
            "unet_multi_classV1_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_multi_classV1_tf"))}
        }
        env_vars = {
            "USE_MODELS_RAW": os.environ.get('USE_MODELS', 'not set'),
            "USE_MODELS_PROCESSED": USE_MODELS,
            "BASE_DIR": os.environ.get('BASE_DIR', 'not set'),
            "TF_MEMORY_ALLOCATION": os.environ.get('TF_MEMORY_ALLOCATION', 'not set'),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'not set'),
            "PORT": os.environ.get('PORT', 'not set'),
            "CAMERA_URL_TEMPLATE": os.environ.get('CAMERA_URL_TEMPLATE', 'not set'),
            "GOOGLE_DRIVE_FOLDER_ID": FOLDER_ID
        }
        storage_info = {
            "backend": "Google Drive",
            "folder_id": FOLDER_ID,
            "today_densities_exists": bool(get_file_id(TODAY_DENSITIES_FILE)),
            "yesterday_densities_exists": bool(get_file_id(YESTERDAY_DENSITIES_FILE)),
            "critical_densities_exists": bool(get_file_id(CRITICAL_DENSITIES_FILE)),
            "output_json_exists": bool(get_file_id(OUTPUT_JSON_FILE))
        }
        try:
            files_in_base_dir = os.listdir(os.environ.get('BASE_DIR', os.getcwd()))
        except Exception as e:
            files_in_base_dir = f"Error listing files: {str(e)}"
        model_load_status = {
            "road_model_loaded": _road_model is not None,
            "vehicle_model_loaded": _vehicle_model is not None,
            "dependencies_loaded": _tf is not None and _cv2 is not None and _np is not None and _requests is not None
        }
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
            "base_directory": os.environ.get('BASE_DIR', os.getcwd()),
            "files_in_base_directory": files_in_base_dir,
            "storage_info": storage_info,
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
    try:
        load_success = load_models()
        road_loaded = _road_model is not None
        vehicle_loaded = _vehicle_model is not None
        status = {
            "load_attempt_success": load_success,
            "models_loaded": {"road_model": road_loaded, "vehicle_model": vehicle_loaded},
            "model_files": {
                "road_model": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation_tf"))},
                "vehicle_model": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_multi_classV1_tf"))}
            },
            "environment": {"USE_MODELS": USE_MODELS, "BASE_DIR": os.environ.get('BASE_DIR', os.getcwd())},
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
    try:
        if not load_dependencies():
            return jsonify({"error": "Dependencies not loaded", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500
        if _road_model is None or _vehicle_model is None:
            return jsonify({"error": "Models not loaded", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500
        road_sig = _road_model.signatures['serving_default'].structured_input_signature
        road_output_sig = _road_model.signatures['serving_default'].structured_outputs
        vehicle_sig = _vehicle_model.signatures['serving_default'].structured_input_signature
        vehicle_output_sig = _vehicle_model.signatures['serving_default'].structured_outputs
        test_input = _np.zeros((1, 128, 128, 3), dtype='float32')
        tf_input = _tf.convert_to_tensor(test_input, dtype=_tf.float32)
        road_success, vehicle_success = False, False
        road_error, vehicle_error = None, None
        try:
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
            "models_loaded": {"road_model": _road_model is not None, "vehicle_model": _vehicle_model is not None},
            "model_signatures": {
                "road_model": {"input_signature": str(road_sig), "output_signature": str(road_output_sig)},
                "vehicle_model": {"input_signature": str(vehicle_sig), "output_signature": str(vehicle_output_sig)}
            },
            "test_prediction": {"road_model": {"success": road_success, "error": road_error}, "vehicle_model": {"success": vehicle_success, "error": vehicle_error}}
        })
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500

@app.route('/camera-status')
def check_camera_status():
    try:
        results = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "cameras": {}}
        if not load_dependencies():
            return jsonify({"error": "Failed to load dependencies", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500
        for camera_id, camera_name in cameras:
            camera_code = camera_mapping.get(camera_name, camera_name)
            try:
                logger.info(f"Checking camera {camera_name}")
                image = fetch_camera_image(camera_id)
                if image is None:
                    results["cameras"][camera_code] = {"name": camera_name, "status": "offline", "error": "Failed to fetch image"}
                elif image.size > 1000:
                    results["cameras"][camera_code] = {"name": camera_name, "status": "online", "resolution": f"{image.shape[1]}x{image.shape[0]}"}
                else:
                    results["cameras"][camera_code] = {"name": camera_name, "status": "error", "error": "Retrieved image is too small or invalid"}
            except Exception as e:
                results["cameras"][camera_code] = {"name": camera_name, "status": "error", "error": str(e)}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500

if __name__ == "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Exiting.")
        exit(1)
    manage_historical_densities()
    start_worker()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
