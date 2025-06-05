import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import cv2
import numpy as np
import requests
import gc

# Initialize Flask
app = Flask(__name__)

# Set up logging with INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google Drive setup
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CLIENT_ID = os.environ.get('GOOGLE_DRIVE_CLIENT_ID')
CLIENT_SECRET = os.environ.get('GOOGLE_DRIVE_CLIENT_SECRET')
REFRESH_TOKEN = os.environ.get('GOOGLE_DRIVE_REFRESH_TOKEN')
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')

# File name for density data
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
        temp_file = f"/tmp/{filename}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        file_id = get_file_id(filename)
        media = MediaFileUpload(temp_file, mimetype='application/json')
        if file_id:
            drive_service.files().update(fileId=file_id, media_body=media).execute()
            logger.info(f"Updated {filename} in Google Drive")
        else:
            file_metadata = {'name': filename, 'parents': [FOLDER_ID], 'mimeType': 'application/json'}
            drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logger.info(f"Uploaded {filename} to Google Drive")
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

# Camera websites list (reduced to 2 for testing)
camera_websites = [
    {'id': 'A', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623e7076f998a001b2523ea&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20S%C6%B0%20V%E1%BA%A1n%20H%E1%BA%A1nh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Lý Thái Tổ - Sư Vạn Hạnh'},
    {'id': 'B', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf8&camLocation=Ba%20Th%C3%A1ng%20Hai%20-%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': '3/2 – Cao Thắng'}
]

# Parse camera data
def parse_camera_data():
    cameras = []
    camera_mapping = {}
    for camera in camera_websites:
        try:
            camera_id = camera['id']
            camera_location = camera['title']
            camera_mapping[camera_id] = camera_location
            cameras.append((camera_id, camera_location))
            logger.info(f"Parsed camera {camera_id}: {camera_location}")
        except Exception as e:
            logger.error(f"Error parsing camera {camera}: {e}")
    logger.info(f"Parsed cameras: {cameras}")
    logger.info(f"Camera mapping: {camera_mapping}")
    return cameras, camera_mapping

cameras, camera_mapping = parse_camera_data()
CAMERA_URL_TEMPLATE = os.environ.get('CAMERA_URL_TEMPLATE', 'https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id={camera_id}&bg=black&w=300&h=230')

# Lazy-load dependencies
_tf, _cv2, _np, _requests, _torch, _transforms, _road_model, _vehicle_model, _session = [None] * 9
USE_MODELS = os.environ.get('USE_MODELS', 'false').lower() == 'true'
last_density_update = None
DEVICE = 'cpu'
worker_lock = threading.Lock()

# Remove model definitions to save memory (re-added only if USE_MODELS is True)
if USE_MODELS:
    import torch.nn as nn  # Added explicit import for nn
    class MiniUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super(MiniUNet, self).__init__()
            self.enc1 = self.mini_block(in_channels, 16)
            self.enc2 = self.mini_block(16, 32)
            self.bottleneck = self.mini_block(32, 64)
            self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = self.mini_block(64, 32)
            self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = self.mini_block(32, 16)
            self.final_conv = nn.Conv2d(16, out_channels, 1)
            self.pool = nn.MaxPool2d(2)
        
        def mini_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool(enc1))
            bottleneck = self.bottleneck(self.pool(enc2))
            dec2 = self.upconv2(bottleneck)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            return torch.sigmoid(self.final_conv(dec1))

def load_dependencies():
    global _tf, _cv2, _np, _requests, _torch, _transforms, _session
    if _tf is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_MEMORY_ALLOCATION'] = '128MB'  # Reduced to 128MB
            import tensorflow as tf
            import cv2
            import numpy as np
            import requests
            import torch
            import torchvision.transforms as transforms
            _tf, _cv2, _np, _requests, _torch, _transforms = tf, cv2, np, requests, torch, transforms
            _session = _requests.Session()
            _session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
                "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://giaothong.hochiminhcity.gov.vn/",
                "Origin": "https://giaothong.hochiminhcity.gov.vn"
            })
            logger.info("Dependencies loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return False
    return True

def load_models():
    global _road_model, _vehicle_model
    if not USE_MODELS:
        logger.info("USE_MODELS is False, skipping model loading")
        return True
    logger.info("=============================================")
    logger.info("LOADING MODELS - FORCED ATTEMPT")
    logger.info("=============================================")
    if not load_dependencies():
        logger.error("Failed to load dependencies - cannot load models")
        return False
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    road_model_path = os.path.join(base_directory, "unet_road_segmentation (Better).keras")
    vehicle_model_path = os.path.join(base_directory, "filtered_model_cpu.pth")
    logger.info(f"Checking for model files: Road: {os.path.exists(road_model_path)}, Vehicle: {os.path.exists(vehicle_model_path)}")
    try:
        logger.info("Loading road segmentation model (TensorFlow)...")
        _road_model = _tf.keras.models.load_model(road_model_path)
        logger.info("Loading vehicle detection model (PyTorch)...")
        _vehicle_model = MiniUNet(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = _torch.load(vehicle_model_path, map_location=DEVICE)
        _vehicle_model.load_state_dict(checkpoint['model_state_dict'])
        _vehicle_model.eval()
        logger.info("=============================================")
        logger.info("MODELS LOADED SUCCESSFULLY")
        logger.info(f"Vehicle model trained for {checkpoint.get('epoch', 'N/A')+1} epochs")
        logger.info(f"Best validation IoU: {checkpoint.get('val_iou', 'N/A'):.3f}")
        logger.info("=============================================")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False
    finally:
        gc.collect()

def preprocess_image(img):
    if not load_dependencies() or img is None or not USE_MODELS:
        return None, None
    try:
        img_road = _cv2.cvtColor(img, _cv2.COLOR_BGR2YCrCb)
        y, cr, cb = _cv2.split(img_road)
        clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        enhanced_img = _cv2.merge((y, cr, cb))
        img_road = _cv2.cvtColor(enhanced_img, _cv2.COLOR_YCrCb2BGR)
        img_road = _cv2.resize(img_road, (128, 128))
        img_road = img_road.astype('float32') / 255.0
        img_road = _np.expand_dims(img_road, axis=0)
        
        transform = _transforms.Compose([
            _transforms.ToPILImage(),
            _transforms.Resize((384, 384)),
            _transforms.ToTensor(),
        ])
        img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        img_vehicle = transform(img_rgb).unsqueeze(0).to(DEVICE)
        
        return img_road, img_vehicle
    finally:
        gc.collect()

def analyze_image(image):
    if not load_dependencies() or image is None or not USE_MODELS:
        return {
            "density": 0.0,
            "vehicle_count": 0,
            "avg_vehicle_size": 0,
            "density_metric": 0.0,
            "estimated_speed": 0.0,
            "traffic_level": "No Traffic"
        }
    try:
        if _road_model is None or _vehicle_model is None:
            logger.warning("Models not loaded, using fallback values")
            return {
                "density": 0.0,
                "vehicle_count": 0,
                "avg_vehicle_size": 0,
                "density_metric": 0.0,
                "estimated_speed": 0.0,
                "traffic_level": "No Traffic"
            }
        img_road, img_vehicle = preprocess_image(image)
        if img_road is None or img_vehicle is None:
            return {
                "density": 0.0,
                "vehicle_count": 0,
                "avg_vehicle_size": 0,
                "density_metric": 0.0,
                "estimated_speed": 0.0,
                "traffic_level": "No Traffic"
            }
        
        road_pred = _road_model.predict(img_road, verbose=0)
        road_mask = (road_pred.squeeze() > 0.5).astype(_np.uint8)
        road_mask_resized = _cv2.resize(road_mask, (image.shape[1], image.shape[0]), interpolation=_cv2.INTER_NEAREST)
        
        contours, _ = _cv2.findContours(road_mask_resized, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=_cv2.contourArea)
            epsilon = 0.01 * _cv2.arcLength(largest_contour, True)
            smoothed_contour = _cv2.approxPolyDP(largest_contour, epsilon, True)
            hull = _cv2.convexHull(smoothed_contour)
            refined_road_mask = _np.zeros_like(road_mask_resized, dtype=_np.uint8)
            _cv2.fillPoly(refined_road_mask, [hull], 255)
        else:
            refined_road_mask = road_mask_resized.copy()
        road_pixels = _np.count_nonzero(refined_road_mask)
        
        with _torch.no_grad():
            vehicle_pred = _vehicle_model(img_vehicle)
        vehicle_mask = vehicle_pred.squeeze().cpu().numpy()
        vehicle_mask_resized = _cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]))
        binary_vehicle_mask = (vehicle_mask_resized > 0.25).astype(_np.uint8)
        
        road_binary = (refined_road_mask > 0).astype(_np.uint8)
        vehicles_on_road = _np.logical_and(binary_vehicle_mask, road_binary).astype(_np.uint8)
        vehicle_pixels_on_road = _np.count_nonzero(vehicles_on_road)
        
        density_percentage = (vehicle_pixels_on_road / road_pixels * 100) if road_pixels > 0 else 0.0
        density_percentage = round(max(0, min(100, density_percentage)), 1)
        
        kernel_open = _np.ones((2, 2), _np.uint8)
        kernel_close = _np.ones((5, 5), _np.uint8)
        vehicles_on_road_cleaned = _cv2.morphologyEx(vehicles_on_road, _cv2.MORPH_OPEN, kernel_open, iterations=1)
        vehicles_on_road_cleaned = _cv2.morphologyEx(vehicles_on_road_cleaned, _cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        num_labels, _, stats, _ = _cv2.connectedComponentsWithStats(vehicles_on_road_cleaned, connectivity=8)
        blob_sizes = [stats[i, _cv2.CC_STAT_AREA] for i in range(1, num_labels) if 500 <= stats[i, _cv2.CC_STAT_AREA] <= 8000]
        
        estimated_vehicle_count = len(blob_sizes) if blob_sizes else 0
        avg_vehicle_size = int(_np.median(blob_sizes)) if blob_sizes else 0
        
        estimated_speed, density_metric, traffic_level = 0, 0, "No Traffic"
        if road_pixels > 0 and estimated_vehicle_count > 0:
            density_metric = estimated_vehicle_count / (road_pixels / 1000000)
            if density_metric < 10:
                traffic_level = "Free Flow"
            elif density_metric < 25:
                traffic_level = "Light Traffic"
            elif density_metric < 50:
                traffic_level = "Moderate Traffic"
            elif density_metric < 70:
                traffic_level = "Heavy Traffic"
            else:
                traffic_level = "Congested"
        
        logger.info(f"Calculated density: {density_percentage}%, vehicles: {estimated_vehicle_count}, speed: {estimated_speed:.1f} km/h, traffic: {traffic_level}")
        return {
            "density": density_percentage,
            "vehicle_count": estimated_vehicle_count,
            "avg_vehicle_size": avg_vehicle_size,
            "density_metric": round(density_metric, 2),
            "estimated_speed": round(estimated_speed, 1),
            "traffic_level": traffic_level
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {
            "density": 0.0,
            "vehicle_count": 0,
            "avg_vehicle_size": 0,
            "density_metric": 0.0,
            "estimated_speed": 0.0,
            "traffic_level": "No Traffic"
        }
    finally:
        gc.collect()

def fetch_camera_image(camera_id):
    if not load_dependencies():
        return None
    try:
        global _session
        if _session is None:
            _session = _requests.Session()
        _session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://giaothong.hochiminhcity.gov.vn/",
            "Origin": "https://giaothong.hochiminhcity.gov.vn"
        })
        try:
            warmup_response = _session.get("https://giaothong.hochiminhcity.gov.vn/", timeout=15)
            warmup_response.raise_for_status()
            logger.info(f"Warm-up request successful: {warmup_response.status_code}")
        except Exception as e:
            logger.error(f"Warm-up request failed: {e}")
            return None

        camera = next((c for c in camera_websites if c['id'] == camera_id), None)
        if not camera:
            logger.error(f"Camera {camera_id} not found in camera_websites")
            return None
        cam_id = camera['url'].split('camId=')[1].split('&')[0]

        url = CAMERA_URL_TEMPLATE.format(camera_id=cam_id)
        logger.info(f"Fetching image from primary URL: {url}")
        try:
            response = _session.get(url, timeout=15)
            response.raise_for_status()
            if response.content and len(response.content) > 100:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'image' in content_type:
                    image_array = _np.asarray(bytearray(response.content), dtype=_np.uint8)
                    image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)
                    if image is not None and image.size > 0:
                        logger.info(f"Successfully fetched and decoded image for camera {camera_id}")
                        return image
        except _requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
        except Exception as e:
            logger.error(f"Error fetching image for {camera_id}: {e}")
        logger.error(f"Failed to fetch valid image for {camera_id}")
        return None
    except Exception as e:
        logger.error(f"Critical error fetching camera image for {camera_id}: {e}")
        return None
    finally:
        gc.collect()

def fetch_and_process_densities():
    global last_density_update
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_density_update = datetime.now()
    results = {"timestamp": timestamp_str, "cameras": {}}
    success_count, failure_count = 0, 0
    for camera_id, camera_name in cameras:
        try:
            logger.info(f"Processing camera {camera_name} (ID: {camera_id})")
            image = fetch_camera_image(camera_id)
            density_data = {
                "name": camera_name,
                "density": 0.0,
                "vehicle_count": 0,
                "avg_vehicle_size": 0,
                "density_metric": 0.0,
                "estimated_speed": 0.0,
                "traffic_level": "No Traffic",
                "timestamp": timestamp_str
            }
            if image is None:
                failure_count += 1
                logger.warning(f"Image fetch failed for {camera_name} (ID: {camera_id})")
                results["cameras"][camera_id] = {
                    "name": camera_name,
                    "density": 0.0,
                    "vehicle_count": 0,
                    "avg_vehicle_size": 0,
                    "density_metric": 0.0,
                    "estimated_speed": 0.0,
                    "traffic_level": "No Image",
                    "timestamp": timestamp_str,
                    "error": "Failed to fetch image"
                }
            else:
                success_count += 1
                logger.info(f"Successfully fetched image for {camera_name}")
                analysis_result = analyze_image(image) if USE_MODELS else {"density": 0.0, "vehicle_count": 0, "avg_vehicle_size": 0, "density_metric": 0.0, "estimated_speed": 0.0, "traffic_level": "No Data"}
                density_data.update(analysis_result)
                results["cameras"][camera_id] = density_data
            logger.info(f"Processed camera {camera_name}: density={density_data['density']}, vehicles={density_data['vehicle_count']}")
        except Exception as e:
            failure_count += 1
            logger.error(f"Error processing camera {camera_name} (ID: {camera_id}): {e}")
            results["cameras"][camera_id] = density_data
        finally:
            gc.collect()
    logger.info(f"Camera processing complete. Success: {success_count}, Failure: {failure_count}")
    try:
        upload_json_to_drive(OUTPUT_JSON_FILE, results)
    except Exception as e:
        logger.error(f"Error saving density.json to Google Drive: {e}")
    return results

def density_worker():
    logger.info("Density worker initialized - running every 60 seconds")
    try:
        logger.info("Starting initial density calculation")
        fetch_and_process_densities()
        logger.info("Initial density calculation completed")
        while True:
            try:
                with worker_lock:
                    logger.info("Starting density processing cycle (60-second interval)")
                    fetch_and_process_densities()
                    logger.info("Density processing cycle completed")
                time.sleep(60)  # Increased to 60 seconds to allow more time
            except Exception as e:
                logger.error(f"Error in density worker cycle: {e}")
                time.sleep(10)
    except Exception as e:
        logger.error(f"Critical error in density worker: {e}")

def start_worker():
    try:
        logger.info("Starting worker - FOCUSING ON MODEL LOADING")
        if USE_MODELS:
            logger.info("Attempting to load models (forced)...")
            load_success = load_models()
            if not load_success:
                logger.error("Failed to load models! Check logs for details.")
        else:
            logger.info("USE_MODELS is False, skipping model loading")
        logger.info("Starting density worker thread...")
        density_thread = threading.Thread(target=density_worker, daemon=True)
        density_thread.start()
        logger.info("Density worker thread started")
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")

if __name__ != "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Application may not function correctly.")
    else:
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

@app.route('/live-densities')
def get_live_densities():
    try:
        density = download_json_from_drive(OUTPUT_JSON_FILE)
        if not density:
            return jsonify({
                "error": "No density data available yet",
                "message": "Please wait for the first calculation cycle"
            }), 404
        density["last_update"] = last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
        density["update_interval"] = "60 seconds"
        if last_density_update:
            next_update = last_density_update + timedelta(seconds=60)
            time_until_next = next_update - datetime.now()
            density["next_update_in"] = f"{int(time_until_next.total_seconds())} seconds" if time_until_next.total_seconds() > 0 else "Updating now..."
        return jsonify(density)
    except Exception as e:
        logger.error(f"Error reading live density: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/densities')
def get_densities():
    try:
        density = download_json_from_drive(OUTPUT_JSON_FILE)
        if not density:
            fetch_and_process_densities()
            density = download_json_from_drive(OUTPUT_JSON_FILE)
        raw_densities = {
            camera_code: {
                "density": camera_data["density"],
                "vehicle_count": camera_data["vehicle_count"],
                "avg_vehicle_size": camera_data["avg_vehicle_size"],
                "density_metric": camera_data["density_metric"],
                "estimated_speed": camera_data["estimated_speed"],
                "traffic_level": camera_data["traffic_level"]
            } for camera_code, camera_data in density["cameras"].items()
        }
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
        output_exists = bool(get_file_id(OUTPUT_JSON_FILE))
        return jsonify({
            "status": "healthy",
            "storage": {"backend": "Google Drive", "folder_id": FOLDER_ID, "output_file_exists": output_exists},
            "using_models": USE_MODELS,
            "last_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/refresh')
def refresh_densities():
    try:
        density_result = fetch_and_process_densities()
        return jsonify({
            "status": "success",
            "message": "Densities refreshed successfully",
            "density_timestamp": density_result["timestamp"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to refresh densities: {str(e)}"}), 500

if __name__ == "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Exiting.")
        exit(1)
    start_worker()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
