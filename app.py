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
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
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
            drive_service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            logger.info(f"Updated {filename} in Google Drive")
        else:
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

# Camera websites list (unchanged)
camera_websites = [
    {'id': 'A', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623e7076f998a001b2523ea&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20S%C6%B0%20V%E1%BA%A1n%20H%E1%BA%A1nh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Lý Thái Tổ - Sư Vạn Hạnh'},
    {'id': 'B', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf8&camLocation=Ba%20Th%C3%A1ng%20Hai%20-%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': '3/2 – Cao Thắng'},
    {'id': 'C', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae7a9cbfd3d90017e8f303&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20%E2%80%93%20Cao%20Th%E1%BA%AFng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Điện Biên Phủ - Cao Thắng'},
    {'id': 'D', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515ad21&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20Nguy%E1%BB%85n%20Tri%20Ph%C6%B0%C6%A1ng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Ngã sáu Nguyễn Tri Phương 1'},
    {'id': 'E', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515ad22&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20Nguy%E1%BB%85n%20Tri%20Ph%C6%B0%C6%A1ng&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Ngã sáu Nguyễn Tri Phương'},
    {'id': 'F', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5d8cdd26766c880017188974&camLocation=N%C3%BUt%20giao%20L%C3%AA%20%C4%90%E1%BA%A1i%20H%C3%A0nh%202%20(L%C3%AA%20%C4%90%E1%BA%A1i%20H%C3%A0nh)&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Lê Đại Hành 2'},
    {'id': 'G', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=63ae763bbfd3d90017e8f0c4&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Chi%E1%BB%83u&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Lý Thái Tổ - Nguyễn Đình Chiểu'},
    {'id': 'H', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf6&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Ngã sáu Cộng hòa 1'},
    {'id': 'I', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf7&camLocation=N%C3%BAt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera& primesVideoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Ngã sáu Cộng Hòa'},
    {'id': 'J', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf2&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A9%20-%20C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Điện Biên Phủ - CMT8'},
    {'id': 'K', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf9&camLocation=N%C3%BAt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Nút giao Công Trường Dân Chủ'},
    {'id': 'L', 'url': 'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acfa&camLocation=N%C3%BAt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8', 'title': 'Nút giao Công Trường Dân Chủ 1'}
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
worker_lock = threading.Lock()  # Add lock to prevent overlapping worker cycles

# Define MiniUNet architecture (unchanged)
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
            os.environ['TF_MEMORY_ALLOCATION'] = '256MB'
            import tensorflow as tf
            import cv2
            import numpy as np
            import requests
            import torch
            import torchvision.transforms as transforms
            _tf, _cv2, _np, _requests, _torch, _transforms = tf, cv2, np, requests, torch, transforms
            physical_devices = _tf.config.list_physical_devices('GPU')
            if physical_devices:
                _tf.config.experimental.set_memory_growth(physical_devices[0], True)
            _tf.config.threading.set_intra_op_parallelism_threads(1)
            _tf.config.threading.set_inter_op_parallelism_threads(1)
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
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=============================================")
        logger.error("MODEL LOADING FAILED")
        logger.error("=============================================")
        return False
    finally:
        gc.collect()  # Force garbage collection after model loading

def preprocess_image(img):
    if not load_dependencies() or img is None:
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
        gc.collect()  # Clean up memory after preprocessing

def estimate_vehicle_count_from_blobs(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    
    def find_vehicle_unit_size(sizes):
        sizes = sorted(sizes)
        smallest_blob = min(sizes)
        if smallest_blob > 1500:
            return 200
        
        q5, q95 = _np.percentile(sizes, [5, 95])
        filtered_sizes = [s for s in sizes if q5 <= s <= q95]
        if not filtered_sizes:
            filtered_sizes = sizes
        
        single_vehicle_candidates = [s for s in filtered_sizes if s <= _np.percentile(filtered_sizes, 25)]
        unit_size = _np.median(single_vehicle_candidates) if single_vehicle_candidates and min(single_vehicle_candidates) <= 1200 else 200
        return max(500, min(unit_size, 1800))
    
    unit_vehicle_size = find_vehicle_unit_size(significant_blobs)
    total_vehicles = 0
    for blob_size in significant_blobs:
        vehicles_in_blob = 1 if blob_size < unit_vehicle_size * 1.2 else max(1, int(blob_size / unit_vehicle_size))
        total_vehicles += vehicles_in_blob
    
    return int(total_vehicles), int(unit_vehicle_size)

def estimate_vehicles_statistical_clustering(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    
    sizes = _np.array(significant_blobs)
    smallest_blob = min(sizes)
    
    if smallest_blob > 1500:
        avg_single_vehicle = 200
        vehicle_count = sum(max(1, int(blob_size / avg_single_vehicle)) for blob_size in sizes)
        return vehicle_count, int(avg_single_vehicle)
    
    q20, q50, q80 = _np.percentile(sizes, [20, 50, 80])
    iqr = q80 - q20
    single_vehicle_upper = q20 + 0.3 * iqr
    single_vehicle_blobs = sizes[sizes <= single_vehicle_upper]
    multi_vehicle_blobs = sizes[sizes > single_vehicle_upper]
    
    avg_single_vehicle = _np.median(single_vehicle_blobs) if len(single_vehicle_blobs) > 0 else max(200, q20)
    avg_single_vehicle = max(200, min(avg_single_vehicle, 1500))
    
    vehicle_count = len(single_vehicle_blobs)
    vehicle_count += sum(max(1, int(blob_size / avg_single_vehicle)) for blob_size in multi_vehicle_blobs)
    
    return vehicle_count, int(avg_single_vehicle)

def estimate_vehicles_histogram_analysis(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    
    sizes = _np.array(significant_blobs)
    smallest_blob = min(sizes)
    
    if smallest_blob > 1500:
        typical_vehicle_size = 200
        total_vehicles = sum(max(1, int(size / typical_vehicle_size)) for size in sizes)
        return int(total_vehicles), int(typical_vehicle_size)
    
    if len(significant_blobs) >= 3:
        n_bins = min(8, len(significant_blobs) // 2)
        hist, bin_edges = _np.histogram(sizes, bins=n_bins)
        peak_bin_idx = _np.argmax(hist)
        peak_range = (bin_edges[peak_bin_idx], bin_edges[peak_bin_idx + 1])
        peak_sizes = sizes[(sizes >= peak_range[0]) & (sizes <= peak_range[1])]
        typical_vehicle_size = _np.median(peak_sizes) if len(peak_sizes) > 0 else _np.percentile(sizes, 20)
    else:
        typical_vehicle_size = _np.median(sizes)
    
    typical_vehicle_size = max(200, min(typical_vehicle_size, 1500))
    total_vehicles = sum(max(1, int(size / typical_vehicle_size)) for size in sizes)
    
    return int(total_vehicles), int(typical_vehicle_size)

def apply_greenshields_model(vehicle_count, road_area_pixels, image_shape):
    if road_area_pixels == 0 or vehicle_count == 0:
        return 0, 0, "No Traffic"
    
    image_area = image_shape[0] * image_shape[1]
    road_ratio = road_area_pixels / image_area
    density_metric = vehicle_count / (road_ratio * 1000)
    
    free_flow_speed = 60
    jam_density = 80
    speed_ratio = max(0, 1 - (density_metric / jam_density)) if jam_density > 0 else 0
    estimated_speed = free_flow_speed * speed_ratio
    
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
    
    return estimated_speed, density_metric, traffic_level

def analyze_image(image):
    if not load_dependencies() or image is None:
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
        
        # Road segmentation
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
        
        # Vehicle detection
        with _torch.no_grad():
            vehicle_pred = _vehicle_model(img_vehicle)
        vehicle_mask = vehicle_pred.squeeze().cpu().numpy()
        vehicle_mask_resized = _cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]))
        binary_vehicle_mask = (vehicle_mask_resized > 0.25).astype(_np.uint8)
        
        # Calculate vehicle pixels on road
        road_binary = (refined_road_mask > 0).astype(_np.uint8)
        vehicles_on_road = _np.logical_and(binary_vehicle_mask, road_binary).astype(_np.uint8)
        vehicle_pixels_on_road = _np.count_nonzero(vehicles_on_road)
        
        # Calculate density
        density_percentage = (vehicle_pixels_on_road / road_pixels * 100) if road_pixels > 0 else 0.0
        density_percentage = round(max(0, min(100, density_percentage)), 1)
        
        # Vehicle counting with blob analysis
        kernel_open = _np.ones((2, 2), _np.uint8)
        kernel_close = _np.ones((5, 5), _np.uint8)
        vehicles_on_road_cleaned = _cv2.morphologyEx(vehicles_on_road, _cv2.MORPH_OPEN, kernel_open, iterations=1)
        vehicles_on_road_cleaned = _cv2.morphologyEx(vehicles_on_road_cleaned, _cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        num_labels, _, stats, _ = _cv2.connectedComponentsWithStats(vehicles_on_road_cleaned, connectivity=8)
        blob_sizes = []
        min_reasonable_blob = 500
        max_reasonable_blob = 8000
        for i in range(1, num_labels):
            blob_size = stats[i, _cv2.CC_STAT_AREA]
            if min_reasonable_blob <= blob_size <= max_reasonable_blob:
                blob_sizes.append(blob_size)
        
        if blob_sizes:
            method1_count, method1_unit = estimate_vehicle_count_from_blobs(blob_sizes)
            method2_count, method2_unit = estimate_vehicles_statistical_clustering(blob_sizes)
            method3_count, method3_unit = estimate_vehicles_histogram_analysis(blob_sizes)
            all_counts = [method1_count, method2_count, method3_count]
            estimated_vehicle_count = min(all_counts)
            avg_vehicle_size = method1_unit if estimated_vehicle_count == method1_count else method2_unit if estimated_vehicle_count == method2_count else method3_unit
        else:
            estimated_vehicle_count = 0
            avg_vehicle_size = 0
        
        # Apply Greenshields model
        estimated_speed, density_metric, traffic_level = apply_greenshields_model(estimated_vehicle_count, road_pixels, image.shape)
        
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
        import traceback
        logger.error(traceback.format_exc())
        return {
            "density": 0.0,
            "vehicle_count": 0,
            "avg_vehicle_size": 0,
            "density_metric": 0.0,
            "estimated_speed": 0.0,
            "traffic_level": "No Traffic"
        }
    finally:
        gc.collect()  # Clean up memory after analysis

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
        # Warm-up request to establish session and get cookies
        try:
            warmup_response = _session.get("https://giaothong.hochiminhcity.gov.vn/", timeout=15)
            warmup_response.raise_for_status()
            logger.info(f"Warm-up request successful: {warmup_response.status_code}")
            logger.debug(f"Warm-up cookies: {_session.cookies.get_dict()}")
        except Exception as e:
            logger.error(f"Warm-up request failed: {e}")
            return None

        # Find the camera's camId and videoUrl from camera_websites
        camera = next((c for c in camera_websites if c['id'] == camera_id), None)
        if not camera:
            logger.error(f"Camera {camera_id} not found in camera_websites")
            return None
        cam_id = camera['url'].split('camId=')[1].split('&')[0]
        video_url = camera['url'].split('videoUrl=')[1] if 'videoUrl=' in camera['url'] else None

        # Try primary URL first
        url = CAMERA_URL_TEMPLATE.format(camera_id=cam_id)
        logger.info(f"Fetching image from primary URL: {url}")
        for attempt in range(3):
            try:
                response = _session.get(url, timeout=15)
                response.raise_for_status()
                logger.debug(f"Response headers: {response.headers}")
                logger.debug(f"Response content length: {len(response.content)} bytes")
                if response.content and len(response.content) > 100:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'image' not in content_type:
                        logger.warning(f"Unexpected Content-Type: {content_type}")
                        continue
                    image_array = _np.asarray(bytearray(response.content), dtype=_np.uint8)
                    image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)
                    if image is not None and image.size > 0:
                        logger.info(f"Successfully fetched and decoded image for camera {camera_id}")
                        return image
                    logger.warning(f"Failed to decode image from {url} (attempt {attempt+1}/3)")
                else:
                    logger.warning(f"Empty or invalid response from {url} (attempt {attempt+1}/3)")
            except _requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {url}: {e} (attempt {attempt+1}/3)")
                if e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    logger.error(f"Response headers: {e.response.headers}")
                    logger.error(f"Response content: {e.response.text[:500]}")
                break  # Stop retries on HTTP error to save memory
            except Exception as e:
                logger.error(f"Error fetching image for {camera_id}: {e} (attempt {attempt+1}/3)")
            time.sleep(1)  # Reduced delay to 1 second
        # Fallback to video URL if available
        if video_url:
            logger.info(f"Falling back to video URL for camera {camera_id}: {video_url}")
            try:
                cap = _cv2.VideoCapture(video_url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None and frame.size > 0:
                        logger.info(f"Successfully extracted frame from video stream for camera {camera_id}")
                        return frame
                    logger.warning(f"Failed to extract frame from {video_url}")
                else:
                    logger.error(f"Failed to open video stream {video_url}")
            except Exception as e:
                logger.error(f"Error extracting frame from video URL {video_url}: {e}")
        else:
            logger.warning(f"No video URL available for camera {camera_id}")
        logger.error(f"Failed to fetch valid image for {camera_id} after 3 attempts")
        return None
    except Exception as e:
        logger.error(f"Critical error fetching camera image for {camera_id}: {e}")
        return None
    finally:
        gc.collect()  # Clean up memory after fetch

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
                analysis_result = analyze_image(image)
                density_data.update(analysis_result)
                results["cameras"][camera_id] = density_data
            logger.info(f"Processed camera {camera_name}: density={density_data['density']}, vehicles={density_data['vehicle_count']}")
        except Exception as e:
            failure_count += 1
            logger.error(f"Error processing camera {camera_name} (ID: {camera_id}): {e}")
            results["cameras"][camera_id] = density_data
        finally:
            gc.collect()  # Clean up memory after each camera
    logger.info(f"Camera processing complete. Success: {success_count}, Failure: {failure_count}")
    try:
        upload_json_to_drive(OUTPUT_JSON_FILE, results)
    except Exception as e:
        logger.error(f"Error saving density.json to Google Drive: {e}")
    return results

def density_worker():
    logger.info("Density worker initialized - running every 30 seconds")
    try:
        logger.info("Starting initial density calculation")
        fetch_and_process_densities()
        logger.info("Initial density calculation completed")
        while True:
            try:
                with worker_lock:  # Ensure only one cycle runs at a time
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
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras")):
                logger.error("Problem: Road model file not found")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth")):
                logger.error("Problem: Vehicle model file not found")
            else:
                logger.error("Problem: Unclear - check model format or compatibility")
        logger.info("Starting density worker thread...")
        density_thread = threading.Thread(target=density_worker, daemon=True)
        density_thread.start()
        logger.info("Density worker thread started")
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Initialize Google Drive and start worker
if __name__ != "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Application may not function correctly.")
    else:
        start_worker()

# Flask routes (unchanged)
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
        density["update_interval"] = "30 seconds"
        if last_density_update:
            next_update = last_density_update + timedelta(seconds=30)
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
            "storage": {
                "backend": "Google Drive",
                "folder_id": FOLDER_ID,
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
        density_result = fetch_and_process_densities()
        return jsonify({
            "status": "success",
            "message": "Densities refreshed successfully",
            "density_timestamp": density_result["timestamp"]
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
            "unet_road_segmentation_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras"))},
            "filtered_model_cpu": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth"))}
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
            "output_json_exists": bool(get_file_id(OUTPUT_JSON_FILE))
        }
        try:
            files_in_base_dir = os.listdir(os.environ.get('BASE_DIR', os.getcwd()))
        except Exception as e:
            files_in_base_dir = f"Error listing files: {str(e)}"
        model_load_status = {
            "road_model_loaded": _road_model is not None,
            "vehicle_model_loaded": _vehicle_model is not None,
            "dependencies_loaded": _tf is not None and _cv2 is not None and _np is not None and _requests is not None and _torch is not None and _transforms is not None
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
                "road_model": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras"))},
                "vehicle_model": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth"))}
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
        road_success, road_error = False, None
        try:
            test_input_tf = _np.zeros((1, 128, 128, 3), dtype='float32')
            _road_model.predict(test_input_tf, verbose=0)
            road_success = True
        except Exception as e:
            road_error = str(e)
        vehicle_success, vehicle_error = False, None
        try:
            test_input_torch = _torch.zeros((1, 3, 384, 384), dtype=_torch.float32).to(DEVICE)
            with _torch.no_grad():
                _vehicle_model(test_input_torch)
            vehicle_success = True
        except Exception as e:
            vehicle_error = str(e)
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "models_loaded": {"road_model": _road_model is not None, "vehicle_model": _vehicle_model is not None},
            "test_prediction": {
                "road_model": {"success": road_success, "error": road_error},
                "vehicle_model": {"success": vehicle_success, "error": vehicle_error}
            }
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
            try:
                logger.info(f"Checking camera {camera_name} (ID: {camera_id})")
                image = fetch_camera_image(camera_id)
                if image is None:
                    results["cameras"][camera_id] = {"name": camera_name, "status": "offline", "error": "Failed to fetch image"}
                elif image.size > 1000:
                    results["cameras"][camera_id] = {"name": camera_name, "status": "online", "resolution": f"{image.shape[1]}x{image.shape[0]}"}
                else:
                    results["cameras"][camera_id] = {"name": camera_name, "status": "error", "error": "Retrieved image is too small or invalid"}
            except Exception as e:
                results["cameras"][camera_id] = {"name": camera_name, "status": "error", "error": str(e)}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500

if __name__ == "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Exiting.")
        exit(1)
    start_worker()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug mode on Render
