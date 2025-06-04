import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
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

# File names for data
DENSITIES_FILE = "densities.json"
VEHICLE_COUNTS_FILE = "vehicle_counts.json"
TODAY_VEHICLE_COUNTS_FILE = "today_vehicle_counts.json"
YESTERDAY_VEHICLE_COUNTS_FILE = "yesterday_vehicle_counts.json"
DAY_BEFORE_YESTERDAY_VEHICLE_COUNTS_FILE = "day_before_yesterday_vehicle_counts.json"
CRITICAL_VEHICLE_COUNTS_FILE = "critical_vehicle_counts.json"
DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE = "day_before_yesterday_critical_vehicle_counts.json"

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

# Camera list (simulated for 12 cameras)
cameras = [
    ('A', 'Lý Thái Tổ - Sư Vạn Hạnh'),
    ('B', '3/2 – Cao Thắng'),
    ('C', 'Điện Biên Phủ - Cao Thắng'),
    ('D', 'Ngã sáu Nguyễn Tri Phương 1'),
    ('E', 'Ngã sáu Nguyễn Tri Phương'),
    ('F', 'Lê Đại Hành 2'),
    ('G', 'Lý Thái Tổ - Nguyễn Đình Chiểu'),
    ('H', 'Ngã sáu Cộng Hòa 1'),
    ('I', 'Ngã sáu Cộng Hòa'),
    ('J', 'Điện Biên Phủ - CMT8'),
    ('K', 'Nút giao Công Trường Dân Chủ'),
    ('L', 'Nút giao Công Trường Dân Chủ 1')
]

# Lazy-load dependencies
_tf, _cv2, _np, _torch, _transforms, _road_model, _vehicle_model = [None] * 7
USE_MODELS = os.environ.get('USE_MODELS', 'true').lower() == 'true'
last_update = None
DEVICE = 'cpu'

# Define MiniUNet architecture
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
    global _tf, _cv2, _np, _torch, _transforms
    if _tf is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            import cv2
            import numpy as np
            import torch
            import torchvision.transforms as transforms
            _tf, _cv2, _np, _torch, _transforms = tf, cv2, np, torch, transforms
            logger.info("Dependencies loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return False
    return True

def load_models():
    global _road_model, _vehicle_model
    logger.info("Loading models...")
    if not load_dependencies():
        logger.error("Failed to load dependencies - cannot load models")
        return False
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    road_model_path = os.path.join(base_directory, "unet_road_segmentation (Better).keras")
    vehicle_model_path = os.path.join(base_directory, "filtered_model_cpu.pth")
    try:
        logger.info(f"Loading road model from {road_model_path}")
        _road_model = _tf.keras.models.load_model(road_model_path)
        logger.info("Road model loaded successfully")
        
        logger.info(f"Loading vehicle model from {vehicle_model_path}")
        _vehicle_model = MiniUNet(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = _torch.load(vehicle_model_path, map_location=DEVICE)
        _vehicle_model.load_state_dict(checkpoint['model_state_dict'])
        _vehicle_model.eval()
        logger.info(f"Vehicle model loaded successfully. Trained for {checkpoint.get('epoch', 'N/A')+1} epochs")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_image_for_road(image):
    if not load_dependencies() or image is None:
        return None
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    img_road = _cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    img_road = _np.expand_dims(img_road, axis=0)
    return img_road

def preprocess_image_for_vehicle(image):
    if not load_dependencies() or image is None:
        return None
    transform = _transforms.Compose([
        _transforms.ToPILImage(),
        _transforms.Resize((384, 384)),
        _transforms.ToTensor(),
    ])
    img_rgb = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
    img_vehicle = transform(img_rgb).unsqueeze(0).to(DEVICE)
    return img_vehicle

def estimate_vehicle_count_from_blobs(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    def find_vehicle_unit_size(sizes):
        sizes = sorted(sizes)
        smallest_blob = min(sizes)
        if smallest_blob > 1500:
            logger.info("Only large blobs detected - using fixed vehicle size estimate")
            return 200
        q5, q95 = _np.percentile(sizes, [5, 95])
        filtered_sizes = [s for s in sizes if q5 <= s <= q95]
        if not filtered_sizes:
            filtered_sizes = sizes
        single_vehicle_candidates = [s for s in filtered_sizes if s <= _np.percentile(filtered_sizes, 25)]
        if single_vehicle_candidates and min(single_vehicle_candidates) <= 1200:
            unit_size = _np.median(single_vehicle_candidates)
            logger.info(f"Found small reference blobs - estimated unit size: {unit_size:.0f}px")
        else:
            logger.info("No small reference blobs found - using fixed vehicle size")
            unit_size = 200
        return max(500, min(unit_size, 1800))
    unit_vehicle_size = find_vehicle_unit_size(significant_blobs)
    total_vehicles = 0
    for blob_size in significant_blobs:
        if blob_size < unit_vehicle_size * 1.2:
            vehicles_in_blob = 1
        else:
            vehicles_in_blob = max(1, int(blob_size / unit_vehicle_size))
        total_vehicles += vehicles_in_blob
    return int(total_vehicles), int(unit_vehicle_size)

def estimate_vehicles_statistical_clustering(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    sizes = _np.array(significant_blobs)
    smallest_blob = min(sizes)
    if smallest_blob > 1500:
        logger.info("Statistical method: Only large blobs - using fixed 200px vehicle size")
        avg_single_vehicle = 200
        vehicle_count = 0
        for blob_size in sizes:
            vehicles_in_blob = max(1, int(blob_size / avg_single_vehicle))
            vehicle_count += vehicles_in_blob
        return vehicle_count, int(avg_single_vehicle)
    q20, q50, q80 = _np.percentile(sizes, [20, 50, 80])
    iqr = q80 - q20
    single_vehicle_upper = q20 + 0.3 * iqr
    single_vehicle_blobs = sizes[sizes <= single_vehicle_upper]
    multi_vehicle_blobs = sizes[sizes > single_vehicle_upper]
    if len(single_vehicle_blobs) > 0:
        avg_single_vehicle = _np.median(single_vehicle_blobs)
    else:
        avg_single_vehicle = max(200, q20)
    avg_single_vehicle = max(200, min(avg_single_vehicle, 1500))
    vehicle_count = len(single_vehicle_blobs)
    for blob_size in multi_vehicle_blobs:
        vehicles_in_blob = max(1, int(blob_size / avg_single_vehicle))
        vehicle_count += vehicles_in_blob
    return vehicle_count, int(avg_single_vehicle)

def estimate_vehicles_histogram_analysis(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    sizes = _np.array(significant_blobs)
    smallest_blob = min(sizes)
    if smallest_blob > 1500:
        logger.info("Histogram method: Only large blobs - using fixed 200px vehicle size")
        typical_vehicle_size = 200
        total_vehicles = 0
        for size in sizes:
            vehicles_in_blob = max(1, int(size / typical_vehicle_size))
            total_vehicles += vehicles_in_blob
        return int(total_vehicles), int(typical_vehicle_size)
    if len(significant_blobs) >= 3:
        n_bins = min(8, len(significant_blobs) // 2)
        hist, bin_edges = _np.histogram(sizes, bins=n_bins)
        peak_bin_idx = _np.argmax(hist)
        peak_range = (bin_edges[peak_bin_idx], bin_edges[peak_bin_idx + 1])
        peak_sizes = sizes[(sizes >= peak_range[0]) & (sizes <= peak_range[1])]
        if len(peak_sizes) > 0:
            typical_vehicle_size = _np.median(peak_sizes)
        else:
            typical_vehicle_size = _np.percentile(sizes, 20)
    else:
        typical_vehicle_size = _np.median(sizes)
    typical_vehicle_size = max(200, min(typical_vehicle_size, 1500))
    total_vehicles = 0
    for size in sizes:
        vehicles_in_blob = max(1, int(size / typical_vehicle_size))
        total_vehicles += vehicles_in_blob
    return int(total_vehicles), int(typical_vehicle_size)

def load_sample_image():
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    image_path = os.path.join(base_directory, "sample_traffic_image.jpg")
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Sample image not found at {image_path}, using fallback logic")
            return None
        image = _cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load sample image from {image_path}")
            return None
        logger.info(f"Loaded sample image from {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading sample image: {e}")
        return None

def analyze_image_for_density(image):
    if not load_dependencies() or image is None or _road_model is None or _vehicle_model is None:
        logger.error("Dependencies or models not loaded, or image is None")
        return {"density_percentage": 0.0, "road_pixels": 0, "vehicle_pixels_on_road": 0}
    try:
        # Road segmentation
        img_road = preprocess_image_for_road(image)
        if img_road is None:
            return {"density_percentage": 0.0, "road_pixels": 0, "vehicle_pixels_on_road": 0}
        road_pred = _road_model.predict(img_road, verbose=0)
        road_mask = (road_pred.squeeze() > 0.5).astype(np.uint8)
        mask_resized = _cv2.resize(road_mask, (image.shape[1], image.shape[0]), interpolation=_cv2.INTER_NEAREST)
        
        # Refine road mask
        contours, _ = _cv2.findContours(mask_resized, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        refined_road_mask = mask_resized
        if contours:
            largest_contour = max(contours, key=_cv2.contourArea)
            hull = _cv2.convexHull(largest_contour)
            refined_road_mask = np.zeros_like(mask_resized, dtype=np.uint8)
            _cv2.fillPoly(refined_road_mask, [hull], 255)
        
        road_pixels = np.count_nonzero(refined_road_mask)
        
        # Vehicle detection
        img_vehicle = preprocess_image_for_vehicle(image)
        if img_vehicle is None:
            return {"density_percentage": 0.0, "road_pixels": road_pixels, "vehicle_pixels_on_road": 0}
        with _torch.no_grad():
            vehicle_pred = _vehicle_model(img_vehicle)
        vehicle_mask = vehicle_pred.squeeze().cpu().numpy()
        vehicle_mask_resized = _cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]))
        binary_vehicle_mask = (vehicle_mask_resized > 0.25).astype(np.uint8)
        
        # Calculate vehicles on road
        road_binary = (refined_road_mask > 0).astype(np.uint8)
        vehicles_on_road = np.logical_and(binary_vehicle_mask, road_binary).astype(np.uint8)
        vehicle_pixels_on_road = np.count_nonzero(vehicles_on_road)
        
        # Density calculation
        density_percentage = (vehicle_pixels_on_road / road_pixels * 100) if road_pixels > 0 else 0.0
        
        logger.info(f"Density analysis: {density_percentage:.2f}%")
        return {
            "density_percentage": density_percentage,
            "road_pixels": road_pixels,
            "vehicle_pixels_on_road": vehicle_pixels_on_road
        }
    except Exception as e:
        logger.error(f"Error analyzing image for density: {e}")
        return {"density_percentage": 0.0, "road_pixels": 0, "vehicle_pixels_on_road": 0}

def analyze_image_for_vehicles(image):
    if not load_dependencies() or image is None or _vehicle_model is None:
        logger.error("Dependencies or vehicle model not loaded, or image is None")
        return {"vehicle_count": 0, "avg_vehicle_size": 0}
    try:
        img_vehicle = preprocess_image_for_vehicle(image)
        if img_vehicle is None:
            return {"vehicle_count": 0, "avg_vehicle_size": 0}
        with _torch.no_grad():
            vehicle_pred = _vehicle_model(img_vehicle)
        vehicle_mask = vehicle_pred.squeeze().cpu().numpy()
        vehicle_mask_resized = _cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]))
        binary_vehicle_mask = (vehicle_mask_resized > 0.25).astype(np.uint8)
        
        # Clean up noise
        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        vehicles_cleaned = _cv2.morphologyEx(binary_vehicle_mask, _cv2.MORPH_OPEN, kernel_open, iterations=1)
        vehicles_cleaned = _cv2.morphologyEx(vehicles_cleaned, _cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Connected components
        num_labels, labels, stats, _ = _cv2.connectedComponentsWithStats(vehicles_cleaned, connectivity=8)
        blob_sizes = [stats[i, _cv2.CC_STAT_AREA] for i in range(1, num_labels) if 500 <= stats[i, _cv2.CC_STAT_AREA] <= 8000]
        
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
        
        logger.info(f"Vehicle count: {estimated_vehicle_count}, Avg size: {avg_vehicle_size}")
        return {
            "vehicle_count": estimated_vehicle_count,
            "avg_vehicle_size": avg_vehicle_size
        }
    except Exception as e:
        logger.error(f"Error analyzing image for vehicles: {e}")
        return {"vehicle_count": 0, "avg_vehicle_size": 0}

def check_new_day():
    today = datetime.now().date()
    today_vehicle_counts = download_json_from_drive(TODAY_VEHICLE_COUNTS_FILE) or {}
    if 'date' in today_vehicle_counts:
        try:
            file_date = datetime.strptime(today_vehicle_counts['date'], '%Y-%m-%d').date()
            if file_date < today:
                logger.info(f"New day detected. Transferring vehicle counts from {file_date}")
                yesterday_vehicle_counts = download_json_from_drive(YESTERDAY_VEHICLE_COUNTS_FILE) or {}
                if yesterday_vehicle_counts:
                    upload_json_to_drive(DAY_BEFORE_YESTERDAY_VEHICLE_COUNTS_FILE, yesterday_vehicle_counts)
                    update_day_before_yesterday_critical_vehicle_counts(yesterday_vehicle_counts)
                upload_json_to_drive(YESTERDAY_VEHICLE_COUNTS_FILE, today_vehicle_counts)
                update_critical_vehicle_counts(today_vehicle_counts)
                today_vehicle_counts = {'date': today.strftime('%Y-%m-%d'), 'counts_by_time': {}}
                upload_json_to_drive(TODAY_VEHICLE_COUNTS_FILE, today_vehicle_counts)
                logger.info("Successfully transferred vehicle counts and reset today's data")
        except Exception as e:
            logger.error(f"Error processing date change: {e}")

def update_critical_vehicle_counts(vehicle_counts_data):
    try:
        critical_vehicle_counts = download_json_from_drive(CRITICAL_VEHICLE_COUNTS_FILE) or {}
        counts_by_time = vehicle_counts_data.get('counts_by_time', {})
        for camera_code, _ in cameras:
            max_vehicle_count = 0
            for timestamp, cameras_data in counts_by_time.items():
                if camera_code in cameras_data:
                    vehicle_count = cameras_data[camera_code].get('vehicle_count', 0)
                    max_vehicle_count = max(max_vehicle_count, vehicle_count)
            if camera_code not in critical_vehicle_counts or max_vehicle_count > critical_vehicle_counts[camera_code]:
                critical_vehicle_counts[camera_code] = max_vehicle_count
                logger.info(f"Updated critical vehicle count for {camera_code}: {max_vehicle_count}")
        upload_json_to_drive(CRITICAL_VEHICLE_COUNTS_FILE, critical_vehicle_counts)
        logger.info("Critical vehicle counts updated successfully")
    except Exception as e:
        logger.error(f"Error updating critical vehicle counts: {e}")

def update_day_before_yesterday_critical_vehicle_counts(vehicle_counts_data):
    try:
        critical_vehicle_counts = download_json_from_drive(DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE) or {}
        counts_by_time = vehicle_counts_data.get('counts_by_time', {})
        for camera_code, _ in cameras:
            max_vehicle_count = 0
            for timestamp, cameras_data in counts_by_time.items():
                if camera_code in cameras_data:
                    vehicle_count = cameras_data[camera_code].get('vehicle_count', 0)
                    max_vehicle_count = max(max_vehicle_count, vehicle_count)
            if camera_code not in critical_vehicle_counts or max_vehicle_count > critical_vehicle_counts[camera_code]:
                critical_vehicle_counts[camera_code] = max_vehicle_count
                logger.info(f"Updated day before yesterday critical vehicle count for {camera_code}: {max_vehicle_count}")
        upload_json_to_drive(DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE, critical_vehicle_counts)
        logger.info("Day before yesterday critical vehicle counts updated successfully")
    except Exception as e:
        logger.error(f"Error updating day before yesterday critical vehicle counts: {e}")

def manage_historical_vehicle_counts():
    check_new_day()
    today_vehicle_counts = download_json_from_drive(TODAY_VEHICLE_COUNTS_FILE) or {}
    if 'date' not in today_vehicle_counts or today_vehicle_counts['date'] != datetime.now().date().strftime('%Y-%m-%d'):
        today_vehicle_counts = {
            'date': datetime.now().date().strftime('%Y-%m-%d'),
            'counts_by_time': {}
        }
        upload_json_to_drive(TODAY_VEHICLE_COUNTS_FILE, today_vehicle_counts)
    for file_name in [CRITICAL_VEHICLE_COUNTS_FILE, DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE]:
        critical_vehicle_counts = download_json_from_drive(file_name)
        if not critical_vehicle_counts:
            sample_critical_vehicle_counts = {camera_id: 0 for camera_id, _ in cameras}
            upload_json_to_drive(file_name, sample_critical_vehicle_counts)
    return today_vehicle_counts

def store_today_vehicle_count(timestamp_str, camera_code, count_data):
    try:
        today_vehicle_counts = download_json_from_drive(TODAY_VEHICLE_COUNTS_FILE) or {}
        if 'counts_by_time' not in today_vehicle_counts:
            today_vehicle_counts['counts_by_time'] = {}
        if timestamp_str not in today_vehicle_counts['counts_by_time']:
            today_vehicle_counts['counts_by_time'][timestamp_str] = {}
        today_vehicle_counts['counts_by_time'][timestamp_str][camera_code] = count_data
        upload_json_to_drive(TODAY_VEHICLE_COUNTS_FILE, today_vehicle_counts)
    except Exception as e:
        logger.error(f"Error storing today's vehicle count: {e}")

def process_traffic_data():
    global last_update
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_update = datetime.now()
    
    # Load sample image
    image = load_sample_image()
    if image is None:
        logger.error("No sample image available, using fallback values")
        density_results = {"density_percentage": 0.0, "road_pixels": 0, "vehicle_pixels_on_road": 0}
        vehicle_results = {"vehicle_count": 0, "avg_vehicle_size": 0}
    else:
        density_results = analyze_image_for_density(image)
        vehicle_results = analyze_image_for_vehicles(image)
    
    # Simulate per-camera data (same results for all cameras due to single sample image)
    density_data = {"timestamp": timestamp_str, "cameras": {}}
    vehicle_data = {"timestamp": timestamp_str, "cameras": {}}
    success_count, failure_count = 0, 0
    
    for camera_id, camera_name in cameras:
        try:
            density_data["cameras"][camera_id] = {
                "name": camera_name,
                "density_percentage": density_results["density_percentage"],
                "road_pixels": density_results["road_pixels"],
                "vehicle_pixels_on_road": density_results["vehicle_pixels_on_road"],
                "timestamp": timestamp_str
            }
            vehicle_data["cameras"][camera_id] = {
                "name": camera_name,
                "vehicle_count": vehicle_results["vehicle_count"],
                "avg_vehicle_size": vehicle_results["avg_vehicle_size"],
                "timestamp": timestamp_str
            }
            store_today_vehicle_count(timestamp_str, camera_id, vehicle_data["cameras"][camera_id])
            success_count += 1
            logger.info(f"Processed camera {camera_name}: density={density_results['density_percentage']:.2f}%, vehicles={vehicle_results['vehicle_count']}")
        except Exception as e:
            failure_count += 1
            logger.error(f"Error processing camera {camera_name}: {e}")
    
    logger.info(f"Processing complete. Success: {success_count}, Failure: {failure_count}")
    
    # Save to Google Drive
    try:
        upload_json_to_drive(DENSITIES_FILE, density_data)
        upload_json_to_drive(VEHICLE_COUNTS_FILE, vehicle_data)
    except Exception as e:
        logger.error(f"Error saving to Google Drive: {e}")
    
    return density_data, vehicle_data

def traffic_worker():
    logger.info("Traffic worker initialized - running every 30 seconds")
    try:
        logger.info("Starting initial traffic calculation")
        manage_historical_vehicle_counts()
        process_traffic_data()
        logger.info("Initial traffic calculation completed")
        while True:
            try:
                logger.info("Starting traffic processing cycle")
                process_traffic_data()
                logger.info("Traffic processing cycle completed")
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in traffic worker cycle: {e}")
                time.sleep(10)
    except Exception as e:
        logger.error(f"Critical error in traffic worker: {e}")

def start_worker():
    try:
        logger.info("Starting worker")
        load_success = load_models()
        if load_success:
            logger.info("Models loaded successfully")
        else:
            logger.error("Failed to load models")
        logger.info("Starting traffic worker thread...")
        traffic_thread = threading.Thread(target=traffic_worker, daemon=True)
        traffic_thread.start()
        logger.info("Traffic worker thread started")
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")

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
        "last_update": last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None
    })

@app.route('/live-densities')
def get_live_densities():
    try:
        densities_data = download_json_from_drive(DENSITIES_FILE)
        if not density_data:
            return jsonify({
                "error": "No density data available yet",
                "message": "Please wait for the first calculation cycle"
            }), 404
        density_data["last_update"] = last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None
        density_data["update_interval"] = "30 seconds"
        if last_update:
            next_update = last_update + timedelta(seconds=30)
            time_until_next = next_update - datetime.now()
            density_data["next_update_in"] = f"{int(time_until_next.total_seconds())} seconds" if time_until_next.total_seconds() > 0 else "Updating now..."
        return jsonify(density_data)
    except Exception as e:
        logger.error(f"Error reading live densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/live-vehicleCounts')
def get_live_vehicle_counts():
    try:
        vehicle_counts = download_json_from_drive(VEHICLE_COUNTS_FILE)
        if not vehicle_counts:
            return jsonify({
                "error": "No vehicle count data available yet",
                "message": "Please wait for the first calculation cycle"
            }), 404
        vehicle_counts["last_update"] = last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None
        vehicle_counts["update_interval"] = "30 seconds"
        if last_update:
            next_update = last_update + timedelta(seconds=30)
            time_until_next = next_update - datetime.now()
            vehicle_counts["next_update_in"] = f"{int(time_until_next.total_seconds())} seconds" if time_until_next.total_seconds() > 0 else "Updating now..."
        return jsonify(vehicle_counts)
    except Exception as e:
        logger.error(f"Error reading live vehicle counts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/today-vehicleCounts')
def get_today_vehicle_counts():
    try:
        today_vehicle_counts = download_json_from_drive(TODAY_VEHICLE_COUNTS_FILE)
        if not today_vehicle_counts:
            manage_historical_vehicle_counts()
            today_vehicle_counts = download_json_from_drive(TODAY_VEHICLE_COUNTS_FILE)
        return jsonify(today_vehicle_counts)
    except Exception as e:
        logger.error(f"Error reading today's vehicle counts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/yesterday-vehicleCounts')
def get_yesterday_vehicle_counts():
    try:
        yesterday_vehicle_counts = download_json_from_drive(YESTERDAY_VEHICLE_COUNTS_FILE)
        if not yesterday_vehicle_counts:
            return jsonify({
                "message": "No yesterday data available yet",
                "date": None,
                "counts_by_time": {}
            })
        return jsonify(yesterday_vehicle_counts)
    except Exception as e:
        logger.error(f"Error reading yesterday's vehicle counts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/day_before_yesterday-vehicleCounts')
def get_day_before_yesterday_vehicle_counts():
    try:
        day_before_yesterday_vehicle_counts = download_json_from_drive(DAY_BEFORE_YESTERDAY_VEHICLE_COUNTS_FILE)
        if not day_before_yesterday_vehicle_counts:
            return jsonify({
                "message": "No day before yesterday data available yet",
                "date": None,
                "counts_by_time": {}
            })
        return jsonify(day_before_yesterday_vehicle_counts)
    except Exception as e:
        logger.error(f"Error reading day before yesterday's vehicle counts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/critical-vehicleCounts')
def get_critical_vehicle_counts():
    try:
        critical_vehicle_counts = download_json_from_drive(CRITICAL_VEHICLE_COUNTS_FILE)
        if not critical_vehicle_counts:
            manage_historical_vehicle_counts()
            critical_vehicle_counts = download_json_from_drive(CRITICAL_VEHICLE_COUNTS_FILE)
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Critical vehicle count thresholds based on yesterday's maximum values",
            "critical_counts": critical_vehicle_counts
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error reading critical vehicle counts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/day_before_yesterday-critical-vehicleCounts')
def get_day_before_yesterday_critical_vehicle_counts():
    try:
        critical_vehicle_counts = download_json_from_drive(DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE)
        if not critical_vehicle_counts:
            manage_historical_vehicle_counts()
            critical_vehicle_counts = download_json_from_drive(DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE)
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Critical vehicle count thresholds based on day before yesterday's maximum values",
            "critical_counts": critical_vehicle_counts
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error reading day before yesterday critical vehicle counts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "status": "running",
        "memory_optimized": True,
        "version": "1.0",
        "using_models": USE_MODELS,
        "total_cameras": len(cameras),
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health')
def health_check():
    try:
        densities_exists = bool(get_file_id(DENSITIES_FILE))
        vehicle_counts_exists = bool(get_file_id(VEHICLE_COUNTS_FILE))
        today_vehicle_counts_exists = bool(get_file_id(TODAY_VEHICLE_COUNTS_FILE))
        yesterday_vehicle_counts_exists = bool(get_file_id(YESTERDAY_VEHICLE_COUNTS_FILE))
        day_before_yesterday_exists = bool(get_file_id(DAY_BEFORE_YESTERDAY_VEHICLE_COUNTS_FILE))
        critical_vehicle_exists = bool(get_file_id(CRITICAL_VEHICLE_COUNTS_FILE))
        day_before_critical_exists = bool(get_file_id(DAY_BEFORE_YESTERDAY_CRITICAL_VEHICLE_COUNTS_FILE))
        return jsonify({
            "status": "healthy",
            "storage": {
                "backend": "Google Drive",
                "folder_id": FOLDER_ID,
                "densities_exists": density_exists,
                "vehicle_counts_exists": vehicle_counts_exists,
                "today_vehicle_counts_exists": today_vehicle_counts_exists,
                "yesterday_vehicle_counts_exists": yesterday_vehicle_counts_exists,
                "day_before_yesterday_counts_exists": day_before_yesterday_exists,
                "critical_vehicle_counts_exists": critical_vehicle_exists,
                "day_before_yesterday_critical_exists": day_before_critical_exists
            },
            "using_models": USE_MODELS,
            "last_update": last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/refresh')
def refresh_traffic_data():
    try:
        density_data, vehicle_data = process_traffic_data()
        return jsonify({
            "status": "success",
            "message": "Traffic data refreshed successfully",
            "density_timestamp": density_data["timestamp"],
            "vehicle_count_timestamp": vehicle_data["timestamp"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to refresh traffic data: {str(e)}"
        }), 500

@app.route('/debug')
def debug():
    try:
        model_info = {
            "road_model": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras"))},
            "vehicle_model": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth"))}
        }
        env_vars = {
            "USE_MODELS": USE_MODELS,
            "BASE_DIR": os.environ.get('BASE_DIR', 'not set'),
            "GOOGLE_DRIVE_FOLDER_ID": FOLDER_ID
        }
        storage_info = {
            "backend": "Google Drive",
            "folder_id": FOLDER_ID,
            "densities_exists": bool(get_file_id(DENSITIES_FILE)),
            "vehicle_counts_exists": bool(get_file_id(VEHICLE_COUNTS_FILE)),
            "today_vehicle_counts_exists": bool(get_file_id(TODAY_VEHICLE_COUNTS_FILE))
        }
        model_load_status = {
            "road_model_loaded": _road_model is not None,
            "vehicle_model_loaded": _vehicle_model is not None,
            "dependencies_loaded": all(x is not None for x in [_tf, _cv2, _np, _torch, _transforms])
        }
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_files": model_info,
            "model_load_status": model_load_status,
            "environment_variables": env_vars,
            "storage_info": storage_info,
            "cameras": len(cameras),
            "last_update": last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug-model')
def debug_model():
    try:
        if not load_dependencies():
            return jsonify({"error": "Dependencies not loaded"}), 500
        if _road_model is None or _vehicle_model is None:
            return jsonify({"error": "Models not loaded"}), 500
        road_success, road_error = False, None
        vehicle_success, vehicle_error = False, None
        try:
            test_input_tf = np.zeros((1, 128, 128, 3), dtype=np.float32)
            _road_model.predict(test_input_tf, verbose=0)
            road_success = True
        except Exception as e:
            road_error = str(e)
        try:
            test_input_torch = torch.zeros((1, 3, 384, 384), dtype=torch.float32).to(DEVICE)
            with _torch.no_grad():
                _vehicle_model(test_input_torch)
            vehicle_success = True
        except Exception as e:
            vehicle_error = str(e)
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "models_loaded": {"road_model": _road_model is not None, "vehicle_model": _vehicle_model is not None},
            "test_prediction": {
                "road_test": {"success": road_success, "error": road_error},
                "vehicle_test": {"success": vehicle_success, "error": vehicle_error}
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Google Drive initialization failed. Exiting.")
        exit(1)
    manage_historical_vehicle_counts()
    start_worker()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
