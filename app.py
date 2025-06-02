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
import numpy as np
import cv2
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests

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

# File names for density and vehicle count data
TODAY_DENSITIES_FILE = "today_densities.json"
YESTERDAY_DENSITIES_FILE = "yesterday_densities.json"
CRITICAL_DENSITIES_FILE = "critical_densities.json"
OUTPUT_JSON_FILE = "densities.json"
VEHICLE_COUNTS_FILE = "vehicle_counts.json"

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
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf6&camLocation=N%C3%BUt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf7&camLocation=N%C3%BUt%20giao%20Ng%C3%A3%20s%C3%A1u%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf2&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20-%20C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf9&camLocation=N%C3%BUt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8',
    'http://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acfa&camLocation=N%C3%BUt%20giao%20C%C3%B4ng%20Tr%C6%B0%E1%BB%9Dng%20D%C3%A2n%20Ch%E1%BB%A7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8'
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
CAMERA_URL_TEMPLATE = base_url + "?id={camera_id}&bg=black&w=300&h=230"

# Lazy-load TensorFlow and other dependencies
_tf, _cv2, _np, _requests, _road_model, _vehicle_model, _session, _torch = [None] * 8
_pytorch_vehicle_model = None
USE_MODELS = os.environ.get('USE_MODELS', 'false').lower() == 'true'
last_density_update = None
last_vehicle_count_update = None

def load_dependencies():
    global _tf, _cv2, _np, _requests, _session, _torch
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
            _tf, _cv2, _np, _requests, _torch = tf, cv2, np, requests, torch
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

# MiniUNet class for vehicle detection
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

def load_pytorch_vehicle_model(model_path, device='cpu'):
    try:
        if not load_dependencies():
            logger.error("Failed to load dependencies for PyTorch model")
            return None
        logger.info(f"Loading PyTorch vehicle model from: {model_path}")
        model = MiniUNet(in_channels=3, out_channels=1).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("PyTorch vehicle model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading PyTorch vehicle model: {e}")
        return None

def predict_vehicles_pytorch(model, image, device='cpu', img_size=384):
    try:
        if not load_dependencies() or model is None or image is None:
            logger.error("Cannot predict vehicles: dependencies or model or image missing")
            return None
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        image_tensor = transform(image_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            vehicle_pred = model(image_tensor)
        
        vehicle_mask = vehicle_pred.squeeze().cpu().numpy()
        vehicle_mask_resized = cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]))
        return vehicle_mask_resized
    except Exception as e:
        logger.error(f"Error predicting vehicles with PyTorch model: {e}")
        return None

def estimate_vehicle_count_from_blobs(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    
    if not significant_blobs:
        return 0, 0
    
    def find_vehicle_unit_size(sizes):
        sizes = sorted(sizes)
        smallest_blob = min(sizes)
        median_blob = np.median(sizes)
        
        if smallest_blob > 1500:
            logger.info("Only large blobs detected - using fixed vehicle size estimate")
            return 200
        
        q5, q95 = np.percentile(sizes, [5, 95])
        filtered_sizes = [s for s in sizes if q5 <= s <= q95]
        
        if not filtered_sizes:
            filtered_sizes = sizes
        
        single_vehicle_candidates = [s for s in filtered_sizes if s <= np.percentile(filtered_sizes, 25)]
        
        if single_vehicle_candidates and min(single_vehicle_candidates) <= 1200:
            unit_size = np.median(single_vehicle_candidates)
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
    
    sizes = np.array(significant_blobs)
    
    smallest_blob = min(sizes)
    
    if smallest_blob > 1500:
        logger.info("Statistical method: Only large blobs - using fixed 200px vehicle size")
        avg_single_vehicle = 200
        vehicle_count = 0
        for blob_size in sizes:
            vehicles_in_blob = max(1, int(blob_size / avg_single_vehicle))
            vehicle_count += vehicles_in_blob
        return vehicle_count, int(avg_single_vehicle)
    
    q20, q50, q80 = np.percentile(sizes, [20, 50, 80])
    iqr = q80 - q20
    
    single_vehicle_upper = q20 + 0.3 * iqr
    single_vehicle_blobs = sizes[sizes <= single_vehicle_upper]
    multi_vehicle_blobs = sizes[sizes > single_vehicle_upper]
    
    if len(single_vehicle_blobs) > 0:
        avg_single_vehicle = np.median(single_vehicle_blobs)
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
    
    sizes = np.array(significant_blobs)
    
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
        hist, bin_edges = np.histogram(sizes, bins=n_bins)
        peak_bin_idx = np.argmax(hist)
        peak_range = (bin_edges[peak_bin_idx], bin_edges[peak_bin_idx + 1])
        peak_sizes = sizes[(sizes >= peak_range[0]) & (sizes <= peak_range[1])]
        
        if len(peak_sizes) > 0:
            typical_vehicle_size = np.median(peak_sizes)
        else:
            typical_vehicle_size = np.percentile(sizes, 20)
    else:
        typical_vehicle_size = np.median(sizes)
    
    typical_vehicle_size = max(200, min(typical_vehicle_size, 1500))
    
    total_vehicles = 0
    for size in sizes:
        vehicles_in_blob = max(1, int(size / typical_vehicle_size))
        total_vehicles += vehicles_in_blob
    
    return int(total_vehicles), int(typical_vehicle_size)

def apply_greenshields_model(vehicle_count, road_area_pixels, image_shape):
    if road_area_pixels == 0 or vehicle_count == 0:
        return 0, 0, "No Traffic"
    
    image_area = image_shape[0] * image_shape[1]
    road_ratio = road_area_pixels / image_area
    density_metric = vehicle_count / (road_ratio * 1000)
    
    free_flow_speed = 60
    jam_density = 80
    
    if jam_density > 0:
        speed_ratio = max(0, 1 - (density_metric / jam_density))
        estimated_speed = free_flow_speed * speed_ratio
    else:
        estimated_speed = 0
    
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

def load_road_model(model_path):
    try:
        if not load_dependencies():
            logger.error("Failed to load dependencies for road model")
            return None
        logger.info(f"Loading TensorFlow road model from: {model_path}")
        road_model = tf.keras.models.load_model(model_path)
        logger.info("TensorFlow road model loaded successfully")
        return road_model
    except Exception as e:
        logger.error(f"Error loading TensorFlow road model: {e}")
        return None

def process_camera_for_vehicle_count(camera_id, camera_name, road_model, vehicle_model, device='cpu'):
    try:
        if not load_dependencies():
            return None
        
        image = fetch_camera_image(camera_id)
        if image is None:
            logger.warning(f"Failed to fetch image for {camera_name}")
            return None
        
        # Step 1: Road Segmentation
        IMG_HEIGHT, IMG_WIDTH = 128, 128
        img_road = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        img_road = np.expand_dims(img_road, axis=0)
        
        road_pred = road_model.predict(img_road, verbose=0)
        road_mask = (road_pred.squeeze() > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(road_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            hull = cv2.convexHull(smoothed_contour)
            refined_road_mask = np.zeros_like(mask_resized, dtype=np.uint8)
            cv2.fillPoly(refined_road_mask, [hull], 255)
        else:
            refined_road_mask = mask_resized.copy()
        
        road_pixels = np.count_nonzero(refined_road_mask)
        
        # Step 2: Vehicle Detection
        vehicle_mask = predict_vehicles_pytorch(vehicle_model, image, device, img_size=384)
        if vehicle_mask is None:
            logger.warning(f"Vehicle prediction failed for {camera_name}")
            return None
        
        vehicle_threshold = 0.25
        binary_vehicle_mask = (vehicle_mask > vehicle_threshold).astype(np.uint8)
        
        # Step 3: Calculate Vehicle Pixels on Road
        road_binary = (refined_road_mask > 0).astype(np.uint8)
        vehicles_on_road = np.logical_and(binary_vehicle_mask, road_binary).astype(np.uint8)
        vehicle_pixels_on_road = np.count_nonzero(vehicles_on_road)
        total_vehicle_pixels = np.count_nonzero(binary_vehicle_mask)
        
        if road_pixels > 0:
            density_percentage = (vehicle_pixels_on_road / road_pixels) * 100
        else:
            density_percentage = 0
        
        # Step 4: Vehicle Counting
        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        vehicles_on_road_cleaned = cv2.morphologyEx(vehicles_on_road, cv2.MORPH_OPEN, kernel_open, iterations=1)
        vehicles_on_road_cleaned = cv2.morphologyEx(vehicles_on_road_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            vehicles_on_road_cleaned, connectivity=8
        )
        
        blob_sizes = []
        min_reasonable_blob = 500
        max_reasonable_blob = 8000
        
        for i in range(1, num_labels):
            blob_size = stats[i, cv2.CC_STAT_AREA]
            if min_reasonable_blob <= blob_size <= max_reasonable_blob:
                blob_sizes.append(blob_size)
        
        if blob_sizes:
            method1_count, method1_unit = estimate_vehicle_count_from_blobs(blob_sizes, min_blob_size=500)
            method2_count, method2_unit = estimate_vehicles_statistical_clustering(blob_sizes, min_blob_size=500)
            method3_count, method3_unit = estimate_vehicles_histogram_analysis(blob_sizes, min_blob_size=500)
            
            all_counts = [method1_count, method2_count, method3_count]
            estimated_vehicle_count = min(all_counts)
            
            if estimated_vehicle_count == method1_count:
                avg_vehicle_size = method1_unit
            elif estimated_vehicle_count == method2_count:
                avg_vehicle_size = method2_unit
            else:
                avg_vehicle_size = method3_unit
        else:
            estimated_vehicle_count = 0
            avg_vehicle_size = 0
        
        # Step 5: Greenshields Model
        estimated_speed, density_metric, traffic_level = apply_greenshields_model(
            estimated_vehicle_count, road_pixels, image.shape
        )
        
        return {
            'camera_name': camera_name,
            'road_pixels': road_pixels,
            'vehicle_pixels': total_vehicle_pixels,
            'vehicle_pixels_on_road': vehicle_pixels_on_road,
            'blob_count': len(blob_sizes),
            'blob_sizes': blob_sizes,
            'estimated_vehicle_count': estimated_vehicle_count,
            'avg_vehicle_size': avg_vehicle_size,
            'density_percentage': density_percentage,
            'density_metric': density_metric,
            'estimated_speed': estimated_speed,
            'traffic_level': traffic_level
        }
    except Exception as e:
        logger.error(f"Error processing vehicle count for {camera_name}: {e}")
        return None

def store_vehicle_counts(timestamp_str, counts_data):
    try:
        vehicle_counts = download_json_from_drive(VEHICLE_COUNTS_FILE) or {}
        if 'counts_by_time' not in vehicle_counts:
            vehicle_counts['counts_by_time'] = {}
        vehicle_counts['counts_by_time'][timestamp_str] = counts_data
        upload_json_to_drive(VEHICLE_COUNTS_FILE, vehicle_counts)
        logger.info("Stored vehicle counts in Google Drive")
    except Exception as e:
        logger.error(f"Error storing vehicle counts: {e}")

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
        _road_model = tf.keras.models.load_model(road_model_path)
        time.sleep(1)
        logger.info("Loading vehicle detection model...")
        _vehicle_model = tf.keras.models.load_model(vehicle_model_path)
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

def update_critical_densities(density_data):
    try:
        critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE) or {}
        density_by_time = density_data.get('densities_by_time', {})
        for camera_code in camera_mapping.values():
            max_density = 0.0
            for timestamp, cameras_data in density_by_time.items():
                if camera_code in cameras_data:
                    density = cameras_data[camera_code].get('density', 0.0)
                    max_density = max(max_density, density)
            if camera_code not in critical_densities or max_density > critical_densities[camera_code]:
                critical_densities[camera_code] = max_density
                logger.info(f"Updated critical density for {camera_code}: {max_density}")
        upload_json_to_drive(CRITICAL_DENSITIES_FILE, critical_densities)
        logger.info("Successfully updated critical density")
    except Exception as e:
        logger.error(f"Error updating critical density: {e}")

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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://giaothong.hochiminhcity.gov.vn/"
        })
        _session.get("https://giaothong.hochiminhcity.gov.vn/", timeout=10)
        url = CAMERA_URL_TEMPLATE.format(camera_id=camera_id)
        logger.info(f"Fetching image from {url}")
        response = _session.get(url, timeout=30)
        response.raise_for_status()
        image_array = _np.asarray(bytearray(response.content), dtype=np.uint8)
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

def camera_detection(camera_id):
    if not load_dependencies() or not USE_MODELS:
        return 0
    try:
        image = fetch_camera_image(camera_id)
        if image is None:
            return 0
        result = process_camera_for_vehicle_count(camera_id, "Camera", _road_model, _pytorch_vehicle_model)
        return result['estimated_vehicle_count'] if result else 0
    except Exception as e:
        logger.error(f"Error in camera_detection: {e}")
        return 0

def analyze_image(image):
    if not load_dependencies() or image is None:
        return {"density": 0.0, "success": False}
    try:
        if _road_model is None or _vehicle_model is None:
            logger.warning("Models not loaded, using fallback values")
            return {"density": 0.0, "success": False}
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"density": 0.0, "success": False}
        input_tensor = _tf.convert_to_tensor(processed_image, dtype=tf.float32)
        try:
            vehicle_prediction = _vehicle_model(input_tensor)
            vehicle_output_np = vehicle_prediction.numpy()
            logger.info(f"Vehicle output shape: {vehicle_output_np.shape}")
            if vehicle_output_np.shape[-1] == 12:
                weights = [0.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05]
                weighted_sum = sum(float(_np.mean(vehicle_output_np[..., i])) * weights[i] for i in range(1, 12))
                density = max(0, min(100, weighted_sum * 100))
            else:
                density = float(_np.mean(vehicle_output_np) * 100)
            logger.info(f"Calculated density: {density}")
            return {"density": round(density, 2), "success": True}
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            density = round(_np.random.normal(50, 20), 1)
            density = max(0, min(100, density))
            return {"density": density, "success": False}
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {"density": 0.0, "success": False}

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
    last_density_update = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    results = {"timestamp": timestamp_str, "cameras": {}}
    success_count = []
    failure_count = []
    for camera_id, camera_name in cameras:
        try:
            logger.info(f"Processing camera {camera_name}...")
            camera_code = camera_mapping.get(camera_name, camera_name)
            image = fetch_camera_image(camera_id)
            if image is None:
                logger.warning(f"Failed to fetch image for {camera_name}")
                results["cameras"][camera_code] = {"density": 0.0, "success": False}
                failure_count.append(1)
                continue
            density_data = analyze_image(image)
            results["cameras"][camera_code] = density_data
            store_today_density(timestamp_str, camera_code, density_data)
            success_count.append(1)
            logger.info(f"Successfully processed {camera_name}: {density_data['density']} density")
        except Exception as e:
            logger.error(f"Error processing camera {camera_name}: {str(e)}")
            results["cameras"][camera_code] = {"density": 0.0, "success": False}
            failure_count.append(1)
    logger.info(f"Processing complete. Success: {len(success_count)}, Failures: {len(failure_count)}")
    return results

def density_worker():
    logger.info("Density worker started - running every 30 seconds")
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
                time.sleep(50)
    except Exception as e:
        logger.error(f"Critical error in density worker: {e}")

def start_worker():
    try:
        logger.info("Starting worker - loading models...")
        load_success = load_models()
        vehicle_count_success = load_vehicle_counting_models()
        if load_success and vehicle_count_success:
            logger.info("All models loaded successfully!")
        else:
            logger.error("Failed to load some models! Check logs for details.")
            if not load_dependencies():
                logger.error("Problem: Dependencies not loaded")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation_tf")):
                logger.error("Problem: Road model (TF) not found")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_multi_classV1_tf")):
                logger.error("Problem: Vehicle model (TF) not found")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras")):
                logger.error("Problem: Road model (Keras) not found")
            elif not os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth")):
                logger.error("Problem: Vehicle model (PyTorch) not found")
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

def date_transition_worker():
    while True:
        now = datetime.now()
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
        seconds_until_midnight = (midnight - now).total_seconds()
        time.sleep(seconds_until_midnight + 1)
        check_new_day()
        logger.info(f"Date transition completed at {datetime.now().strftime('%Y-%m-%d')}")

def load_vehicle_counting_models():
    global _pytorch_vehicle_model, _road_model
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    road_model_path = os.path.join(base_directory, "unet_road_segmentation (Better).keras")
    vehicle_model_path = os.path.join(base_directory, "filtered_model_cpu.pth")
    
    if not os.path.exists(road_model_path):
        logger.error(f"Road model file not found: {road_model_path}")
        return False
    if not os.path.exists(vehicle_model_path):
        logger.error(f"Vehicle model file not found: {vehicle_model_path}")
        return False
    
    _road_model = load_road_model(road_model_path)
    _pytorch_vehicle_model = load_pytorch_vehicle_model(vehicle_model_path, device='cpu')
    
    if _road_model is None or _pytorch_vehicle_model is None:
        logger.error("Failed to load one or both vehicle counting models")
        return False
    
    return True

@app.route('/')
def index():
    return jsonify({
        "status": "healthy",
        "version": "1.1",
        "message": "Vehicle Detection Service is operational",
        "using_models": USE_MODELS,
        "last_density_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
    })

@app.route('/cameras')
def get_cameras():
    try:
        cameras_info = [
            {
                "code": chr(65 + idx),
                "id": camera_id,
                "name": camera_name,
                "url": camera_websites[idx] if idx < len(camera_websites) else None
            }
            for idx, (camera_id, camera_name) in enumerate(cameras)
        ]
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_cameras": len(cameras_info),
            "cameras": cameras_info
        })
    except Exception as e:
        logger.error(f"Error getting cameras: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve camera information",
            "details": str(e)
        }), 500

@app.route('/live-densities')
def get_live_densities():
    try:
        densities = download_json_from_drive(TODAY_DENSITIES_FILE)
        if not densities:
            return jsonify({
                "error": "No density data available",
                "message": "Please wait for the next calculation cycle"
            }), 503
        density["last_updated"] = last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
        density["update_interval"] = "30 seconds"
        if last_density_update:
            next_update = last_density_update + timedelta(seconds=30)
            time_until_next = next_update - datetime.now()
            density["next_update_in"] = f"{int(time_until_next.total_seconds())} seconds" if time_until_next.total_seconds() > 0 else "Updating now"
        return jsonify(density)
    except Exception as e:
        logger.error(f"Error getting live densities: {str(e)}")
        return jsonify({"error": "Failed to get live densities", "details": str(e)}), 500

@app.route('/today-densities')
def get_today_densities():
    try:
        today_densities = download_json_from_drive(TODAY_DENSITIES_FILE)
        if not today_densities:
            manage_historical_densities()
            today_densities = download_json_from_drive(TODAY_DENSITIES_FILE)
        return jsonify(today_densities)
    except Exception as e:
        logger.error(f"Error getting today's densities: {str(e)}")
        return jsonify({"error": "Failed to retrieve today's density data", "details": str(e)}), 500

@app.route('/yesterday-densities')
def get_yesterday_densities():
    try:
        yesterday_densities = download_json_from_drive(YESTERDAY_DENSITIES_FILE)
        if not yesterday_densities:
            return jsonify({
                "message": "No yesterday data available",
                "date": None,
                "densities_by_time": {}
            })
        return jsonify(yesterday_densities)
    except Exception as e:
        logger.error(f"Error getting yesterday's densities: {str(e)}")
        return jsonify({"error": "Failed to get yesterday's density", "details": str(e)}), 500

@app.route('/critical-densities')
def get_critical_densities():
    try:
        critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE)
        if not critical_densities:
            manage_historical_densities()
            critical_densities = download_json_from_drive(CRITICAL_DENSITIES_FILE)
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Critical density thresholds based on historical data",
            "critical_densities": critical_densities
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting critical densities: {str(e)}")
        return jsonify({"error": "Failed to get critical density data", "details": str(e)}), 500

@app.route('/densities')
def get_densities():
    try:
        densities = download_json_from_drive(TODAY_DENSITIES_FILE)
        if not density:
            manage_historical_densities()
            density = download_json_from_drive(TODAY_DENSITIES_FILE)
        
        raw_densities = {}
        for timestamp, cameras_data in density.get('densities_by_time', {}).items():
            for camera_code, data in cameras_data.items():
                if camera_code not in raw_densities:
                    raw_densities[camera_code] = []
                raw_densities[camera_code].append({
                    "timestamp": timestamp,
                    "density": data.get("density", 0.0)
                })
        
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "densities": raw_densities
        })
    except Exception as e:
        logger.error(f"Error getting densities: {str(e)}")
        return jsonify({"error": "Failed to retrieve density data", "details": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "status": "healthy",
        "memory_optimized": True,
        "version": "1.1",
        "using_models": USE_MODELS,
        "last_density_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None,
        "total_cameras": len(cameras),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health')
def health_check():
    try:
        today_exists = bool(get_file_id(TODAY_DENSITIES_FILE))
        yesterday_exists = bool(get_file_id(YESTERDAY_DENSITIES_FILE))
        critical_densities_exists = bool(get_file_id(CRITICAL_DENSITIES_FILE))
        output_exists = bool(get_file_id(OUTPUT_JSON_FILE))
        
        return jsonify({
            "status": "healthy",
            "storage": {
                "backend": "Google Drive",
                "folder_id": FOLDER_ID,
                "today_densities_exists": today_exists,
                "yesterday_densities_exists": yesterday_exists,
                "critical_densities_exists": critical_densities_exists,
                "output_json_exists": output_exists
            },
            "using_models": USE_MODELS,
            "last_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "details": str(e)
        }), 500

@app.route('/refresh')
def refresh_densities():
    try:
        result = fetch_and_process_densities()
        return jsonify({
            "status": "successful",
            "message": "Successfully refreshed density data",
            "timestamp": result["timestamp"]
        })
    except Exception as e:
        logger.error(f"Error refreshing densities: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to refresh density data: {str(e)}",
            "details": str(e)
        }), 500

@app.route('/debug')
def debug():
    try:
        model_info = {
            "unet_road_segmentation_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation_tf"))},
            "unet_multi_classV1_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_multi_classV1_tf"))},
            "unet_road_segmentation (Better).keras": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras"))},
            "filtered_model_cpu.pth": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth"))}
        }
        env_vars = {
            "USE_MODELS": os.environ.get('USE_MODELS', 'not set'),
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
            "output_json_exists": bool(get_file_id(OUTPUT_JSON_FILE)),
            "vehicle_counts_exists": bool(get_file_id(VEHICLE_COUNTS_FILE))
        }
        
        try:
            files_in_base_dir = os.listdir(os.environ.get('BASE_DIR', os.getcwd()))
        except Exception as e:
            files_in_base_dir = f"Error listing files: {str(e)}"
            
        model_load_status = {
            "road_model_tf_loaded": _road_model is not None,
            "vehicle_model_tf_loaded": _vehicle_model is not None,
            "pytorch_vehicle_model_loaded": _pytorch_vehicle_model is not None,
            "dependencies_loaded": _tf is not None and _cv2 is not None and _np is not None and _requests is not None and _torch is not None
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
            memory_info = f"Error getting memory info: {str(e)}"
            
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_files": model_info,
            "model_load_status": model_load_status,
            "environment_variables": env_vars,
            "base_directory": os.environ.get('BASE_DIR', os.getcwd()),
            "files_in_base_directory": files_in_base_dir,
            "storage": storage_info,
            "system_resources": memory_info,
            "cameras": len(cameras),
            "last_density_update": last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None,
            "last_vehicle_count_update": last_vehicle_count_update
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({
            "error": "Debug information collection failed",
            "details": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/load-models')
def force_load_models():
    try:
        load_success = load_models()
        vehicle_count_success = load_vehicle_counting_models()
        road_loaded = _road_model is not None
        vehicle_tf_loaded = _vehicle_model is not None
        vehicle_pytorch_loaded = _pytorch_vehicle_model is not None
        
        status = {
            "load_success": {
                "density_models": load_success,
                "vehicle_counting_models": vehicle_count_success
            },
            "models_loaded": {
                "road_model": road_loaded,
                "vehicle_model_tf": vehicle_tf_loaded,
                "vehicle_model_pytorch": vehicle_pytorch_loaded
            },
            "model_files": {
                "road_model_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation_tf"))},
                "vehicle_model_tf": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_multi_classV1_tf"))},
                "road_model_counting": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "unet_road_segmentation (Better).keras"))},
                "vehicle_model_pytorch": {"exists": os.path.exists(os.path.join(os.environ.get('BASE_DIR', os.getcwd()), "filtered_model_cpu.pth"))}
            },
            "environment": {
                "USE_MODELS": USE_MODELS,
                "BASE_DIR": os.environ.get('BASE_DIR', os.getcwd())
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(status)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": error_details,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/debug-model')
def debug_model():
    try:
        if not load_dependencies():
            return jsonify({"error": "Dependencies not loaded", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500
        if _road_model is None or _vehicle_model is None:
            return jsonify({"error": "TF models not loaded", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500
        
        test_input = _np.zeros((1, 128, 128, 3), dtype='float32')
        tf_input = _tf.convert_to_tensor(test_input, dtype=_tf.float32)
        
        road_success, vehicle_success = False, False
        road_error, vehicle_error = None, None
        
        try:
            _road_model.predict(tf_input, verbose=0)
            road_success = True
        except Exception as e:
            road_error = str(e)
        
        try:
            _vehicle_model.predict(tf_input, verbose=0)
            vehicle_success = True
        except Exception as e:
            vehicle_error = str(e)
        
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "models_loaded": {
                "road_model_tf": _road_model is not None,
                "vehicle_model_tf": _vehicle_model is not None,
                "vehicle_model_pytorch": _pytorch_vehicle_model is not None
            },
            "test_prediction": {
                "road_tf": {"success": road_success, "error": road_error},
                "vehicle_tf": {"success": vehicle_success, "error": vehicle_error}
            }
        })
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500

@app.route('/camera-status')
def check_camera_status():
    try:
        results = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "cameras": []}
        if not load_dependencies():
            return jsonify({"error": "Failed to load dependencies", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500
        
        for camera_id, camera_name in cameras:
            camera_code = camera_mapping.get(camera_name, camera_name)
            try:
                logger.info(f"Checking camera {camera_name}...")
                image = fetch_camera_image(camera_id)
                if image is None:
                    results["cameras"].append({"name": camera_name, "code": camera_code, "status": "offline", "error": "Failed to fetch image"})
                else:
                    results["cameras"].append({"name": camera_name, "code": camera_code, "status": "online", "resolution": f"{image.shape[1]}x{image.shape[0]}"})
            except Exception as e:
                results["cameras"].append({"name": camera_name, "code": camera_code, "status": "error", "error": str(e)})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}), 500

@app.route('/count_vehicles')
def count_vehicles():
    global last_vehicle_count_update
    try:
        if not load_vehicle_counting_models():
            return jsonify({
                "error": "Failed to load vehicle counting models",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 500
        
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        last_vehicle_count_update = timestamp_str
        results = {"timestamp": timestamp_str, "cameras": {}, "counts_summary": ""}
        counts_data = {}
        
        success_count = []
        failure_count = []
        
        for camera_id, camera_name in cameras:
            try:
                camera_code = camera_mapping.get(camera_name, camera_name)
                logger.info(f"Processing vehicle count for camera {camera_name} ({camera_code})")
                
                result = process_camera_for_vehicle_count(camera_id, camera_name, _road_model, _pytorch_vehicle_model)
                
                if result is None:
                    failure_count.append(1)
                    results["cameras"][camera_code] = {
                        "name": camera_name,
                        "error": "Failed to process image",
                        "success": False
                    }
                    counts_data[camera_code] = {"success": False, "error": "Failed to process image"}
                else:
                    success_count.append(1)
                    results["cameras"][camera_code] = {
                        "name": camera_name,
                        "success": True,
                        "road_pixels": result["road_pixels"],
                        "vehicle_pixels": result["vehicle_pixels"],
                        "vehicle_pixels_on_road": result["vehicle_pixels_on_road"],
                        "blob_count": result["blob_count"],
                        "blob_sizes": result["blob_sizes"],
                        "estimated_vehicle_count": result["estimated_vehicle_count"],
                        "avg_vehicle_size": result["avg_vehicle_size"],
                        "density_percentage": round(result["density_percentage"], 2),
                        "density_metric": round(result["density_metric"], 2),
                        "estimated_speed": round(result["estimated_speed"], 1),
                        "traffic_level": result["traffic_level"]
                    }
                    counts_data[camera_code] = results["cameras"][camera_code]
                
                logger.info(f"Processed {camera_name}: {result['estimated_vehicle_count'] if result else 0} vehicles")
            except Exception as e:
                logger.error(f"Error processing camera {camera_name}: {str(e)}")
                failure_count.append(1)
                results["cameras"][camera_code] = {
                    "name": camera_name,
                    "error": str(e),
                    "success": False
                }
                counts_data[camera_code] = {"success": False, "error": str(e)}
        
        counts_summary = "; ".join([f"{camera_code}: {counts_data[camera_code]['estimated_vehicle_count']}" if counts_data[camera_code].get("success", False) else f"{camera_code}: 0" for camera_code in counts_data])
        results["counts_summary"] = counts_summary
        
        try:
            store_vehicle_counts(timestamp_str, counts_data)
        except Exception as e:
            logger.error(f"Error saving vehicle counts: {str(e)}")
        
        results["success_count"] = len(success_count)
        results["failure_count"] = len(failure_count)
        
        logger.info(f"Vehicle count completed. Success: {len(success_count)}, Failed: {len(failure_count)}")
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in count_vehicles endpoint: {str(e)}")
        return jsonify({
            "error": "Failed to count vehicles",
            "details": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

if __name__ == "__main__":
    init_google_drive()
    if drive_service is None:
        logger.error("Failed to initialize Google Drive. Exiting.")
        exit(1)
    manage_historical_densities()
    date_transition_thread = threading.Thread(target=date_transition_worker, daemon=True)
    date_transition_thread.start()
    start_worker()
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
