import os
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
import requests
import gc

# Initialize Flask
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera configurations
camera_websites = [
    {'id': 'A', 'camId': '6623e7076f998a001b2523ea', 'title': 'Lý Thái Tổ - Sư Vạn Hạnh'},
    {'id': 'B', 'camId': '5deb576d1dc17d7c5515acf8', 'title': '3/2 – Cao Thắng'},
    {'id': 'C', 'camId': '63ae7a9cbfd3d90017e8f303', 'title': 'Điện Biên Phủ - Cao Thắng'},
    {'id': 'D', 'camId': '5deb576d1dc17d7c5515ad21', 'title': 'Ngã sáu Nguyễn Tri Phương 1'},
    {'id': 'E', 'camId': '5deb576d1dc17d7c5515ad22', 'title': 'Ngã sáu Nguyễn Tri Phương'},
    {'id': 'F', 'camId': '5d8cdd26766c880017188974', 'title': 'Lê Đại Hành 2'},
    {'id': 'G', 'camId': '63ae763bbfd3d90017e8f0c4', 'title': 'Lý Thái Tổ - Nguyễn Đình Chiểu'},
    {'id': 'H', 'camId': '5deb576d1dc17d7c5515acf6', 'title': 'Ngã sáu Cộng hòa 1'},
    {'id': 'I', 'camId': '5deb576d1dc17d7c5515acf7', 'title': 'Ngã sáu Cộng Hòa'},
    {'id': 'J', 'camId': '5deb576d1dc17d7c5515acf2', 'title': 'Điện Biên Phủ - CMT8'},
    {'id': 'K', 'camId': '5deb576d1dc17d7c5515acf9', 'title': 'Nút giao Công Trường Dân Chủ'},
    {'id': 'L', 'camId': '5deb576d1dc17d7c5515acfa', 'title': 'Nút giao Công Trường Dân Chủ 1'}
]

CAMERA_URL_TEMPLATE = 'https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id={camId}&bg=black&w=300&h=230'
WARMUP_URL = "https://giaothong.hochiminhcity.gov.vn/"

# Headers for requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
    "Origin": "https://giaothong.hochiminhcity.gov.vn"
}

# Global variables
_session = None
_tf, _cv2, _np, _requests, _torch, _transforms, _road_model, _vehicle_model = [None] * 8
USE_MODELS = os.environ.get('USE_MODELS', 'false').lower() == 'true'
last_density_update = None
DEVICE = 'cpu'
worker_lock = threading.Lock()

# Parse camera data
cameras = [(c['id'], c['title']) for c in camera_websites]
camera_mapping = {c['id']: c['title'] for c in camera_websites}
logger.info(f"Parsed cameras: {cameras}")

# Dependency and model loading
def load_dependencies():
    global _tf, _cv2, _np, _requests, _torch, _transforms, _session
    if _tf is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            import cv2
            import numpy as np
            import requests
            import torch
            import torchvision.transforms as transforms
            _tf, _cv2, _np, _requests, _torch, _transforms = tf, cv2, np, requests, torch, transforms
            _session = _requests.Session()
            _session.headers.update(headers)
            if _tf and physical_devices := _tf.config.list_physical_devices('GPU'):
                _tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info("Dependencies loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return False
    return True

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

def load_models():
    global _road_model, _vehicle_model
    if not load_dependencies():
        logger.error("Failed to load dependencies")
        return False
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    road_model_path = os.path.join(base_directory, "unet_road_segmentation.keras")
    vehicle_model_path = os.path.join(base_directory, "filtered_model_cpu.pth")
    try:
        logger.info("Loading road segmentation model...")
        _road_model = _tf.keras.models.load_model(road_model_path)
        logger.info("Loading vehicle detection model...")
        _vehicle_model = MiniUNet(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = _torch.load(vehicle_model_path, map_location=DEVICE)
        _vehicle_model.load_state_dict(checkpoint['model_state_dict'])
        _vehicle_model.eval()
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False
    finally:
        gc.collect()

# Image fetching
def fetch_camera_image(camera_id):
    if not load_dependencies():
        return None
    try:
        # Warm-up request
        warmup_response = _session.get(WARMUP_URL, timeout=15)
        warmup_response.raise_for_status()
        logger.info(f"Warm-up request successful: {warmup_response.status_code}")

        # Find camera
        camera = next((c for c in camera_websites if c['id'] == camera_id), None)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return None

        # Fetch image
        url = CAMERA_URL_TEMPLATE.format(camId=camera['camId'])
        for attempt in range(3):
            try:
                response = _session.get(url, timeout=15)
                response.raise_for_status()
                if len(response.content) > 100:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'image' in content_type:
                        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        if image is not None and image.size > 0:
                            logger.info(f"Fetched image for {camera_id}")
                            return image
                        logger.warning(f"Failed to decode image for {camera_id} (attempt {attempt+1}/3)")
                    else:
                        logger.warning(f"Unexpected Content-Type: {content_type} (attempt {attempt+1}/3)")
                else:
                    logger.warning(f"Empty response for {camera_id} (attempt {attempt+1}/3)")
            except Exception as e:
                logger.error(f"Error fetching image for {camera_id}: {e} (attempt {attempt+1}/3)")
            time.sleep(1)
        logger.error(f"Failed to fetch image for {camera_id} after 3 attempts")
        return None
    except Exception as e:
        logger.error(f"Critical error fetching image for {camera_id}: {e}")
        return None
    finally:
        gc.collect()

# Image processing
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
        gc.collect()

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
    total_vehicles = sum(1 if blob_size < unit_vehicle_size * 1.2 else max(1, int(blob_size / unit_vehicle_size)) for blob_size in significant_blobs)
    return int(total_vehicles), int(unit_vehicle_size)

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
    if not load_dependencies() or image is None or _road_model is None or _vehicle_model is None:
        return {
            "density": 0.0,
            "vehicle_count": 0,
            "avg_vehicle_size": 0,
            "density_metric": 0.0,
            "estimated_speed": 0.0,
            "traffic_level": "No Traffic"
        }
    try:
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
        
        estimated_vehicle_count, avg_vehicle_size = estimate_vehicle_count_from_blobs(blob_sizes)
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

# Density processing
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
                logger.warning(f"Image fetch failed for {camera_name}")
                results["cameras"][camera_id] = {**density_data, "error": "Failed to fetch image"}
            else:
                success_count += 1
                analysis_result = analyze_image(image)
                density_data.update(analysis_result)
                results["cameras"][camera_id] = density_data
            logger.info(f"Processed camera {camera_name}: density={density_data['density']}%")
        except Exception as e:
            failure_count += 1
            logger.error(f"Error processing camera {camera_name}: {e}")
            results["cameras"][camera_id] = {**density_data, "error": str(e)}
        finally:
            gc.collect()
    logger.info(f"Processing complete. Success: {success_count}, Failure: {failure_count}")
    return results

def density_worker():
    logger.info("Density worker started")
    try:
        while True:
            with worker_lock:
                logger.info("Starting density processing cycle")
                fetch_and_process_densities()
                logger.info("Density processing cycle completed")
            time.sleep(30)
    except Exception as e:
        logger.error(f"Error in density worker: {e}")

# Flask routes
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "1.0",
        "message": "Traffic Analysis Service is operational",
        "using_models": USE_MODELS
    })

@app.route('/fetch-size/<camera_id>')
def fetch_image_size(camera_id):
    image = fetch_camera_image(camera_id)
    if image is None:
        return jsonify({"success": False, "error": f"Failed to fetch image for {camera_id}"})
    return jsonify({
        "success": True,
        "camera_id": camera_id,
        "image_size_bytes": len(_cv2.imencode('.jpg', image)[1]) if image is not None else 0
    })

@app.route('/live-densities')
def get_live_densities():
    try:
        density = fetch_and_process_densities()
        density["last_update"] = last_density_update.strftime('%Y-%m-%d %H:%M:%S') if last_density_update else None
        density["next_update_in"] = f"{int((last_density_update + timedelta(seconds=30) - datetime.now()).total_seconds())} seconds" if last_density_update else "Updating now..."
        return jsonify(density)
    except Exception as e:
        logger.error(f"Error generating live densities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera-status')
def check_camera_status():
    try:
        results = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "cameras": {}}
        for camera_id, camera_name in cameras:
            image = fetch_camera_image(camera_id)
            results["cameras"][camera_id] = {
                "name": camera_name,
                "status": "online" if image is not None and image.size > 1000 else "offline",
                "resolution": f"{image.shape[1]}x{image.shape[0]}" if image is not None and image.size > 1000 else None,
                "error": None if image is not None else "Failed to fetch image"
            }
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Startup
if __name__ == "__main__":
    if USE_MODELS:
        load_models()
    threading.Thread(target=density_worker, daemon=True).start()
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
