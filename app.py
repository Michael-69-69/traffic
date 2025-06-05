import os
import logging
import requests
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify
from datetime import datetime
import gc
import threading

# Initialize Flask
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera URLs for A and B
camera_urls = {
    'A': "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id=6623e7076f998a001b2523ea&bg=black&w=300&h=230",
    'B': "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id=5deb576d1dc17d7c5515acf8&bg=black&w=300&h=230"
}

# Warm-up URL
warmup_url = "https://giaothong.hochiminhcity.gov.vn/"

# Headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "image/jpeg,image/png,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
    "Origin": "https://giaothong.hochiminhcity.gov.vn"
}

# Lazy-load dependencies
_session = None
_tf, _cv2, _np, _torch, _transforms, _road_model, _vehicle_model = [None] * 7
DEVICE = 'cpu'
worker_lock = threading.Lock()
last_density_update = None

# MiniUNet architecture
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
            import tensorflow as tf
            import cv2
            import numpy as np
            import requests
            import torch
            import torchvision.transforms as transforms
            _tf, _cv2, _np, _requests, _torch, _transforms = tf, cv2, np, requests, torch, transforms
            _session = _requests.Session()
            _session.headers.update(headers)
            logger.info("Dependencies loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return False
    return True

def load_models():
    global _road_model, _vehicle_model
    if not load_dependencies():
        logger.error("Failed to load dependencies")
        return False
    base_directory = os.environ.get('BASE_DIR', os.getcwd())
    road_model_path = os.path.join(base_directory, "unet_road_segmentation (Better).keras")
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

def fetch_camera_image(camera_id):
    if not load_dependencies():
        return None, 0
    try:
        warmup_response = _session.get(warmup_url, timeout=15)
        warmup_response.raise_for_status()
        logger.info(f"Warm-up request successful: {warmup_response.status_code}")
        
        url = camera_urls.get(camera_id)
        if not url:
            logger.error(f"Camera {camera_id} not found")
            return None, 0
        
        response = _session.get(url, timeout=15)
        if response.status_code == 200 and len(response.content) > 100:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'image' in content_type:
                image_array = _np.asarray(bytearray(response.content), dtype=_np.uint8)
                image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)
                if image is not None and image.size > 0:
                    image_size = len(response.content)
                    logger.info(f"Successfully fetched image for {camera_id}, size: {image_size} bytes")
                    return image, image_size
                logger.warning(f"Failed to decode image for {camera_id}")
                return None, 0
            logger.warning(f"Unexpected Content-Type for {camera_id}: {content_type}")
            return None, 0
        logger.error(f"Failed to fetch image for {camera_id}: Status {response.status_code}")
        return None, 0
    except Exception as e:
        logger.error(f"Error fetching image for {camera_id}: {e}")
        return None, 0
    finally:
        gc.collect()

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
    
    sizes = _np.array(significant_blobs)
    q20 = _np.percentile(sizes, 20)
    single_vehicle_blobs = sizes[sizes <= q20]
    avg_single_vehicle = _np.median(single_vehicle_blobs) if len(single_vehicle_blobs) > 0 else 200
    avg_single_vehicle = max(200, min(avg_single_vehicle, 1500))
    
    vehicle_count = sum(max(1, int(size / avg_single_vehicle)) for size in significant_blobs)
    return int(vehicle_count), int(avg_single_vehicle)

def analyze_image(image):
    if not load_dependencies() or image is None or _road_model is None or _vehicle_model is None:
        return {"density": 0.0, "vehicle_count": 0, "avg_vehicle_size": 0}
    try:
        img_road, img_vehicle = preprocess_image(image)
        if img_road is None or img_vehicle is None:
            return {"density": 0.0, "vehicle_count": 0, "avg_vehicle_size": 0}
        
        # Road segmentation
        road_pred = _road_model.predict(img_road, verbose=0)
        road_mask = (road_pred.squeeze() > 0.5).astype(_np.uint8)
        road_mask_resized = _cv2.resize(road_mask, (image.shape[1], image.shape[0]), interpolation=_cv2.INTER_NEAREST)
        road_pixels = _np.count_nonzero(road_mask_resized)
        
        # Vehicle detection
        with _torch.no_grad():
            vehicle_pred = _vehicle_model(img_vehicle)
        vehicle_mask = vehicle_pred.squeeze().cpu().numpy()
        vehicle_mask_resized = _cv2.resize(vehicle_mask, (image.shape[1], image.shape[0]))
        binary_vehicle_mask = (vehicle_mask_resized > 0.25).astype(_np.uint8)
        
        # Calculate vehicle pixels on road
        road_binary = (road_mask_resized > 0).astype(_np.uint8)
        vehicles_on_road = _np.logical_and(binary_vehicle_mask, road_binary).astype(_np.uint8)
        vehicle_pixels_on_road = _np.count_nonzero(vehicles_on_road)
        
        # Calculate density
        density_percentage = (vehicle_pixels_on_road / road_pixels * 100) if road_pixels > 0 else 0.0
        density_percentage = round(max(0, min(100, density_percentage)), 1)
        
        # Vehicle counting
        vehicles_on_road_cleaned = _cv2.morphologyEx(vehicles_on_road, _cv2.MORPH_OPEN, _np.ones((2, 2), _np.uint8), iterations=1)
        vehicles_on_road_cleaned = _cv2.morphologyEx(vehicles_on_road_cleaned, _cv2.MORPH_CLOSE, _np.ones((5, 5), _np.uint8), iterations=1)
        
        num_labels, _, stats, _ = _cv2.connectedComponentsWithStats(vehicles_on_road_cleaned, connectivity=8)
        blob_sizes = [stats[i, _cv2.CC_STAT_AREA] for i in range(1, num_labels) if 500 <= stats[i, _cv2.CC_STAT_AREA] <= 8000]
        
        vehicle_count, avg_vehicle_size = estimate_vehicle_count_from_blobs(blob_sizes)
        
        return {
            "density": density_percentage,
            "vehicle_count": vehicle_count,
            "avg_vehicle_size": avg_vehicle_size
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {"density": 0.0, "vehicle_count": 0, "avg_vehicle_size": 0}
    finally:
        gc.collect()

def fetch_and_process_densities():
    global last_density_update
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_density_update = datetime.now()
    results = {"timestamp": timestamp_str, "cameras": {}}
    
    for camera_id in ['A', 'B']:
        try:
            image, image_size = fetch_camera_image(camera_id)
            density_data = {
                "name": f"Camera {camera_id}",
                "density": 0.0,
                "vehicle_count": 0,
                "avg_vehicle_size": 0,
                "image_size_bytes": image_size,
                "timestamp": timestamp_str
            }
            if image is None:
                logger.warning(f"Image fetch failed for Camera {camera_id}")
                density_data["error"] = "Failed to fetch image"
            else:
                analysis_result = analyze_image(image)
                density_data.update(analysis_result)
                density_data["image_size_bytes"] = image_size
            results["cameras"][camera_id] = density_data
            logger.info(f"Processed Camera {camera_id}: density={density_data['density']}%, vehicles={density_data['vehicle_count']}, image_size={image_size} bytes")
        except Exception as e:
            logger.error(f"Error processing Camera {camera_id}: {e}")
            results["cameras"][camera_id] = {
                "name": f"Camera {camera_id}",
                "density": 0.0,
                "vehicle_count": 0,
                "avg_vehicle_size": 0,
                "image_size_bytes": 0,
                "timestamp": timestamp_str,
                "error": str(e)
            }
        finally:
            gc.collect()
    return results

def density_worker():
    logger.info("Density worker started - running every 30 seconds")
    try:
        load_models()
        while True:
            with worker_lock:
                logger.info("Starting density processing cycle")
                fetch_and_process_densities()
                logger.info("Density processing cycle completed")
            time.sleep(30)
    except Exception as e:
        logger.error(f"Error in density worker: {e}")

# Routes
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "1.0",
        "message": "Traffic Analysis Service for Cameras A and B"
    })

@app.route('/densities')
def get_densities():
    try:
        results = fetch_and_process_densities()
        raw_densities = {
            camera_id: {
                "density": data["density"],
                "vehicle_count": data["vehicle_count"],
                "image_size_bytes": data["image_size_bytes"]
            } for camera_id, data in results["cameras"].items()
        }
        return jsonify(raw_densities)
    except Exception as e:
        logger.error(f"Error in densities endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera-status')
def check_camera_status():
    try:
        results = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "cameras": {}}
        for camera_id in ['A', 'B']:
            image, image_size = fetch_camera_image(camera_id)
            if image is None:
                results["cameras"][camera_id] = {"name": f"Camera {camera_id}", "status": "offline", "image_size_bytes": 0}
            else:
                results["cameras"][camera_id] = {
                    "name": f"Camera {camera_id}",
                    "status": "online",
                    "resolution": f"{image.shape[1]}x{image.shape[0]}",
                    "image_size_bytes": image_size
                }
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in camera-status endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    threading.Thread(target=density_worker, daemon=True).start()
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
