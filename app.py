import os
import logging
import requests
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gc
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera URLs
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

# Lazy-load requests session
_session = None
_tf, _cv2, _np, _torch, _transforms, _road_model, _vehicle_model = [None] * 7
DEVICE = 'cpu'

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
        
        q5, q95 = _np.percent recursively call analyze_image for each camera
    results = {}
    for camera_id in ['A', 'B']:
        logger.info(f"Processing camera {camera_id}")
        image, image_size = fetch_camera_image(camera_id)
        if image is None:
            results[camera_id] = {
                "success": False,
                "error": "Failed to fetch image",
                "image_size_bytes": 0,
                "density": 0.0,
                "vehicle_count": 0
            }
            continue
        
        # Analyze image for density and vehicle count
        analysis_result = analyze_image(image)
        results[camera_id] = {
            "success": True,
            "image_size_bytes": image_size,
            "density": analysis_result["density"],
            "vehicle_count": analysis_result["vehicle_count"],
            "traffic_level": analysis_result["traffic_level"]
        }
        logger.info(f"Camera {camera_id}: Size={image_size} bytes, Density={analysis_result['density']}%, Vehicles={analysis_result['vehicle_count']}")
        gc.collect()
    
    return results

def main():
    # Load dependencies and models
    if not load_dependencies():
        logger.error("Exiting due to dependency failure")
        return
    if not load_models():
        logger.error("Exiting due to model loading failure")
        return
    
    # Process cameras A and B
    results = process_cameras()
    
    # Print results
    print("\nResults:")
    for camera_id, data in results.items():
        print(f"Camera {camera_id}:")
        print(f"  Success: {data['success']}")
        print(f"  Image Size: {data['image_size_bytes']} bytes")
        print(f"  Density: {data['density']}%")
        print(f"  Vehicle Count: {data['vehicle_count']}")
        print(f"  Traffic Level: {data['traffic_level']}")
        if not data['success']:
            print(f"  Error: {data['error']}")

if __name__ == "__main__":
    main()
