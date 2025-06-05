import os
import logging
import requests
import cv2
import numpy as np
from flask import Flask, jsonify

# Initialize Flask
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera URLs (limited to A and B for now)
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

# Lazy-load requests session
_session = None

def load_dependencies():
    global _session, cv2, np
    if _session is None:
        try:
            import cv2
            import numpy as np
            _session = requests.Session()
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

def fetch_camera_image(camera_id):
    if not load_dependencies():
        return None
    try:
        global _session
        # Warm-up request
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
        for attempt in range(3):
            try:
                response = _session.get(url, timeout=15)
                response.raise_for_status()
                if response.content and len(response.content) > 100:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'image' in content_type:
                        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        if image is not None and image.size > 0:
                            logger.info(f"Successfully fetched and decoded image for camera {camera_id}")
                            return image
                        logger.warning(f"Failed to decode image from {url} (attempt {attempt+1}/3)")
                    else:
                        logger.warning(f"Unexpected Content-Type: {content_type}")
                else:
                    logger.warning(f"Empty or invalid response from {url} (attempt {attempt+1}/3)")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {url}: {e} (attempt {attempt+1}/3)")
                break
            except Exception as e:
                logger.error(f"Error fetching image for {camera_id}: {e} (attempt {attempt+1}/3)")
            time.sleep(1)
        logger.error(f"Failed to fetch valid image for {camera_id} after 3 attempts")
        return None
    except Exception as e:
        logger.error(f"Critical error fetching camera image for {camera_id}: {e}")
        return None

def estimate_vehicle_count_from_blobs(blob_sizes, min_blob_size=500):
    significant_blobs = [size for size in blob_sizes if size >= min_blob_size]
    if not significant_blobs:
        return 0, 0
    unit_vehicle_size = max(500, min(1800, np.median([s for s in significant_blobs if s <= np.percentile(significant_blobs, 25)] or [200])))
    total_vehicles = sum(1 if size < unit_vehicle_size * 1.2 else max(1, int(size / unit_vehicle_size)) for size in significant_blobs)
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
    traffic_level = "No Traffic" if density_metric < 10 else "Light Traffic" if density_metric < 25 else "Moderate Traffic" if density_metric < 50 else "Heavy Traffic" if density_metric < 70 else "Congested"
    return estimated_speed, density_metric, traffic_level

def analyze_image(image):
    if image is None:
        return {
            "density": 0.0,
            "vehicle_count": 0,
            "avg_vehicle_size": 0,
            "density_metric": 0.0,
            "estimated_speed": 0.0,
            "traffic_level": "No Traffic"
        }
    try:
        # Simple vehicle detection using blob analysis (no models yet)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        blob_sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if min_reasonable_blob <= stats[i, cv2.CC_STAT_AREA] <= max_reasonable_blob]
        min_reasonable_blob = 500
        max_reasonable_blob = 8000

        if blob_sizes:
            vehicle_count, avg_vehicle_size = estimate_vehicle_count_from_blobs(blob_sizes)
        else:
            vehicle_count, avg_vehicle_size = 0, 0

        # Road area (simplified, no segmentation model yet)
        road_area_pixels = image.shape[0] * image.shape[1]  # Placeholder; full app uses road_mask
        estimated_speed, density_metric, traffic_level = apply_greenshields_model(vehicle_count, road_area_pixels, image.shape)
        density_percentage = (vehicle_count / (road_area_pixels / 100)) if road_area_pixels > 0 else 0.0
        density_percentage = round(max(0, min(100, density_percentage)), 1)

        logger.info(f"Analyzed image for camera: density={density_percentage}%, vehicles={vehicle_count}, speed={estimated_speed:.1f} km/h, traffic={traffic_level}")
        return {
            "density": density_percentage,
            "vehicle_count": vehicle_count,
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

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "1.0",
        "message": "Traffic Analysis Service is operational",
        "cameras": [cam[0] for cam in cameras]
    })

@app.route('/analyze/<camera_id>')
def analyze_camera(camera_id):
    try:
        logger.info(f"Analyzing camera {camera_id}")
        image = fetch_camera_image(camera_id)
        if image is None:
            return jsonify({"error": f"Failed to fetch image for camera {camera_id}", "success": False}), 500
        analysis_result = analyze_image(image)
        return jsonify({"camera_id": camera_id, "success": True, **analysis_result})
    except Exception as e:
        logger.error(f"Error in analyze endpoint for {camera_id}: {e}")
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
