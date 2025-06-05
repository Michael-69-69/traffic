import os
import logging
import requests
from flask import Flask, jsonify

# Initialize Flask
app = Flask(__name__)

# Set up logging with INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Lazy-load requests session
_session = None

def load_dependencies():
    global _session
    if _session is None:
        try:
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

def fetch_camera_image_size(camera_id):
    if not load_dependencies():
        return {"success": False, "error": "Failed to load dependencies"}
    try:
        global _session
        try:
            warmup_response = _session.get("https://giaothong.hochiminhcity.gov.vn/", timeout=15)
            warmup_response.raise_for_status()
            logger.info(f"Warm-up request successful: {warmup_response.status_code}")
        except Exception as e:
            logger.error(f"Warm-up request failed: {e}")
            return {"success": False, "error": f"Warm-up request failed: {str(e)}"}

        camera = next((c for c in camera_websites if c['id'] == camera_id), None)
        if not camera:
            logger.error(f"Camera {camera_id} not found in camera_websites")
            return {"success": False, "error": f"Camera {camera_id} not found"}

        cam_id = camera['url'].split('camId=')[1].split('&')[0]
        url = CAMERA_URL_TEMPLATE.format(camera_id=cam_id)
        logger.info(f"Fetching image from primary URL: {url}")
        try:
            response = _session.get(url, timeout=15)
            response.raise_for_status()
            if response.content:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'image' in content_type:
                    image_size = len(response.content)
                    logger.info(f"Successfully fetched image for camera {camera_id}, size: {image_size} bytes")
                    return {"success": True, "camera_id": camera_id, "image_size_bytes": image_size}
            logger.error(f"Failed to fetch valid image for {camera_id}")
            return {"success": False, "error": "Failed to fetch valid image"}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return {"success": False, "error": f"HTTP error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error fetching image for {camera_id}: {e}")
            return {"success": False, "error": f"Error fetching image: {str(e)}"}
    except Exception as e:
        logger.error(f"Critical error fetching camera image for {camera_id}: {e}")
        return {"success": False, "error": f"Critical error: {str(e)}"}

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "version": "1.0",
        "message": "Image Fetch Size Test Service is operational"
    })

@app.route('/fetch-size/<camera_id>')
def fetch_image_size(camera_id):
    result = fetch_camera_image_size(camera_id)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
