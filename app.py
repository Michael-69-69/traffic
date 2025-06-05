import os
import logging
import requests
from flask import Flask, jsonify

# Initialize Flask
app = Flask(__name__)

# Set up logging to mimic Render's log output
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

def load_dependencies():
    global _session
    if _session is None:
        try:
            _session = requests.Session()
            _session.headers.update(headers)
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
        # Warm-up request
        warmup_response = _session.get(warmup_url, timeout=15)
        warmup_response.raise_for_status()
        logger.info(f"Warm-up request successful: {warmup_response.status_code}")
        logger.info(f"Cookies set: {_session.cookies.get_dict()}")

        # Fetch image
        url = camera_urls.get(camera_id)
        if not url:
            logger.error(f"Camera {camera_id} not found")
            return {"success": False, "error": f"Camera {camera_id} not found"}

        response = _session.get(url, timeout=15)
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Content length: {len(response.content)} bytes")
        logger.info(f"Content-Type: {response.headers.get('Content-Type')}")

        if response.status_code == 200 and len(response.content) > 100:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'image' in content_type:
                image_size = len(response.content)
                logger.info(f"Image fetched successfully for {camera_id}, size: {image_size} bytes")
                return {"success": True, "camera_id": camera_id, "image_size_bytes": image_size}
            else:
                logger.warning(f"Unexpected Content-Type for {camera_id}: {content_type}")
                logger.warning(f"Response content (first 500 chars): {response.text[:500]}")
                return {"success": False, "error": f"Unexpected Content-Type: {content_type}"}
        else:
            logger.error(f"Failed to fetch image for {camera_id}")
            logger.error(f"Response headers: {response.headers}")
            logger.error(f"Response content (first 500 chars): {response.text[:500]}")
            return {"success": False, "error": "Failed to fetch image"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching image for {camera_id}: {e}")
        return {"success": False, "error": f"Error fetching image: {str(e)}"}

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
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 for Oracle
    app.run(host='0.0.0.0', port=port, debug=False)
