# Use official Python runtime as base image with specific version for stability
FROM python:3.9.18-slim

# Add metadata labels
LABEL maintainer="Developer" \
      description="Traffic Analysis Application" \
      version="1.0"

# Set working directory
WORKDIR /app

# Set environment variables for optimizing TensorFlow memory usage and app behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_MEMORY_ALLOCATION=512MB \
    PORT=10000 \
    USE_MODELS=true \
    BASE_DIR=/app \
    CAMERA_URL_TEMPLATE="https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id={camera_id}"

# Install system dependencies required for TensorFlow and OpenCV, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libatlas-base-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -r -u 1001 appuser

# Copy requirements first (optimization for caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY model_converter.py app.py ./
COPY unet_road_segmentation.keras unet_multi_classV1.keras ./

# Run model conversion during build to ensure models are in SavedModel format
RUN python model_converter.py

# Create necessary directories and initialize JSON files with proper ownership
RUN mkdir -p /app/densities && \
    touch /app/densities/today_densities.json && \
    touch /app/densities/yesterday_max_densities.json && \
    touch /app/densities/critical_densities.json && \
    touch /app/densities/densities.json && \
    echo "{}" > /app/densities/today_densities.json && \
    echo "{}" > /app/densities/yesterday_max_densities.json && \
    echo "{}" > /app/densities/critical_densities.json && \
    echo "{}" > /app/densities/densities.json && \
    chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Add healthcheck to align with Render's healthCheckPath
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run application with optimized Gunicorn settings for Render's free tier
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--threads", "1", "--timeout", "120", "--log-level", "info", "app:app"]