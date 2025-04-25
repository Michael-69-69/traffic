# Use official Python runtime as base image with specific version for stability
FROM python:3.9.18-slim

# Add metadata labels
LABEL maintainer="Developer" \
      description="Traffic Analysis Application" \
      version="1.0"

# Set working directory
WORKDIR /app

# Set environment variables for optimizing TensorFlow memory usage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_MEMORY_ALLOCATION=512MB \
    PORT=10000

# Install system dependencies and clean up in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -r -u 1001 appuser

# Copy requirements first (optimization for caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY model_converter.py app.py ./
COPY unet_road_segmentation.keras unet_multi_classV1.keras ./

# Create necessary directories and files
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

# Switch to non-root user
USER appuser

# Command to run application with memory optimizations
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 0 "app:app"