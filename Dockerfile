# Use official Python runtime as base image with specific version for stability
FROM python:3.9.18-slim

# Add metadata labels
LABEL maintainer="Developer" \
      description="Traffic Analysis Application" \
      version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in a single layer
RUN apt-get update && apt-get install -y \
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
COPY app.py .
COPY "unet_road_segmentation (Better).keras" .
COPY unet_multi_classV1.keras .

# Create necessary directories
RUN mkdir -p /app/densities

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run the app
CMD ["python", "app.py"]