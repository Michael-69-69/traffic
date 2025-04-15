# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (only essentials)
RUN apt-get update && apt-get install -y \
    # Add any required system packages (e.g., for opencv-python if needed)
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (optimization for caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY app.py .
COPY unet_road_segmentation\ \(Better\).keras .
COPY unet_multi_classV1.keras .

# Run the app
CMD ["python", "app.py"]