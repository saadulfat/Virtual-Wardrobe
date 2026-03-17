# Stage 1: Build stage
FROM python:3.11 AS builder

# Set working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install system dependencies for OpenCV, Mediapipe, and crypto packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    pkg-config \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Optional: use non-root user
RUN useradd -m appuser
USER appuser

# Run the app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]