# Use Python 3.10 (mediapipe supports this)
FROM python:3.10-slim

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies (fixes libGL error)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Set Render environment variable
ENV PORT=10000

# Start the FastAPI app
CMD ["python", "face_api.py"]
