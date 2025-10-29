# Use Python 3.10 (mediapipe supports this)
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Create app folder
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your source code
COPY . .

# Expose Render port
ENV PORT=10000

# Start the FastAPI app
CMD ["python", "face_api.py"]
