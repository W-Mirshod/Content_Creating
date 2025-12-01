FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Wav2Lip project (if it exists)
COPY Wav2Lip-master/ ./Wav2Lip-master/

# Copy application code
COPY app/ ./app/

# Copy static files (dashboard)
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p uploads outputs Wav2Lip-master/temp Wav2Lip-master/checkpoints

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 6070

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6070"]

