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
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy sd-wav2lip-uhq directory (contains Wav2Lip implementation)
COPY sd-wav2lip-uhq/ ./sd-wav2lip-uhq/

# Copy wav2lip_uhq directory (contains preprocessing tools)
COPY wav2lip_uhq/ ./wav2lip_uhq/

# Copy application code
COPY app/ ./app/

# Copy static files (dashboard)
COPY static/ ./static/

# Copy documentation
COPY *.md ./

# Create necessary directories
RUN mkdir -p uploads outputs sd-wav2lip-uhq/scripts/wav2lip/temp sd-wav2lip-uhq/scripts/wav2lip/checkpoints wav2lip_uhq/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 6070

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6070"]

