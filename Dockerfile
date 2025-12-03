FROM python:3.10-slim

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

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY sd-wav2lip-uhq/ ./sd-wav2lip-uhq/

COPY app/ ./app/

COPY static/ ./static/

COPY *.md ./

RUN mkdir -p uploads outputs sd-wav2lip-uhq/scripts/wav2lip/temp sd-wav2lip-uhq/scripts/wav2lip/checkpoints wav2lip_uhq/temp

ENV PYTHONPATH=/app:/app/sd-wav2lip-uhq
ENV PYTHONUNBUFFERED=1

RUN echo '#!/bin/bash\n\
NUM_CORES=$(nproc)\n\
echo "[STARTUP] Detected $NUM_CORES CPU cores/threads"\n\
export OMP_NUM_THREADS=$NUM_CORES\n\
export MKL_NUM_THREADS=$NUM_CORES\n\
export NUMEXPR_NUM_THREADS=$NUM_CORES\n\
export TORCH_NUM_THREADS=$NUM_CORES\n\
export OPENBLAS_NUM_THREADS=$NUM_CORES\n\
export VECLIB_MAXIMUM_THREADS=$NUM_CORES\n\
export OMP_DYNAMIC=FALSE\n\
export OMP_SCHEDULE=STATIC\n\
export OMP_PROC_BIND=SPREAD\n\
export OMP_PLACES=threads\n\
export GOMP_CPU_AFFINITY="0-$((NUM_CORES-1))"\n\
export MKL_DYNAMIC=FALSE\n\
export MALLOC_TRIM_THRESHOLD_=128000\n\
echo "[STARTUP] CPU optimization enabled for $NUM_CORES threads"\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 6070

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6070"]

