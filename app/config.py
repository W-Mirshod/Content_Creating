import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
WAV2LIP_ROOT = BASE_DIR / "Wav2Lip-master"
WAV2LIP_CHECKPOINT = os.getenv(
    "WAV2LIP_CHECKPOINT_PATH",
    str(WAV2LIP_ROOT / "checkpoints" / "wav2lip_gan.pth")
)
WAV2LIP_INFERENCE_SCRIPT = WAV2LIP_ROOT / "inference.py"
WAV2LIP_TEMP_DIR = WAV2LIP_ROOT / "temp"

# File upload configuration
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB default
ALLOWED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv"]
ALLOWED_AUDIO_FORMATS: List[str] = ["wav", "mp3", "m4a"]

# Wav2Lip processing parameters
WAV2LIP_PADS = list(map(int, os.getenv("WAV2LIP_PADS", "0,30,0,0").split(",")))
WAV2LIP_RESIZE_FACTOR = int(os.getenv("WAV2LIP_RESIZE_FACTOR", "2"))
WAV2LIP_FPS = float(os.getenv("WAV2LIP_FPS", "25.0"))

# Wav2Lip UHQ Post-Processing Configuration
WAV2LIP_UHQ_ENABLED = os.getenv("WAV2LIP_UHQ_ENABLED", "False").lower() == "true"
WAV2LIP_UHQ_DENOISING_STRENGTH = float(os.getenv("WAV2LIP_UHQ_DENOISING_STRENGTH", "1.0"))
WAV2LIP_UHQ_MASK_BLUR = int(os.getenv("WAV2LIP_UHQ_MASK_BLUR", "8"))
STABLE_DIFFUSION_API_URL = os.getenv("STABLE_DIFFUSION_API_URL", "http://localhost:7860")
WAV2LIP_UHQ_TEMP_DIR = BASE_DIR / "wav2lip_uhq" / "temp"

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WAV2LIP_TEMP_DIR.mkdir(exist_ok=True)
WAV2LIP_UHQ_TEMP_DIR.mkdir(exist_ok=True)

