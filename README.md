# Uzbek TTS Wav2Lip API

A FastAPI application that integrates Uzbek Text-to-Speech (Aisha AI) with Wav2Lip to produce lip-synced videos from Uzbek text input.

## Features

- **Uzbek TTS Integration**: Converts Uzbek text to speech using Aisha AI
- **Wav2Lip Processing**: Lip-syncs videos with generated audio using Wav2Lip
- **RESTful API**: Simple HTTP API for video processing
- **File Management**: Automatic cleanup of temporary files

## Prerequisites

- Python 3.8+
- ffmpeg (required by Wav2Lip)
- Wav2Lip checkpoint file (`wav2lip_gan.pth`)
- Aisha AI API key

## Installation

### 1. Clone or navigate to the project directory

```bash
cd Content_Creating
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The Wav2Lip dependencies (PyTorch, OpenCV, etc.) may require additional setup based on your system. For GPU support, you may need to install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 5. Download Wav2Lip Checkpoint

Download the Wav2Lip checkpoint file and place it in the checkpoints directory:

```bash
# Create checkpoints directory if it doesn't exist
mkdir -p Wav2Lip-master/checkpoints

# Download wav2lip_gan.pth
# You can find the download link in the Wav2Lip repository README
# Place the file at: Wav2Lip-master/checkpoints/wav2lip_gan.pth
```

The checkpoint file can be downloaded from:
- [Wav2Lip GAN Model](https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view?usp=share_link)

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Aisha AI API credentials:

```env
AISHA_AI_API_KEY=your_aisha_ai_api_key_here
AISHA_AI_API_URL=https://api.aisha.ai/v1/tts
AISHA_AI_VOICE_ID=uzbek

WAV2LIP_CHECKPOINT_PATH=Wav2Lip-master/checkpoints/wav2lip_gan.pth
```

## Usage

### Start the API Server

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### Process a Video

#### Using curl

```bash
curl -X POST "http://localhost:8000/process-video" \
  -F "video=@path/to/your/video.mp4" \
  -F "text=Salom, bu test video" \
  --output result.mp4
```

#### Using Python requests

```python
import requests

url = "http://localhost:8000/process-video"
files = {"video": open("path/to/your/video.mp4", "rb")}
data = {"text": "Salom, bu test video"}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open("result.mp4", "wb") as f:
        f.write(response.content)
    print("Video processed successfully!")
else:
    print(f"Error: {response.json()}")
```

#### Using JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('video', videoFile);
formData.append('text', 'Salom, bu test video');

fetch('http://localhost:8000/process-video', {
  method: 'POST',
  body: formData
})
.then(response => response.blob())
.then(blob => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'result.mp4';
  a.click();
});
```

### Health Check

Check if the API is running and properly configured:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "wav2lip_available": true,
  "checkpoint_exists": true,
  "aisha_ai_configured": true
}
```

## API Endpoints

### POST `/process-video`

Process a video file with Uzbek TTS and Wav2Lip lip-sync.

**Request:**
- `video` (file): Video file (MP4, AVI, MOV, etc.)
- `text` (form field): Uzbek text to convert to speech

**Response:**
- Returns the processed video file as download

**Example:**
```bash
curl -X POST "http://localhost:8000/process-video" \
  -F "video=@input.mp4" \
  -F "text=Salom, mening ismim Aisha" \
  --output output.mp4
```

### GET `/health`

Check API health and configuration status.

**Response:**
```json
{
  "status": "ok",
  "wav2lip_available": true,
  "checkpoint_exists": true,
  "aisha_ai_configured": true
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AISHA_AI_API_KEY` | Aisha AI API key | Required |
| `AISHA_AI_API_URL` | Aisha AI API endpoint | `https://api.aisha.ai/v1/tts` |
| `AISHA_AI_VOICE_ID` | Voice ID for TTS | `uzbek` |
| `WAV2LIP_CHECKPOINT_PATH` | Path to Wav2Lip checkpoint | `Wav2Lip-master/checkpoints/wav2lip_gan.pth` |
| `MAX_FILE_SIZE` | Maximum upload file size (bytes) | `104857600` (100MB) |
| `MIN_TEXT_LENGTH` | Minimum text length | `1` |
| `MAX_TEXT_LENGTH` | Maximum text length | `5000` |
| `WAV2LIP_PADS` | Face bounding box padding | `0,10,0,0` |
| `WAV2LIP_RESIZE_FACTOR` | Resolution reduction factor | `1` |
| `WAV2LIP_FPS` | Frames per second | `25.0` |

## Troubleshooting

### Wav2Lip Checkpoint Not Found

Ensure the checkpoint file is downloaded and placed at:
```
Wav2Lip-master/checkpoints/wav2lip_gan.pth
```

### ffmpeg Not Found

Install ffmpeg:
```bash
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### Face Detection Fails

- Ensure the video contains visible faces
- Try adjusting the `WAV2LIP_PADS` parameter
- Use `--resize_factor` to reduce video resolution if needed

### TTS API Errors

- Verify your Aisha AI API key is correct
- Check the API URL and voice ID settings
- Ensure your API key has sufficient credits/quota

### Audio Format Issues

The TTS service automatically converts audio to WAV format (16kHz) required by Wav2Lip. If conversion fails, ensure ffmpeg is installed.

## Project Structure

```
Content_Creating/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models.py              # Pydantic models
│   ├── config.py              # Configuration
│   ├── services/
│   │   ├── tts_service.py     # Aisha AI TTS integration
│   │   └── wav2lip_service.py # Wav2Lip wrapper
│   └── utils/
│       └── file_manager.py    # File utilities
├── Wav2Lip-master/           # Wav2Lip project
├── uploads/                  # Temporary uploads
├── outputs/                  # Processed videos
├── requirements.txt
├── .env                      # Environment variables
└── README.md
```

## License

This project integrates with:
- **Wav2Lip**: Non-commercial use only (see Wav2Lip license)
- **Aisha AI**: Subject to Aisha AI terms of service

## Support

For issues related to:
- **Wav2Lip**: See [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip)
- **Aisha AI**: Contact Aisha AI support
- **This API**: Open an issue in this repository

## Notes

- Wav2Lip requires videos with visible faces
- Audio length should match or be shorter than video length
- Processing time depends on video length and system resources
- GPU acceleration is recommended for faster processing

