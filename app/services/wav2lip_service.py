import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from app.config import (
    WAV2LIP_ROOT,
    WAV2LIP_CHECKPOINT,
    WAV2LIP_INFERENCE_SCRIPT,
    WAV2LIP_TEMP_DIR,
    WAV2LIP_PADS,
    WAV2LIP_RESIZE_FACTOR,
    WAV2LIP_FPS
)


class Wav2LipServiceError(Exception):
    """Custom exception for Wav2Lip service errors"""
    pass


def process_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    pads: Optional[List[int]] = None,
    resize_factor: Optional[int] = None,
    fps: Optional[float] = None
) -> Path:
    """
    Process video with Wav2Lip to lip-sync with audio.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to audio file (WAV format)
        output_path: Path to save output video
        checkpoint_path: Path to Wav2Lip checkpoint (uses config default if None)
        pads: Face bounding box padding [top, bottom, left, right]
        resize_factor: Resolution reduction factor
        fps: Frames per second (for static images)
        
    Returns:
        Path to processed video file
        
    Raises:
        Wav2LipServiceError: If processing fails
    """
    # Validate inputs
    if not video_path.exists():
        raise Wav2LipServiceError(f"Video file does not exist: {video_path}")
    
    if not audio_path.exists():
        raise Wav2LipServiceError(f"Audio file does not exist: {audio_path}")
    
    if not WAV2LIP_INFERENCE_SCRIPT.exists():
        raise Wav2LipServiceError(
            f"Wav2Lip inference script not found: {WAV2LIP_INFERENCE_SCRIPT}"
        )
    
    # Use provided checkpoint or default from config
    checkpoint = Path(checkpoint_path) if checkpoint_path else Path(WAV2LIP_CHECKPOINT)
    if not checkpoint.exists():
        raise Wav2LipServiceError(
            f"Wav2Lip checkpoint not found: {checkpoint}. "
            f"Please download the checkpoint file and place it in the checkpoints directory."
        )
    
    # Use provided parameters or defaults from config
    pads_list = pads if pads is not None else WAV2LIP_PADS
    resize = resize_factor if resize_factor is not None else WAV2LIP_RESIZE_FACTOR
    fps_value = fps if fps is not None else WAV2LIP_FPS
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure Wav2Lip temp directory exists
    WAV2LIP_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build command to run Wav2Lip inference
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(WAV2LIP_INFERENCE_SCRIPT),
        '--checkpoint_path', str(checkpoint),
        '--face', str(video_path),
        '--audio', str(audio_path),
        '--outfile', str(output_path),
        '--pads', *map(str, pads_list),
        '--resize_factor', str(resize),
    ]
    
    # Add FPS if provided (useful for static images)
    if fps_value:
        cmd.extend(['--fps', str(fps_value)])
    
    try:
        # Change to Wav2Lip root directory for proper execution
        # (Wav2Lip may have relative imports)
        result = subprocess.run(
            cmd,
            cwd=str(WAV2LIP_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Check if output file was created
        if not output_path.exists():
            raise Wav2LipServiceError(
                f"Wav2Lip processing completed but output file not found: {output_path}\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Wav2LipServiceError(
            "Wav2Lip processing timed out after 1 hour. "
            "The video may be too long or processing may have failed."
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Wav2Lip processing failed with return code {e.returncode}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        raise Wav2LipServiceError(error_msg)
    except Exception as e:
        raise Wav2LipServiceError(f"Unexpected error during Wav2Lip processing: {str(e)}")


def validate_wav2lip_setup() -> Tuple[bool, Optional[str]]:
    """
    Validate that Wav2Lip is properly set up.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    if not WAV2LIP_ROOT.exists():
        return False, f"Wav2Lip root directory not found: {WAV2LIP_ROOT}"
    
    if not WAV2LIP_INFERENCE_SCRIPT.exists():
        errors.append(f"Inference script not found: {WAV2LIP_INFERENCE_SCRIPT}")
    
    checkpoint = Path(WAV2LIP_CHECKPOINT)
    if not checkpoint.exists():
        errors.append(
            f"Checkpoint not found: {checkpoint}. "
            f"Please download wav2lip_gan.pth and place it in the checkpoints directory."
        )
    
    # Check if ffmpeg is available (required by Wav2Lip)
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        errors.append("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None

