import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from app.config import (
    WAV2LIP_SCRIPTS_ROOT,
    WAV2LIP_ROOT,
    WAV2LIP_CHECKPOINT,
    WAV2LIP_TEMP_DIR,
    WAV2LIP_PADS,
    WAV2LIP_RESIZE_FACTOR,
    WAV2LIP_FPS,
    WAV2LIP_UHQ_ENABLED,
)

# Add sd-wav2lip-uhq/scripts to Python path for imports
if str(WAV2LIP_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WAV2LIP_SCRIPTS_ROOT))

# Import W2l class from sd-wav2lip-uhq
try:
    from scripts.wav2lip.w2l import W2l
    W2L_AVAILABLE = True
except ImportError as e:
    W2L_AVAILABLE = False
    W2L_IMPORT_ERROR = str(e)


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
    fps: Optional[float] = None,
    use_uhq: Optional[bool] = None
) -> Path:
    """
    Process video with Wav2Lip to lip-sync with audio, optionally enhancing with UHQ post-processing.

    Args:
        video_path: Path to input video file
        audio_path: Path to audio file (WAV format)
        output_path: Path to save output video
        checkpoint_path: Path to Wav2Lip checkpoint (uses config default if None)
        pads: Face bounding box padding [top, bottom, left, right]
        resize_factor: Resolution reduction factor
        fps: Frames per second (for static images)
        use_uhq: Whether to apply UHQ post-processing (uses config default if None)

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

    if not W2L_AVAILABLE:
        raise Wav2LipServiceError(
            f"W2l class not available: {W2L_IMPORT_ERROR}. "
            "Please ensure sd-wav2lip-uhq is properly installed."
        )

    # Use provided checkpoint or default from config
    checkpoint = Path(checkpoint_path) if checkpoint_path else Path(WAV2LIP_CHECKPOINT)
    if not checkpoint.exists():
        raise Wav2LipServiceError(
            f"Wav2Lip checkpoint not found: {checkpoint}. "
            f"Please download the checkpoint file and place it in the checkpoints directory."
        )

    # Extract checkpoint name from path (e.g., "wav2lip_gan" from "wav2lip_gan.pth")
    checkpoint_name = checkpoint.stem  # Remove .pth extension

    # Use provided parameters or defaults from config
    pads_list = pads if pads is not None else WAV2LIP_PADS
    resize = resize_factor if resize_factor is not None else WAV2LIP_RESIZE_FACTOR
    uhq_enabled = use_uhq if use_uhq is not None else WAV2LIP_UHQ_ENABLED

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure Wav2Lip temp directory exists
    WAV2LIP_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Instantiate W2l class with parameters
        w2l = W2l(
            face=str(video_path),
            audio=str(audio_path),
            checkpoint=checkpoint_name,
            nosmooth=True,  # Disable smoothing for better quality (corresponds to --nosmooth)
            resize_factor=resize,
            pad_top=pads_list[0],
            pad_bottom=pads_list[1],
            pad_left=pads_list[2],
            pad_right=pads_list[3],
            face_swap_img=None  # No face swapping for basic functionality
        )

        # Execute Wav2Lip processing
        w2l.execute()

        # W2l saves output to results/result_voice.mp4 by default
        default_output = WAV2LIP_ROOT / "results" / "result_voice.mp4"
        if not default_output.exists():
            raise Wav2LipServiceError(
                f"Wav2Lip processing completed but output file not found: {default_output}"
            )

        # Move the result to the desired output path
        default_output.rename(output_path)

        # Apply UHQ post-processing if enabled
        if uhq_enabled:
            print("[INFO] Applying Wav2Lip UHQ enhancement...")
            try:
                from app.services.wav2lip_uhq_service import enhance_video, cleanup_enhancement_cache

                uhq_output = output_path.parent / f"{output_path.stem}_uhq{output_path.suffix}"
                enhanced_video = enhance_video(
                    output_path,
                    video_path,
                    uhq_output,
                    use_controlnet=True
                )

                # Replace original with enhanced version
                output_path.unlink()
                enhanced_video.rename(output_path)

                # Cleanup enhancement cache
                cleanup_enhancement_cache(uhq_output.parent)

            except ImportError:
                print("[WARNING] Wav2Lip UHQ service not available, skipping enhancement")
            except Exception as e:
                print(f"[WARNING] UHQ enhancement failed, returning base Wav2Lip output: {e}")
                # Continue with non-enhanced output

        return output_path

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

    if not W2L_AVAILABLE:
        errors.append(f"W2l class not available: {W2L_IMPORT_ERROR}")

    # Check if ffmpeg is available (required by Wav2Lip)
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True,
            timeout=5
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        errors.append("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg")

    # Note: We don't check checkpoint here as it's checked separately in health endpoint
    # This allows the service to be "available" even if checkpoint is missing

    if errors:
        return False, "; ".join(errors)

    return True, None

