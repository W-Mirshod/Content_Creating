import subprocess
import sys
import traceback
import torch
import shutil
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
    WAV2LIP_FACE_DET_BATCH_SIZE,
    WAV2LIP_BATCH_SIZE,
    TORCH_NUM_THREADS,
)

# Import W2l class from sd-wav2lip-uhq
try:
    # Add sd-wav2lip-uhq to Python path for imports
    sd_wav2lip_root = WAV2LIP_SCRIPTS_ROOT.parent
    if str(sd_wav2lip_root) not in sys.path:
        sys.path.insert(0, str(sd_wav2lip_root))

    from scripts.wav2lip.w2l import W2l
    W2L_AVAILABLE = True
except Exception as e:
    W2L_AVAILABLE = False
    W2L_IMPORT_ERROR = str(e)


class Wav2LipServiceError(Exception):
    """Custom exception for Wav2Lip service errors"""
    pass


def validate_checkpoint_file(checkpoint_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a checkpoint file exists and can be loaded.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not checkpoint_path.exists():
        return False, f"Checkpoint file does not exist: {checkpoint_path}"
    
    if not checkpoint_path.is_file():
        return False, f"Checkpoint path is not a file: {checkpoint_path}"
    
    # Check file size (corrupted files are often very small)
    file_size = checkpoint_path.stat().st_size
    if file_size < 1024:  # Less than 1KB is definitely corrupted
        return False, (
            f"Checkpoint file appears corrupted (size: {file_size} bytes). "
            f"Please re-download the checkpoint file from: "
            f"https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW"
        )
    
    # Try to load the checkpoint to verify it's not corrupted
    try:
        import zipfile
        import pickle

        try:
            # Try to manually load from zip file to prevent auto-dispatch
            with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                # Regular PyTorch checkpoints in zip format contain 'data.pkl'
                if 'data.pkl' in zf.namelist():
                    with zf.open('data.pkl') as pkl_file:
                        unpickler = pickle.Unpickler(pkl_file)
                        unpickler.weights_only = False
                        checkpoint = unpickler.load()
                else:
                    # Fallback to torch.load (may auto-dispatch)
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except (zipfile.BadZipFile, KeyError):
            # Not a zip file or different format, use regular torch.load
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Allow both dictionaries (regular checkpoints) and TorchScript models
        if not isinstance(checkpoint, (dict, torch.jit.ScriptModule)):
            return False, (
                f"Expected checkpoint to be a dictionary or TorchScript model, but got {type(checkpoint)}. "
                f"The checkpoint file appears to be corrupted or in an unsupported format."
            )
    except EOFError:
        return False, (
            f"Checkpoint file is corrupted or incomplete (EOFError). "
            f"File size: {file_size} bytes. "
            f"Please delete the file and re-download it from: "
            f"https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW"
        )
    except Exception as e:
        return False, (
            f"Checkpoint file cannot be loaded: {type(e).__name__}: {str(e)}. "
            f"The file may be corrupted. Please re-download it."
        )
    
    return True, None


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
    
    # Validate checkpoint file exists and is not corrupted
    is_valid, error_msg = validate_checkpoint_file(checkpoint)
    if not is_valid:
        raise Wav2LipServiceError(error_msg)

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
    
    # Ensure Wav2Lip required directories exist (results, temp, checkpoints)
    (WAV2LIP_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (WAV2LIP_ROOT / "temp").mkdir(parents=True, exist_ok=True)
    (WAV2LIP_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)

    try:
        # Optimize batch sizes for CPU performance
        # Temporarily patch W2l class to use optimized batch sizes
        original_init = W2l.__init__
        def optimized_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.device == 'cpu':
                # Use optimized batch sizes for CPU
                self.face_det_batch_size = WAV2LIP_FACE_DET_BATCH_SIZE
                self.wav2lip_batch_size = WAV2LIP_BATCH_SIZE
                print(f"[INFO] Using optimized batch sizes for CPU - Face Detection: {self.face_det_batch_size}, Wav2Lip: {self.wav2lip_batch_size}")
        W2l.__init__ = optimized_init
        
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
        # Use shutil.move instead of rename to handle cross-device moves
        shutil.move(str(default_output), str(output_path))

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
                shutil.move(str(enhanced_video), str(output_path))

                # Cleanup enhancement cache
                cleanup_enhancement_cache(uhq_output.parent)

            except ImportError:
                print("[WARNING] Wav2Lip UHQ service not available, skipping enhancement")
            except Exception as e:
                print(f"[WARNING] UHQ enhancement failed, returning base Wav2Lip output: {e}")
                # Continue with non-enhanced output

        return output_path

    except EOFError as e:
        # Specific handling for corrupted checkpoint files
        error_msg = (
            f"Checkpoint file appears corrupted or incomplete (EOFError). "
            f"This usually means the file download was interrupted or the file is damaged. "
            f"Please delete the checkpoint file at {checkpoint} and re-download it from: "
            f"https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW"
        )
        print(f"[ERROR] Wav2Lip processing failed: {error_msg}")
        raise Wav2LipServiceError(error_msg)
    except Exception as e:
        # Capture full error details including traceback
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "No error message provided"
        tb_str = traceback.format_exc()
        
        # Log full error for debugging
        print(f"[ERROR] Wav2Lip processing failed:")
        print(f"  Type: {error_type}")
        print(f"  Message: {error_msg}")
        print(f"  Traceback:\n{tb_str}")
        
        # Create detailed error message
        if error_msg:
            detailed_error = f"{error_type}: {error_msg}"
        else:
            detailed_error = f"{error_type} (no message provided)"
        
        raise Wav2LipServiceError(f"Unexpected error during Wav2Lip processing: {detailed_error}")


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


def validate_checkpoint_in_setup() -> Tuple[bool, Optional[str]]:
    """
    Validate checkpoint file for health check endpoint.
    This is a lighter check that doesn't raise exceptions.

    Returns:
        Tuple of (is_valid, error_message)
    """
    checkpoint_path = Path(WAV2LIP_CHECKPOINT)
    return validate_checkpoint_file(checkpoint_path)
    