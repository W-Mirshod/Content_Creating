import subprocess
import sys
import os
import json
import requests
import base64
import io
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import cv2

try:
    import dlib
    from imutils import face_utils
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from app.config import (
    WAV2LIP_UHQ_ENABLED,
    STABLE_DIFFUSION_API_URL,
    WAV2LIP_UHQ_TEMP_DIR,
    WAV2LIP_UHQ_DENOISING_STRENGTH,
    WAV2LIP_UHQ_MASK_BLUR,
)


class Wav2LipUHQError(Exception):
    """Custom exception for Wav2Lip UHQ service errors"""
    pass


def assure_path_exists(path: Path):
    """Create directory if it doesn't exist"""
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_framerate(video_file: Path) -> float:
    """Get framerate of video file"""
    video = cv2.VideoCapture(str(video_file))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def create_video_from_images(image_dir: Path, output_video: Path, input_video: Path, nb_frames: int):
    """Create video from enhanced images"""
    fps = str(int(get_framerate(input_video)))
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", fps,
        "-start_number", "0",
        "-i", str(image_dir / "output_%05d.png"),
        "-vframes", str(nb_frames),
        "-b:v", "5000k",
        str(output_video)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Wav2LipUHQError(f"FFmpeg video creation failed: {result.stderr}")


def extract_audio_from_video(video_file: Path, audio_output: Path):
    """Extract audio from video file"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_file),
        "-vn",
        "-acodec", "copy",
        str(audio_output)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Wav2LipUHQError(f"Audio extraction failed: {result.stderr}")


def has_audio(video_file: Path) -> bool:
    """Check if video file has audio"""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_file)],
        text=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    return "Audio:" in result.stderr


def add_audio_to_video(video_file: Path, audio_file: Path, output_file: Path):
    """Add audio to video file"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_file),
        "-i", str(audio_file),
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        str(output_file)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Wav2LipUHQError(f"Audio merging failed: {result.stderr}")


def initialize_dlib_predictor():
    """Initialize dlib face detector and landmark predictor"""
    if not DLIB_AVAILABLE:
        raise Wav2LipUHQError("dlib not available. Install it: pip install dlib")
    
    print("[INFO] Loading the predictor...")
    detector = dlib.get_frontal_face_detector()
    
    # Path to the shape predictor model
    predictor_path = Path(__file__).parent.parent.parent / "wav2lip_uhq" / "predicator" / "shape_predictor_68_face_landmarks.dat"
    
    if not predictor_path.exists():
        raise Wav2LipUHQError(
            f"Shape predictor model not found: {predictor_path}\n"
            f"Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
    
    predictor = dlib.shape_predictor(str(predictor_path))
    return detector, predictor


def initialize_video_streams(wav2lip_video: Path, original_video: Path):
    """Initialize video capture streams"""
    print("[INFO] Loading video files...")
    vs = cv2.VideoCapture(str(wav2lip_video))
    vi = cv2.VideoCapture(str(original_video))
    
    if not vs.isOpened():
        raise Wav2LipUHQError(f"Cannot open Wav2Lip video: {wav2lip_video}")
    if not vi.isOpened():
        raise Wav2LipUHQError(f"Cannot open original video: {original_video}")
    
    return vs, vi


def enhance_image_with_controlnet(image_path: Path, mask_path: Path, payload_config: dict, frame_count: int, output_dir: Path) -> bool:
    """
    Send image to Stable Diffusion API for ControlNet enhancement
    
    Args:
        image_path: Path to the composited image
        mask_path: Path to the mouth mask
        payload_config: Payload configuration from controlNet.json
        frame_count: Frame number for naming
        output_dir: Output directory for enhanced images
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image and mask
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        with open(mask_path, "rb") as f:
            mask_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare payload
        payload = payload_config.get("payload", {}).copy()
        payload["init_images"] = [f"data:image/png;base64,{image_data}"]
        payload["mask"] = f"data:image/png;base64,{mask_data}"
        
        # Send to Stable Diffusion API
        url = payload_config.get("url", STABLE_DIFFUSION_API_URL)
        if not url:
            print("[WARNING] Stable Diffusion API URL not configured, skipping ControlNet enhancement")
            return False
        
        response = requests.post(url=f"{url}/sdapi/v1/img2img", json=payload, timeout=300)
        
        if response.status_code != 200:
            print(f"[WARNING] ControlNet API returned status {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        # Save enhanced image
        for idx, img_b64 in enumerate(result.get('images', [])):
            if isinstance(img_b64, str) and ',' in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            
            try:
                image = Image.open(io.BytesIO(base64.b64decode(img_b64)))
                output_name = output_dir / f"output_{str(frame_count).rjust(5, '0')}.png"
                image.save(output_name)
                return True
            except Exception as e:
                print(f"[WARNING] Failed to save enhanced image: {e}")
                return False
        
        return False
        
    except requests.exceptions.ConnectionError:
        print("[WARNING] Cannot connect to Stable Diffusion API. Is it running?")
        return False
    except Exception as e:
        print(f"[WARNING] ControlNet enhancement failed: {e}")
        return False


def enhance_video(
    wav2lip_video: Path,
    original_video: Path,
    output_video: Path,
    use_controlnet: bool = True
) -> Path:
    """
    Enhance Wav2Lip video with UHQ post-processing.
    
    Process:
    1. Extract frames from Wav2Lip video
    2. Detect mouth region in each frame
    3. Composite Wav2Lip mouth onto original video frame
    4. Optionally enhance with ControlNet (requires Stable Diffusion API)
    5. Create final video with audio
    
    Args:
        wav2lip_video: Path to Wav2Lip generated video
        original_video: Path to original input video
        output_video: Path to save enhanced output video
        use_controlnet: Whether to use ControlNet enhancement (requires SD API)
        
    Returns:
        Path to enhanced video
        
    Raises:
        Wav2LipUHQError: If processing fails
    """
    # Validate inputs
    if not wav2lip_video.exists():
        raise Wav2LipUHQError(f"Wav2Lip video not found: {wav2lip_video}")
    if not original_video.exists():
        raise Wav2LipUHQError(f"Original video not found: {original_video}")
    
    # Create output directories
    output_dir = Path(output_video).parent
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    assure_path_exists(images_dir)
    assure_path_exists(masks_dir)
    
    # Load ControlNet payload if available
    payload = None
    if use_controlnet:
        payload_path = Path(__file__).parent.parent.parent / "wav2lip_uhq" / "payloads" / "controlNet.json"
        if payload_path.exists():
            with open(payload_path, "r") as f:
                payload = json.load(f)
        else:
            print(f"[WARNING] ControlNet payload not found: {payload_path}")
            use_controlnet = False
    
    # Initialize dlib for face detection
    detector, predictor = initialize_dlib_predictor()
    
    # Initialize video streams
    vs, vi = initialize_video_streams(wav2lip_video, original_video)
    
    # Get facial landmarks indices for mouth
    (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    max_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    
    print(f"[INFO] Processing {max_frames} frames...")
    
    try:
        while True:
            # Check if frame already processed
            output_image = images_dir / f"image_{str(frame_number).rjust(5, '0')}.png"
            
            ret_wav2lip, frame_wav2lip = vs.read()
            ret_original, frame_original = vi.read()
            
            if not ret_wav2lip or not ret_original:
                break
            
            if output_image.exists():
                frame_number += 1
                continue
            
            print(f"[INFO] Processing frame {frame_number} of {max_frames}")
            
            # Convert to RGB (dlib expects RGB)
            frame_rgb = cv2.cvtColor(frame_wav2lip, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            rects = detector(frame_rgb, 0)
            
            if len(rects) == 0:
                print(f"[WARNING] No face detected in frame {frame_number}")
                frame_number += 1
                continue
            
            # Initialize mask and result
            mask = np.zeros_like(frame_rgb)
            result = frame_original.copy()
            
            # Process each detected face
            for rect in rects:
                # Get facial landmarks
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Extract mouth region
                mouth = shape[mstart:mend]
                external_mouth_shape = mouth[:-7]
                
                # Create mouth mask
                kernel = np.ones((3, 3), np.uint8)
                mouth_mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mouth_mask, external_mouth_shape, 255)
                mouth_dilated = cv2.dilate(mouth_mask, kernel, iterations=8)
                
                # Find contours
                mouth_contours, _ = cv2.findContours(mouth_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(mouth_contours) == 0:
                    continue
                
                external_mouth_shape_extended = mouth_contours[0]
                
                # Draw mask
                cv2.fillConvexPoly(mask, np.array(external_mouth_shape_extended), (255, 255, 255))
                mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
                
                # Save mask
                mask_path = masks_dir / f"image_{str(frame_number).rjust(5, '0')}.png"
                cv2.imwrite(str(mask_path), mask_blur)
                
                # Composite Wav2Lip mouth onto original frame
                mask_normalized = mask_blur / 255.0
                frame_wav2lip_rgb = cv2.cvtColor(frame_wav2lip, cv2.COLOR_BGR2RGB)
                
                # Blend the mouth region
                dst = frame_wav2lip_rgb * mask_normalized
                result = (result * (1 - mask_normalized) + dst).astype(np.uint8)
            
            # Convert back to BGR for saving with OpenCV
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_image), result_bgr)
            
            # Enhance with ControlNet if enabled
            if use_controlnet and payload and frame_number < 10:  # Limit ControlNet calls for efficiency
                enhance_image_with_controlnet(
                    output_image,
                    masks_dir / f"image_{str(frame_number).rjust(5, '0')}.png",
                    payload,
                    frame_number,
                    output_dir
                )
            
            frame_number += 1
        
        # Release video streams
        vs.release()
        vi.release()
        
        print("[INFO] Creating output video...")
        
        # Create video from images
        temp_video = output_dir / "enhanced_video.avi"
        create_video_from_images(images_dir, temp_video, original_video, frame_number)
        
        # Handle audio
        if has_audio(wav2lip_video):
            print("[INFO] Extracting audio from Wav2Lip video...")
            audio_file = output_dir / "audio.aac"
            extract_audio_from_video(wav2lip_video, audio_file)
            
            print("[INFO] Adding audio to enhanced video...")
            add_audio_to_video(temp_video, audio_file, output_video)
            
            # Cleanup temp audio
            audio_file.unlink(missing_ok=True)
        else:
            print("[INFO] No audio found in input video")
            temp_video.rename(output_video)
        
        # Cleanup temp video
        temp_video.unlink(missing_ok=True)
        
        print(f"[INFO] Enhancement complete! Output saved to: {output_video}")
        return Path(output_video)
        
    except Exception as e:
        vs.release()
        vi.release()
        raise Wav2LipUHQError(f"UHQ enhancement failed: {str(e)}")


def cleanup_enhancement_cache(output_dir: Path):
    """Clean up temporary files from enhancement process"""
    try:
        import shutil
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if masks_dir.exists():
            shutil.rmtree(masks_dir)
            
        print("[INFO] Cleanup complete")
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}")
