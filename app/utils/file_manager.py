import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
import aiofiles
from fastapi import UploadFile

from app.config import UPLOAD_DIR, OUTPUT_DIR, WAV2LIP_TEMP_DIR, MAX_FILE_SIZE, ALLOWED_VIDEO_FORMATS


async def save_uploaded_file(upload_file: UploadFile, directory: Path = UPLOAD_DIR) -> Path:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        upload_file: FastAPI UploadFile object
        directory: Directory to save the file to
        
    Returns:
        Path to the saved file
    """
    # Generate unique filename
    file_ext = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = directory / unique_filename
    
    # Ensure directory exists
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes")
        await f.write(content)
    
    return file_path


def validate_video_file(file_path: Path) -> bool:
    """
    Validate that the file is a supported video format.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
    file_ext = file_path.suffix.lower().lstrip('.')
    if file_ext not in ALLOWED_VIDEO_FORMATS:
        raise ValueError(
            f"Unsupported video format: {file_ext}. "
            f"Allowed formats: {', '.join(ALLOWED_VIDEO_FORMATS)}"
        )
    
    return True


def create_output_path(prefix: str = "output", extension: str = "mp4") -> Path:
    """
    Create a unique output file path.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Path to output file
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unique_filename = f"{prefix}_{uuid.uuid4()}.{extension}"
    return OUTPUT_DIR / unique_filename


def create_temp_file(directory: Path, prefix: str = "temp", extension: str = "wav") -> Path:
    """
    Create a temporary file path.
    
    Args:
        directory: Directory for temporary file
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Path to temporary file
    """
    directory.mkdir(parents=True, exist_ok=True)
    unique_filename = f"{prefix}_{uuid.uuid4()}.{extension}"
    return directory / unique_filename


def cleanup_file(file_path: Path) -> None:
    """
    Delete a file if it exists.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")


def cleanup_directory(directory: Path, pattern: Optional[str] = None) -> None:
    """
    Clean up files in a directory.
    
    Args:
        directory: Directory to clean
        pattern: Optional pattern to match files (e.g., "*.wav")
    """
    try:
        if not directory.exists():
            return
        
        if pattern:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
        else:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up directory {directory}: {e}")


def ensure_wav2lip_temp_dir() -> Path:
    """
    Ensure Wav2Lip temp directory exists and is clean.
    
    Returns:
        Path to Wav2Lip temp directory
    """
    WAV2LIP_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return WAV2LIP_TEMP_DIR

