from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoProcessResponse(BaseModel):
    status: ProcessingStatus
    message: str
    output_file: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    wav2lip_available: bool
    checkpoint_exists: bool
    aisha_ai_configured: bool
    wav2lip_error: Optional[str] = None
    checkpoint_path: Optional[str] = None

