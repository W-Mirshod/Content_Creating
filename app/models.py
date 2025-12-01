from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Uzbek text to convert to speech")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


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

