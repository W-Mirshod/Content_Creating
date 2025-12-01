from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import traceback

from app.models import VideoProcessResponse, ProcessingStatus, HealthResponse
from app.services.tts_service import generate_audio, TTSServiceError
from app.services.wav2lip_service import process_video, validate_wav2lip_setup, Wav2LipServiceError
from app.utils.file_manager import (
    save_uploaded_file,
    validate_video_file,
    create_output_path,
    create_temp_file,
    cleanup_file,
    ensure_wav2lip_temp_dir
)
from app.config import AISHA_AI_API_KEY, WAV2LIP_CHECKPOINT

app = FastAPI(
    title="Uzbek TTS Wav2Lip API",
    description="API for generating lip-synced videos from Uzbek text using Aisha AI TTS and Wav2Lip",
    version="1.0.0"
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    wav2lip_valid, wav2lip_error = validate_wav2lip_setup()
    checkpoint_exists = Path(WAV2LIP_CHECKPOINT).exists() if wav2lip_valid else False
    
    return HealthResponse(
        status="ok",
        wav2lip_available=wav2lip_valid,
        checkpoint_exists=checkpoint_exists,
        aisha_ai_configured=bool(AISHA_AI_API_KEY)
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return await root()


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video_endpoint(
    video: UploadFile = File(..., description="Video file to process"),
    text: str = Form(..., description="Uzbek text to convert to speech and lip-sync")
):
    """
    Process a video file with Uzbek TTS and Wav2Lip lip-sync.
    
    This endpoint:
    1. Accepts a video file and Uzbek text
    2. Generates TTS audio using Aisha AI
    3. Processes video with Wav2Lip to lip-sync with the generated audio
    4. Returns the processed video file
    
    Args:
        video: Video file (MP4, AVI, MOV, etc.)
        text: Uzbek text to convert to speech
        
    Returns:
        Processed video file download
    """
    video_path = None
    audio_path = None
    output_path = None
    
    try:
        # Validate text input
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )
        
        text = text.strip()
        
        # Save uploaded video file
        video_path = await save_uploaded_file(video)
        
        # Validate video file
        try:
            validate_video_file(video_path)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Ensure Wav2Lip temp directory exists
        ensure_wav2lip_temp_dir()
        
        # Generate TTS audio
        try:
            audio_path = create_temp_file(
                ensure_wav2lip_temp_dir(),
                prefix="tts_audio",
                extension="wav"
            )
            await generate_audio(text, audio_path)
        except TTSServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS generation failed: {str(e)}"
            )
        
        # Process video with Wav2Lip
        try:
            output_path = create_output_path(prefix="lip_synced", extension="mp4")
            process_video(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path
            )
        except Wav2LipServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Wav2Lip processing failed: {str(e)}"
            )
        
        # Return processed video file
        if not output_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video processing completed but output file not found"
            )
        
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=output_path.name,
            headers={
                "Content-Disposition": f"attachment; filename={output_path.name}"
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        error_detail = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        if video_path and video_path.exists():
            cleanup_file(video_path)
        if audio_path and audio_path.exists():
            cleanup_file(audio_path)
        # Note: output_path is kept for download, will be cleaned up later if needed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

