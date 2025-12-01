from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback

from app.models import VideoProcessResponse, ProcessingStatus, HealthResponse
from app.services.wav2lip_service import process_video, validate_wav2lip_setup, Wav2LipServiceError
from app.utils.file_manager import (
    save_uploaded_file,
    validate_video_file,
    validate_audio_file,
    create_output_path,
    cleanup_file,
    ensure_wav2lip_temp_dir
)
from app.config import WAV2LIP_CHECKPOINT, BASE_DIR

app = FastAPI(
    title="Wav2Lip API",
    description="API for generating lip-synced videos from video and audio files using Wav2Lip",
    version="1.0.0"
)

# Mount static files directory
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML page"""
    dashboard_path = BASE_DIR / "static" / "index.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Please ensure static/index.html exists.</p>",
            status_code=404
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    wav2lip_valid, wav2lip_error = validate_wav2lip_setup()
    checkpoint_path = Path(WAV2LIP_CHECKPOINT)
    checkpoint_exists = checkpoint_path.exists()
    
    return HealthResponse(
        status="ok",
        wav2lip_available=wav2lip_valid,
        checkpoint_exists=checkpoint_exists,
        aisha_ai_configured=False,
        wav2lip_error=wav2lip_error,
        checkpoint_path=str(checkpoint_path)
    )


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video_endpoint(
    video: UploadFile = File(..., description="Video file to process"),
    audio: UploadFile = File(..., description="Audio file to lip-sync with video")
):
    """
    Process a video file with Wav2Lip lip-sync using an uploaded audio file.
    
    This endpoint:
    1. Accepts a video file and an audio file
    2. Processes video with Wav2Lip to lip-sync with the provided audio
    3. Returns the processed video file
    
    Args:
        video: Video file (MP4, AVI, MOV, etc.)
        audio: Audio file (WAV, MP3, M4A, etc.)
        
    Returns:
        Processed video file download
    """
    video_path = None
    audio_path = None
    output_path = None
    
    try:
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
        
        # Save uploaded audio file
        audio_path = await save_uploaded_file(audio, directory=ensure_wav2lip_temp_dir())
        
        # Validate audio file
        try:
            validate_audio_file(audio_path)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
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
            error_msg = str(e)
            if "Face not detected" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Face not detected in video. Please ensure the video contains a clear, well-lit face visible throughout all frames. Try using a head-and-shoulders video with good lighting."
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Wav2Lip processing failed: {error_msg}"
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
    uvicorn.run(app, host="0.0.0.0", port=6070)

