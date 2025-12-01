import httpx
import aiofiles
from pathlib import Path
from typing import Optional
import subprocess
import os

from app.config import (
    AISHA_AI_API_KEY,
    AISHA_AI_API_URL,
    AISHA_AI_VOICE_ID
)


class TTSServiceError(Exception):
    """Custom exception for TTS service errors"""
    pass


async def generate_audio(text: str, output_path: Path) -> Path:
    """
    Generate audio from Uzbek text using Aisha AI TTS service.
    
    Args:
        text: Uzbek text to convert to speech
        output_path: Path where the audio file will be saved (should be .wav)
        
    Returns:
        Path to the generated audio file
        
    Raises:
        TTSServiceError: If TTS generation fails
    """
    if not AISHA_AI_API_KEY:
        raise TTSServiceError("Aisha AI API key is not configured")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure output is WAV format (Wav2Lip requires WAV)
    if output_path.suffix.lower() != '.wav':
        output_path = output_path.with_suffix('.wav')
    
    try:
        # Call Aisha AI TTS API
        # Note: API structure may vary - adjust based on actual Aisha AI API documentation
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Common TTS API patterns - adjust based on actual API
            payload = {
                "text": text,
                "voice_id": AISHA_AI_VOICE_ID,
                "format": "wav",  # Request WAV format for Wav2Lip
                "sample_rate": 16000  # Wav2Lip expects 16kHz
            }
            
            headers = {
                "Authorization": f"Bearer {AISHA_AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Try POST request with JSON payload
            response = await client.post(
                AISHA_AI_API_URL,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                # If API returns audio data directly
                audio_data = response.content
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(audio_data)
            elif response.status_code == 202:
                # If API returns a job ID (async processing)
                job_data = response.json()
                job_id = job_data.get("job_id") or job_data.get("id")
                
                if not job_id:
                    raise TTSServiceError("API returned 202 but no job ID found")
                
                # Poll for completion
                audio_url = await _poll_tts_job(client, job_id, headers)
                await _download_audio_file(client, audio_url, output_path)
            else:
                # Try alternative: API might return audio URL
                try:
                    result = response.json()
                    audio_url = result.get("audio_url") or result.get("url") or result.get("file_url")
                    if audio_url:
                        await _download_audio_file(client, audio_url, output_path)
                    else:
                        raise TTSServiceError(
                            f"Aisha AI API returned status {response.status_code}: {response.text}"
                        )
                except ValueError:
                    raise TTSServiceError(
                        f"Aisha AI API returned status {response.status_code}: {response.text}"
                    )
        
        # Ensure audio is in correct format for Wav2Lip (16kHz WAV)
        await _ensure_wav_format(output_path, sample_rate=16000)
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TTSServiceError("Generated audio file is empty or does not exist")
        
        return output_path
        
    except httpx.TimeoutException:
        raise TTSServiceError("TTS API request timed out")
    except httpx.RequestError as e:
        raise TTSServiceError(f"TTS API request failed: {str(e)}")
    except Exception as e:
        raise TTSServiceError(f"TTS generation failed: {str(e)}")


async def _poll_tts_job(client: httpx.AsyncClient, job_id: str, headers: dict, max_attempts: int = 60) -> str:
    """
    Poll TTS job until completion.
    
    Args:
        client: HTTP client
        job_id: Job ID to poll
        headers: Request headers
        max_attempts: Maximum polling attempts
        
    Returns:
        URL to the generated audio file
    """
    import asyncio
    
    status_url = f"{AISHA_AI_API_URL}/jobs/{job_id}"
    
    for attempt in range(max_attempts):
        await asyncio.sleep(2)  # Wait 2 seconds between polls
        
        response = await client.get(status_url, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "").lower()
            
            if status == "completed" or status == "success":
                audio_url = result.get("audio_url") or result.get("url") or result.get("file_url")
                if audio_url:
                    return audio_url
                else:
                    raise TTSServiceError("Job completed but no audio URL found")
            elif status == "failed" or status == "error":
                error_msg = result.get("error", "Unknown error")
                raise TTSServiceError(f"TTS job failed: {error_msg}")
            # Continue polling if status is "processing" or "pending"
    
    raise TTSServiceError("TTS job polling timed out")


async def _download_audio_file(client: httpx.AsyncClient, audio_url: str, output_path: Path) -> None:
    """
    Download audio file from URL.
    
    Args:
        client: HTTP client
        audio_url: URL to audio file
        output_path: Path to save the file
    """
    response = await client.get(audio_url, timeout=60.0)
    response.raise_for_status()
    
    async with aiofiles.open(output_path, 'wb') as f:
        await f.write(response.content)


async def _ensure_wav_format(audio_path: Path, sample_rate: int = 16000) -> None:
    """
    Ensure audio is in WAV format with correct sample rate using ffmpeg.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16000 for Wav2Lip)
    """
    # Check if file is already WAV with correct sample rate
    # For simplicity, we'll always convert to ensure compatibility
    temp_path = audio_path.with_suffix('.temp.wav')
    
    try:
        # Use ffmpeg to convert to WAV with correct sample rate
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', str(audio_path),
            '-ar', str(sample_rate),  # Set sample rate
            '-ac', '1',  # Mono channel
            '-f', 'wav',  # WAV format
            str(temp_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Replace original with converted file
        if temp_path.exists():
            audio_path.unlink(missing_ok=True)
            temp_path.rename(audio_path)
            
    except subprocess.CalledProcessError as e:
        # If conversion fails, check if file is already in correct format
        if audio_path.suffix.lower() == '.wav':
            # Assume it's already correct format
            temp_path.unlink(missing_ok=True)
        else:
            raise TTSServiceError(f"Failed to convert audio to WAV format: {e.stderr}")
    except FileNotFoundError:
        # ffmpeg not found - assume file is already correct
        temp_path.unlink(missing_ok=True)
        if audio_path.suffix.lower() != '.wav':
            raise TTSServiceError("ffmpeg not found and audio is not in WAV format")

