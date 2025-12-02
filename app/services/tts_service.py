"""
[DEPRECATED] Aisha AI TTS Service

This module is kept for backward compatibility but is NO LONGER ACTIVELY MAINTAINED.

The project focus has shifted from TTS+Wav2Lip integration to high-quality video 
lip-sync enhancement using Wav2Lip-UHQ.

If you need TTS functionality:
1. Use an external TTS service (Aisha AI, Google Cloud TTS, etc.)
2. Generate audio files with your TTS service
3. Use the /process-video endpoint with pre-generated audio files

For more information, see DEPRECATION_NOTICE.md
"""

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
    
    Based on official API documentation: https://aisha.group/en/api-documentation
    
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
    
    if not AISHA_AI_API_URL:
        raise TTSServiceError("Aisha AI API URL is not configured")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure output is WAV format (Wav2Lip requires WAV)
    if output_path.suffix.lower() != '.wav':
        output_path = output_path.with_suffix('.wav')
    
    try:
        print(f"Calling Aisha AI TTS API: {AISHA_AI_API_URL}")
        
        # According to API docs: POST with multipart/form-data
        # Endpoint: https://back.aisha.group/api/v1/tts/post/
        # Headers: x-api-key, Content-Type: multipart/form-data, X-Channels, X-Quality, X-Rate, X-Format
        # Form fields: transcript, language, run_diarization, model, mood (optional)
        
        # Map voice_id to model (default to gulnoza if not specified)
        model = "gulnoza"  # Default model
        if AISHA_AI_VOICE_ID and AISHA_AI_VOICE_ID.lower() in ["jaxongir", "jaxon"]:
            model = "jaxongir"
        
        # Language: uz, en, ru
        language = "uz"  # Default to Uzbek
        
        # Prepare form data
        form_data = {
            "transcript": text,
            "language": language,
            "run_diarization": "false",
            "model": model,
            "mood": "neutral"  # Optional: happy, neutral, sad
        }
        
        # Prepare headers
        headers = {
            "x-api-key": AISHA_AI_API_KEY,
            "X-Channels": "stereo",
            "X-Quality": "64k",
            "X-Rate": "16000",  # Sample rate for Wav2Lip
            "X-Format": "mp3"  # API returns MP3, we'll convert to WAV
        }
        
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.post(
                AISHA_AI_API_URL,
                data=form_data,
                headers=headers
            )
            
            if response.status_code == 200:
                # API returns audio data directly
                audio_data = response.content
                
                if len(audio_data) == 0:
                    raise TTSServiceError("API returned empty audio data")
                
                # Save the audio (likely MP3 format from API)
                temp_audio_path = output_path.with_suffix('.mp3')
                async with aiofiles.open(temp_audio_path, 'wb') as f:
                    await f.write(audio_data)
                
                # Convert to WAV format with correct sample rate for Wav2Lip
                await _convert_to_wav(temp_audio_path, output_path, sample_rate=16000)
                
                # Clean up temp MP3 file
                if temp_audio_path.exists():
                    temp_audio_path.unlink(missing_ok=True)
                    
            elif response.status_code == 201:
                # API returns 201 with JSON containing audio_path URL
                try:
                    result = response.json()
                    audio_url = result.get("audio_path") or result.get("audio_url") or result.get("url")
                    
                    if not audio_url:
                        raise TTSServiceError(
                            f"API returned 201 but no audio_path in response: {result}"
                        )
                    
                    print(f"Downloading audio from: {audio_url}")
                    
                    # Download audio from the provided URL
                    temp_audio_path = output_path.with_suffix('.mp3')
                    await _download_audio_file(client, audio_url, temp_audio_path)
                    
                    # Convert to WAV format with correct sample rate for Wav2Lip
                    await _convert_to_wav(temp_audio_path, output_path, sample_rate=16000)
                    
                    # Clean up temp MP3 file
                    if temp_audio_path.exists():
                        temp_audio_path.unlink(missing_ok=True)
                        
                except ValueError as e:
                    raise TTSServiceError(
                        f"API returned 201 but response is not valid JSON: {response.text}"
                    )
                except Exception as e:
                    raise TTSServiceError(
                        f"Failed to process API response (201): {str(e)}"
                    )
            else:
                error_msg = f"Aisha AI API returned status {response.status_code}"
                try:
                    error_detail = response.text
                    error_msg += f": {error_detail}"
                except:
                    pass
                raise TTSServiceError(error_msg)
        
        # Verify output file exists and is not empty
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TTSServiceError("Generated audio file is empty or does not exist")
        
        print(f"Successfully generated audio: {output_path} ({output_path.stat().st_size} bytes)")
        return output_path
        
    except httpx.TimeoutException:
        raise TTSServiceError("TTS API request timed out after 120 seconds")
    except httpx.ConnectError as e:
        raise TTSServiceError(
            f"TTS API connection failed. Cannot reach {AISHA_AI_API_URL}. "
            f"Error: {str(e)}. Please check your network connection and verify the API URL is correct."
        )
    except httpx.RequestError as e:
        raise TTSServiceError(f"TTS API request failed: {str(e)}")
    except TTSServiceError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        raise TTSServiceError(f"TTS generation failed: {str(e)}")


async def _download_audio_file(client: httpx.AsyncClient, audio_url: str, output_path: Path) -> None:
    """
    Download audio file from URL.
    
    Args:
        client: HTTP client
        audio_url: URL to audio file
        output_path: Path to save the file
    """
    try:
        response = await client.get(audio_url, timeout=120.0, follow_redirects=True)
        response.raise_for_status()
        
        if len(response.content) == 0:
            raise TTSServiceError(f"Downloaded audio file from {audio_url} is empty")
        
        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(response.content)
            
        print(f"Downloaded audio file: {output_path} ({output_path.stat().st_size} bytes)")
    except httpx.HTTPStatusError as e:
        raise TTSServiceError(f"Failed to download audio from {audio_url}: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise TTSServiceError(f"Failed to download audio from {audio_url}: {str(e)}")


async def _convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    """
    Convert audio file to WAV format with specified sample rate using ffmpeg.
    
    Args:
        input_path: Path to input audio file (MP3, etc.)
        output_path: Path to output WAV file
        sample_rate: Target sample rate (default 16000 for Wav2Lip)
    """
    try:
        # Use ffmpeg to convert to WAV with correct sample rate
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', str(input_path),
            '-ar', str(sample_rate),  # Set sample rate
            '-ac', '1',  # Mono channel (Wav2Lip works better with mono)
            '-f', 'wav',  # WAV format
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if not output_path.exists():
            raise TTSServiceError(f"FFmpeg conversion failed: output file not created")
            
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to convert audio to WAV format"
        if e.stderr:
            error_msg += f": {e.stderr}"
        raise TTSServiceError(error_msg)
    except FileNotFoundError:
        raise TTSServiceError(
            "ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg"
        )
