#!/usr/bin/env python3
"""
Script to download Wav2Lip checkpoint file.
Note: SharePoint links may require manual download in a browser.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import WAV2LIP_ROOT, WAV2LIP_CHECKPOINT

CHECKPOINT_URL = "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW"
CHECKPOINT_PATH = Path(WAV2LIP_CHECKPOINT)

def main():
    print("=" * 70)
    print("Wav2Lip Checkpoint Downloader")
    print("=" * 70)
    print()
    print(f"Checkpoint URL: {CHECKPOINT_URL}")
    print(f"Destination: {CHECKPOINT_PATH}")
    print()
    
    # Create directory if it doesn't exist
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if CHECKPOINT_PATH.exists():
        file_size = CHECKPOINT_PATH.stat().st_size
        if file_size > 1024 * 1024:  # More than 1MB
            print(f"✓ Checkpoint file already exists (size: {file_size:,} bytes)")
            print("  If you want to re-download, delete it first:")
            print(f"  rm {CHECKPOINT_PATH}")
            return 0
        else:
            print(f"⚠ Existing file is too small ({file_size} bytes), removing it...")
            CHECKPOINT_PATH.unlink()
    
    print("Attempting to download checkpoint file...")
    print()
    print("NOTE: SharePoint links often require authentication.")
    print("If automatic download fails, please download manually:")
    print()
    print(f"1. Open this URL in your browser:")
    print(f"   {CHECKPOINT_URL}")
    print()
    print(f"2. Download the file and save it as:")
    print(f"   {CHECKPOINT_PATH}")
    print()
    
    # Try downloading with requests
    try:
        import requests
        print("Attempting download with requests library...")
        
        response = requests.get(CHECKPOINT_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            print("⚠ Warning: Server did not provide content-length")
            print("  This might be a redirect page. Please download manually.")
            return 1
        
        print(f"Downloading {total_size:,} bytes...")
        
        downloaded = 0
        with open(CHECKPOINT_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='', flush=True)
        
        print()  # New line after progress
        
        # Verify download
        if CHECKPOINT_PATH.exists():
            file_size = CHECKPOINT_PATH.stat().st_size
            if file_size > 1024 * 1024:  # More than 1MB
                print(f"✓ Successfully downloaded checkpoint file (size: {file_size:,} bytes)")
                return 0
            else:
                print(f"✗ Downloaded file is too small ({file_size} bytes). Download may have failed.")
                CHECKPOINT_PATH.unlink()
                return 1
        else:
            print("✗ Download failed - file not found")
            return 1
            
    except ImportError:
        print("⚠ requests library not available. Install it with: pip install requests")
        print("  Or download manually using the instructions above.")
        return 1
    except Exception as e:
        print(f"✗ Download failed: {type(e).__name__}: {e}")
        print()
        print("Please download manually using the instructions above.")
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        return 1

if __name__ == "__main__":
    sys.exit(main())

