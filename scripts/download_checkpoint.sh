#!/bin/bash
# Script to download Wav2Lip checkpoint file

set -e

CHECKPOINT_DIR="sd-wav2lip-uhq/scripts/wav2lip/checkpoints"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/wav2lip_gan.pth"
CHECKPOINT_URL="https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW"

echo "Downloading Wav2Lip GAN checkpoint..."
echo "URL: ${CHECKPOINT_URL}"
echo "Destination: ${CHECKPOINT_FILE}"

# Create directory if it doesn't exist
mkdir -p "${CHECKPOINT_DIR}"

# Check if file already exists and has content
if [ -f "${CHECKPOINT_FILE}" ]; then
    FILE_SIZE=$(stat -f%z "${CHECKPOINT_FILE}" 2>/dev/null || stat -c%s "${CHECKPOINT_FILE}" 2>/dev/null || echo "0")
    if [ "${FILE_SIZE}" -gt 1024 ]; then
        echo "Checkpoint file already exists and appears valid (size: ${FILE_SIZE} bytes)"
        echo "To re-download, delete the file first: rm ${CHECKPOINT_FILE}"
        exit 0
    else
        echo "Existing file is too small (${FILE_SIZE} bytes), removing it..."
        rm -f "${CHECKPOINT_FILE}"
    fi
fi

# Try downloading with wget first (better for SharePoint links)
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget --no-check-certificate -O "${CHECKPOINT_FILE}" "${CHECKPOINT_URL}" || {
        echo "wget failed, trying curl..."
        curl -L -o "${CHECKPOINT_FILE}" "${CHECKPOINT_URL}" || {
            echo "ERROR: Both wget and curl failed to download the checkpoint."
            echo ""
            echo "Please download manually:"
            echo "1. Open this URL in your browser:"
            echo "   ${CHECKPOINT_URL}"
            echo "2. Download the file and save it as:"
            echo "   ${CHECKPOINT_FILE}"
            exit 1
        }
    }
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L -o "${CHECKPOINT_FILE}" "${CHECKPOINT_URL}" || {
        echo "ERROR: curl failed to download the checkpoint."
        echo ""
        echo "Please download manually:"
        echo "1. Open this URL in your browser:"
        echo "   ${CHECKPOINT_URL}"
        echo "2. Download the file and save it as:"
        echo "   ${CHECKPOINT_FILE}"
        exit 1
    }
else
    echo "ERROR: Neither wget nor curl is available."
    echo ""
    echo "Please download manually:"
    echo "1. Open this URL in your browser:"
    echo "   ${CHECKPOINT_URL}"
    echo "2. Download the file and save it as:"
    echo "   ${CHECKPOINT_FILE}"
    exit 1
fi

# Verify download
if [ -f "${CHECKPOINT_FILE}" ]; then
    FILE_SIZE=$(stat -f%z "${CHECKPOINT_FILE}" 2>/dev/null || stat -c%s "${CHECKPOINT_FILE}" 2>/dev/null || echo "0")
    if [ "${FILE_SIZE}" -lt 1024 ]; then
        echo "WARNING: Downloaded file is very small (${FILE_SIZE} bytes). It may be corrupted."
        echo "Please check the file and re-download if necessary."
        exit 1
    else
        echo "✓ Successfully downloaded checkpoint file (size: ${FILE_SIZE} bytes)"
        echo "File location: ${CHECKPOINT_FILE}"
    fi
else
    echo "ERROR: Download failed - file not found"
    exit 1
fi

