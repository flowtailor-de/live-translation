"""
Model download script.
Downloads all required models for offline use.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_whisper(model_size: str = "medium"):
    """Download Whisper model."""
    logger.info(f"Downloading Whisper {model_size} model...")
    
    from faster_whisper import WhisperModel
    
    # This will download the model to cache
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    del model
    
    logger.info("Whisper model downloaded")


def download_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    """Download NLLB translation model."""
    logger.info(f"Downloading NLLB model: {model_name}...")
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Download tokenizer and model
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    logger.info("NLLB model downloaded")


def download_silero_vad():
    """Download Silero VAD model."""
    logger.info("Downloading Silero VAD model...")
    
    import torch
    
    torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
    )
    
    logger.info("Silero VAD downloaded")


def download_piper_voice(voice: str = "fa_IR-amir-medium"):
    """Download Piper voice model."""
    logger.info(f"Downloading Piper voice: {voice}...")
    
    try:
        # Try using piper-tts package
        from piper import PiperVoice
        PiperVoice.load(voice)
        logger.info("Piper voice downloaded")
    except Exception as e:
        logger.warning(f"Could not download Piper voice automatically: {e}")
        logger.info("You may need to download manually from: https://github.com/rhasspy/piper/releases")


def main():
    """Download all models."""
    print("=" * 60)
    print("Live Translation System - Model Downloader")
    print("=" * 60)
    print("\nThis will download approximately 4-5 GB of model files.")
    print("Models will be cached for offline use.\n")
    
    try:
        # 1. Whisper
        download_whisper("medium")
        
        # 2. NLLB
        download_nllb()
        
        # 3. Silero VAD
        download_silero_vad()
        
        # 4. Piper TTS
        download_piper_voice()
        
        print("\n" + "=" * 60)
        print("All models downloaded successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
