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
    
    # Use MLX-Whisper to download model
    try:
        from src.transcriber import Transcriber
        transcriber = Transcriber(model_size=model_size)
        transcriber.load_model()
        logger.info("Whisper model (MLX) downloaded")
    except Exception as e:
        logger.warning(f"Failed to download MLX Whisper model: {e}")


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
    
    # URLs
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR"
    voice_name_parts = voice.split("-")
    if len(voice_name_parts) >= 3:
        speaker = voice_name_parts[1]
        quality = voice_name_parts[2]
        onnx_url = f"{base_url}/{speaker}/{quality}/{voice}.onnx"
        json_url = f"{base_url}/{speaker}/{quality}/{voice}.onnx.json"
        
        # Download paths
        from pathlib import Path
        import urllib.request
        
        cache_dir = Path.home() / ".cache" / "piper-voices"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        onnx_path = cache_dir / f"{voice}.onnx"
        json_path = cache_dir / f"{voice}.onnx.json"
        
        def _dl(url, dest):
            if not dest.exists():
                logger.info(f"Downloading {url}...")
                urllib.request.urlretrieve(url, dest)
        
        try:
            _dl(onnx_url, onnx_path)
            _dl(json_url, json_path)
            logger.info("Piper voice downloaded")
        except Exception as e:
            logger.warning(f"Failed to download Piper voice: {e}")
            logger.info("You may need to download manually from HuggingFace.")
    else:
        logger.warning(f"Could not parse voice name: {voice}")


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
