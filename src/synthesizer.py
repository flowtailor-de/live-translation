"""
Text-to-Speech module using Piper.
Generates Farsi audio from text.

Uses the new piper-tts Python package (piper1-gpl) which embeds espeak-ng
directly and works natively on Apple Silicon.
"""

import numpy as np
import threading
import tempfile
import wave
import os
import io
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Piper voice model names for Farsi
# These will be downloaded automatically by piper-tts
PIPER_VOICES = {
    "fa_IR-amir-medium": "fa_IR-amir-medium",
    "fa_IR-gyro-medium": "fa_IR-gyro-medium",
}


def get_piper_cache_dir() -> Path:
    """Get the cache directory for Piper models."""
    cache_dir = Path.home() / ".cache" / "piper-voices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class Synthesizer:
    """Synthesizes Farsi speech using Piper TTS (Python API)."""
    
    def __init__(
        self,
        voice: str = "fa_IR-amir-medium",
        speaker_id: int = 0,
        length_scale: float = 1.0,
        sample_rate: int = 22050,
    ):
        """
        Initialize synthesizer.
        
        Args:
            voice: Piper voice model name
            speaker_id: Speaker ID for multi-speaker models
            length_scale: Speech speed (< 1 = faster, > 1 = slower)
            sample_rate: Output sample rate
        """
        self.voice = voice
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.sample_rate = sample_rate
        
        self._piper_voice = None
        self._model_path: Optional[Path] = None
        self._load_lock = threading.Lock()
        
        logger.info(f"Synthesizer initialized with voice: {voice}")
    
    def _ensure_voice_loaded(self):
        """Ensure the Piper voice is loaded."""
        with self._load_lock:
            if self._piper_voice is not None:
                return
                
            try:
                from piper import PiperVoice
            except ImportError:
                raise ImportError(
                    "piper-tts package not found. Install with: pip install piper-tts"
                )
            
            # Check if model exists in cache
            cache_dir = get_piper_cache_dir()
            model_path = cache_dir / f"{self.voice}.onnx"
            config_path = cache_dir / f"{self.voice}.onnx.json"
            
            # Must check for BOTH files to avoid race conditions with partial downloads
            if not model_path.exists() or not config_path.exists():
                logger.info(f"Downloading voice model: {self.voice}")
                self._download_voice(self.voice)
            
            self._model_path = model_path
            logger.info(f"Loading Piper voice from: {model_path}")
            self._piper_voice = PiperVoice.load(str(model_path))
            logger.info("Piper voice loaded successfully")
    
    def _download_voice(self, voice_name: str):
        """Download a voice model using piper's download utility."""
        import subprocess
        
        cache_dir = get_piper_cache_dir()
        
        # Try using piper.download_voices module
        try:
            # Use --download-dir instead of -d
            result = subprocess.run(
                ["python3", "-m", "piper.download_voices", voice_name, "--download-dir", str(cache_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"Downloaded voice: {voice_name}")
                return
            else:
                logger.warning(f"piper.download_voices failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not use piper.download_voices: {e}")
        
        # Fallback: download directly from HuggingFace
        import urllib.request
        
        # Approximate URL structure for piper-voices: fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx
        # But it varies. The surest way is using the python module. 
        # If that fails, we can try to construct it, but it's brittle.
        # Example: en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium.onnx
        
        parts = voice_name.split("-")
        # e.g., "fa_IR-amir-medium" -> lang="fa_IR", speaker="amir", quality="medium"
        # e.g., "en_US-lessac-medium" -> lang="en_US", speaker="lessac", quality="medium"
        
        if len(parts) >= 3:
            lang_code = parts[0] # fa_IR or en_US
            family = lang_code.split("_")[0] # fa or en
            speaker = parts[1]
            quality = parts[2]
            
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{family}/{lang_code}/{speaker}/{quality}"
            
            onnx_url = f"{base_url}/{voice_name}.onnx"
            json_url = f"{base_url}/{voice_name}.onnx.json"
            
            onnx_path = cache_dir / f"{voice_name}.onnx"
            json_path = cache_dir / f"{voice_name}.onnx.json"
            
            logger.info(f"Downloading ONNX model from: {onnx_url}")
            try:
                urllib.request.urlretrieve(onnx_url, onnx_path)
                logger.info(f"Downloading config from: {json_url}")
                urllib.request.urlretrieve(json_url, json_path)
                logger.info(f"Downloaded voice: {voice_name}")
            except Exception as e:
                logger.error(f"Fallback download failed: {e}")
                # Clean up partial downloads
                if onnx_path.exists(): onnx_path.unlink()
                if json_path.exists(): json_path.unlink()
                raise e
        else:
            raise ValueError(f"Cannot parse voice name: {voice_name}")
    
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using Piper Python API.
        
        Args:
            text: Farsi text to synthesize
            
        Returns:
            Tuple of (audio data as numpy array, sample rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.sample_rate
        
        self._ensure_voice_loaded()
        
        try:
            from piper import SynthesisConfig
            
            # Create synthesis config with our settings
            syn_config = SynthesisConfig(
                length_scale=self.length_scale,
            )
            
            # Synthesize to WAV in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, "wb") as wav_file:
                    self._piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                
                wav_buffer.seek(0)
                audio, sr = self._read_wav_from_buffer(wav_buffer)
                return audio, sr
                
        except ImportError:
            # Fallback if SynthesisConfig not available
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, "wb") as wav_file:
                    self._piper_voice.synthesize_wav(text, wav_file)
                
                wav_buffer.seek(0)
                audio, sr = self._read_wav_from_buffer(wav_buffer)
                return audio, sr
    
    def _read_wav_from_buffer(self, buffer: io.BytesIO) -> Tuple[np.ndarray, int]:
        """Read a WAV from a BytesIO buffer and return audio data."""
        with wave.open(buffer, "rb") as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_bytes = wav.readframes(n_frames)
            
            if wav.getsampwidth() == 2:
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
            elif wav.getsampwidth() == 1:
                audio = np.frombuffer(audio_bytes, dtype=np.uint8)
                audio = (audio.astype(np.float32) - 128) / 128.0
            else:
                raise ValueError(f"Unsupported sample width: {wav.getsampwidth()}")
            
            return audio, sample_rate
    
    def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize speech and return as WAV bytes."""
        if not text or not text.strip():
            return b""
            
        self._ensure_voice_loaded()
        
        try:
            from piper import SynthesisConfig
            
            syn_config = SynthesisConfig(
                length_scale=self.length_scale,
            )
            
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, "wb") as wav_file:
                    self._piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                return wav_buffer.getvalue()
                
        except ImportError:
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, "wb") as wav_file:
                    self._piper_voice.synthesize_wav(text, wav_file)
                return wav_buffer.getvalue()


class MockSynthesizer:
    """Mock synthesizer for testing without Piper."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate silence of appropriate length."""
        duration = len(text) * 0.1
        n_samples = int(duration * self.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        return audio, self.sample_rate
    
    def synthesize_to_bytes(self, text: str) -> bytes:
        audio, sr = self.synthesize(text)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(audio_int16.tobytes())
            return wav_buffer.getvalue()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Synthesizer with piper-tts Python API...")
    
    # Test downloading and synthesizing
    synth = Synthesizer()
    
    test_text = "سلام، به کلیسای ما خوش آمدید"
    print(f"Synthesizing: {test_text}")
    
    try:
        audio, sr = synth.synthesize(test_text)
        print(f"Output: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f}s)")
        
        # Save to file for manual testing
        output_path = "/tmp/test_farsi.wav"
        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
