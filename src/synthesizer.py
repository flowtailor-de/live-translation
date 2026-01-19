"""
Text-to-Speech module using Piper.
Generates Farsi audio from text.
"""

import numpy as np
import subprocess
import tempfile
import wave
import os
import urllib.request
import json
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Piper voice model URLs from HuggingFace
PIPER_VOICES = {
    "fa_IR-amir-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx.json",
    },
    "fa_IR-gyro-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/gyro/medium/fa_IR-gyro-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/gyro/medium/fa_IR-gyro-medium.onnx.json",
    },
}


def get_piper_cache_dir() -> Path:
    """Get the cache directory for Piper models."""
    cache_dir = Path.home() / ".cache" / "piper-voices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress."""
    logger.info(f"{desc}: {url}")
    
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded: {dest.name}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def ensure_voice_downloaded(voice_name: str) -> Tuple[Path, Path]:
    """
    Ensure the voice model is downloaded.
    
    Returns:
        Tuple of (onnx_path, json_path)
    """
    if voice_name not in PIPER_VOICES:
        raise ValueError(f"Unknown voice: {voice_name}. Available: {list(PIPER_VOICES.keys())}")
    
    cache_dir = get_piper_cache_dir()
    voice_urls = PIPER_VOICES[voice_name]
    
    onnx_path = cache_dir / f"{voice_name}.onnx"
    json_path = cache_dir / f"{voice_name}.onnx.json"
    
    # Download ONNX model if not exists
    if not onnx_path.exists():
        download_file(voice_urls["onnx"], onnx_path, f"Downloading {voice_name} ONNX model")
    
    # Download JSON config if not exists
    if not json_path.exists():
        download_file(voice_urls["json"], json_path, f"Downloading {voice_name} config")
    
    return onnx_path, json_path


class Synthesizer:
    """Synthesizes Farsi speech using Piper TTS."""
    
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
        
        self.model_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.piper_voice = None
        
        logger.info(f"Synthesizer initialized with voice: {voice}")
    
    def _ensure_model(self) -> Tuple[Path, Path]:
        """Ensure the voice model is downloaded and return paths."""
        if self.model_path is None or self.config_path is None:
            self.model_path, self.config_path = ensure_voice_downloaded(self.voice)
        return self.model_path, self.config_path
    
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using Piper CLI.
        
        Args:
            text: Farsi text to synthesize
            
        Returns:
            Tuple of (audio data as numpy array, sample rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.sample_rate
        
        # Ensure model is downloaded
        onnx_path, json_path = self._ensure_model()
        
        # Use CLI
        return self._synthesize_cli(text, onnx_path)
    

    
    def _synthesize_cli(self, text: str, model_path: Path) -> Tuple[np.ndarray, int]:
        """Synthesize using Piper CLI."""
        # Find piper executable
        piper_path = self._find_piper()
        piper_dir = os.path.dirname(piper_path)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Prepare environment with DYLD_LIBRARY_PATH for macOS
            env = os.environ.copy()
            if "DYLD_LIBRARY_PATH" in env:
                env["DYLD_LIBRARY_PATH"] = f"{piper_dir}:{env['DYLD_LIBRARY_PATH']}"
            else:
                env["DYLD_LIBRARY_PATH"] = piper_dir
                
            process = subprocess.run(
                [
                    piper_path,
                    "--model", str(model_path),
                    "--output_file", temp_path,
                    "--length_scale", str(self.length_scale),
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                env=env,
            )
            
            if process.returncode != 0:
                error = process.stderr.decode()
                raise RuntimeError(f"Piper CLI failed: {error}")
            
            audio, sr = self._read_wav(temp_path)
            return audio, sr
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _find_piper(self) -> str:
        """Find Piper executable."""
        # 1. Project local bin (highest priority)
        project_root = Path(__file__).parent.parent
        local_bin = project_root / "bin" / "piper" / "piper"
        if local_bin.exists():
            return str(local_bin)
            
        # 2. System path
        try:
            result = subprocess.run(["which", "piper"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # 3. Common paths
        common_paths = [
            os.path.expanduser("~/.local/bin/piper"),
            "/usr/local/bin/piper",
            "/opt/homebrew/bin/piper",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Piper binary not found at {local_bin} or in PATH. run ./bin/setup.sh"
        )
    
    def _read_wav(self, path: str) -> Tuple[np.ndarray, int]:
        """Read a WAV file and return audio data."""
        with wave.open(path, "rb") as wav:
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
        audio, sr = self.synthesize(text)
        
        if len(audio) == 0:
            return b""
        
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            with wave.open(temp_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(audio_int16.tobytes())
            
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


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
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            with wave.open(temp_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(audio_int16.tobytes())
            
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Synthesizer...")
    
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
