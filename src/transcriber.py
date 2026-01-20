"""
Speech-to-Text module using MLX-Whisper (Apple Silicon GPU).
Transcribes German speech to text.

NOTE: MLX is not thread-safe, so we use a Lock to serialize transcription calls.
"""

import numpy as np
import threading
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Global lock for MLX transcription (MLX is not thread-safe)
_mlx_lock = threading.Lock()


class Transcriber:
    """Transcribes speech using MLX-Whisper (Apple Silicon GPU)."""
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",  # Kept for compatibility, ignored
        compute_type: str = "int8",  # Kept for compatibility, ignored
        language: str = "de",
    ):
        """
        Initialize MLX transcriber.
        """
        self.model_size = model_size
        self.language = language
        self._loaded = False
        
        # Map model sizes to HuggingFace MLX model names
        self.model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }
        
        self.model_name = self.model_map.get(model_size, f"mlx-community/whisper-{model_size}-mlx")
        logger.info(f"Transcriber initialized (MLX model: {self.model_name})")
        
        # Verify MLX is installed
        try:
            import mlx_whisper
        except ImportError:
            raise RuntimeError(
                "mlx-whisper not found. This project requires Apple Silicon and MLX.\n"
                "Please run: pip install mlx-whisper"
            )

    def load_model(self) -> None:
        """Pre-load the model (optional, happens on first transcribe)."""
        if self._loaded:
            return
        
        logger.info(f"Pre-loading MLX Whisper model: {self.model_name}")
        import mlx_whisper
        
        # Do a tiny test to trigger model load (with lock)
        test_audio = np.zeros(16000, dtype=np.float32)
        try:
            with _mlx_lock:
                mlx_whisper.transcribe(
                    test_audio,
                    path_or_hf_repo=self.model_name,
                    language=self.language,
                )
            self._loaded = True
            logger.info("MLX Whisper model loaded")
        except Exception as e:
            logger.warning(f"Model pre-load failed: {e}")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[str, float]:
        """
        Transcribe audio to text using MLX-Whisper (GPU accelerated).
        """
        import mlx_whisper
        
        if len(audio) == 0:
            return "", 0.0
        
        # Ensure correct sample rate
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Transcribe with MLX (thread-safe via lock)
        with _mlx_lock:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_name,
                language=self.language,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
        
        text = result.get("text", "").strip()
        
        # Calculate confidence from segments
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s.get("avg_logprob", -1) for s in segments) / len(segments)
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob / 5))
        else:
            confidence = 0.5 if text else 0.0
        
        return text, confidence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Transcriber with MLX support...")
    
    transcriber = Transcriber(model_size="small")
    transcriber.load_model()
    
    sample_rate = 16000
    audio = np.random.randn(sample_rate * 3).astype(np.float32) * 0.1
    
    import time
    start = time.time()
    text, confidence = transcriber.transcribe(audio)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Result: '{text}'")
