"""
Speech-to-Text module with MLX-Whisper (Apple Silicon GPU) and faster-whisper (CPU) support.
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


class MLXTranscriber:
    """Transcribes speech using MLX-Whisper (Apple Silicon GPU)."""
    
    def __init__(
        self,
        model_size: str = "small",
        language: str = "de",
    ):
        """
        Initialize MLX transcriber.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large-v3, turbo)
            language: Language code
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
        logger.info(f"MLX Transcriber initialized (model: {self.model_name})")
    
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
        
        NOTE: Uses a lock to ensure thread safety - MLX is not thread-safe.
        """
        import mlx_whisper
        
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
            confidence = 0.5
        
        return text, confidence


class Transcriber:
    """Transcribes speech using MLX (GPU) with faster-whisper (CPU) fallback."""
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "de",
        use_mlx: bool = True,
    ):
        """
        Initialize transcriber.
        
        Args:
            model_size: Whisper model size
            device: Device (auto, cpu, cuda, mps, mlx)
            compute_type: Compute type for faster-whisper
            language: Language code
            use_mlx: Whether to try MLX (Apple GPU) first
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self._use_mlx = use_mlx and device in ("auto", "mps", "mlx")
        self._mlx_available = False
        
        # Check MLX availability
        if self._use_mlx:
            try:
                import mlx_whisper
                self._mlx_available = True
                logger.info("MLX-Whisper available, will use Apple GPU")
            except ImportError:
                logger.info("MLX-Whisper not available, using faster-whisper CPU")
                self._mlx_available = False
        
        if self._mlx_available:
            self._mlx_transcriber = MLXTranscriber(model_size=model_size, language=language)
        else:
            self._mlx_transcriber = None
        
        # Determine device for faster-whisper fallback
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.compute_type = compute_type if device != "mps" else "float32"
        
        logger.info(f"Transcriber: MLX={self._mlx_available}, fallback device=cpu")
    
    def load_model(self) -> None:
        """Load the model."""
        if self._mlx_available:
            self._mlx_transcriber.load_model()
        else:
            try:
                self._load_faster_whisper()
            except ImportError:
                logger.error("faster-whisper not installed and MLX not available/selected.")
                raise RuntimeError("No transcription engine available. Install faster-whisper or use MLX.")
    
    def _load_faster_whisper(self) -> None:
        """Load faster-whisper model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading faster-whisper {self.model_size}...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning("faster-whisper package not found.")
            raise

        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=self.compute_type,
        )
        logger.info("faster-whisper model loaded")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[str, float]:
        """Transcribe audio to text."""
        
        # Try MLX first
        if self._mlx_available:
            try:
                return self._mlx_transcriber.transcribe(audio, sample_rate)
            except Exception as e:
                logger.warning(f"MLX transcription failed: {e}, falling back to CPU")
        
        # Fallback to faster-whisper
        return self._transcribe_faster_whisper(audio, sample_rate)
    
    def _transcribe_faster_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[str, float]:
        """Transcribe using faster-whisper."""
        if self.model is None:
            self._load_faster_whisper()
        
        # Resample if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        
        # Collect segments
        text_parts = []
        total_confidence = 0
        segment_count = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            total_confidence += segment.avg_logprob
            segment_count += 1
        
        text = " ".join(text_parts).strip()
        avg_confidence = (total_confidence / segment_count) if segment_count > 0 else 0
        confidence = min(1.0, max(0.0, 1.0 + avg_confidence / 5))
        
        return text, confidence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Transcriber with MLX support...")
    
    transcriber = Transcriber(model_size="small", use_mlx=True)
    transcriber.load_model()
    
    sample_rate = 16000
    audio = np.random.randn(sample_rate * 3).astype(np.float32) * 0.1
    
    import time
    start = time.time()
    text, confidence = transcriber.transcribe(audio)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Result: '{text}'")
