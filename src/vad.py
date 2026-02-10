"""
Voice Activity Detection (VAD) module.
Detects speech segments in audio stream using Silero VAD.
Forces maximum chunk duration for real-time translation.
"""

import numpy as np
import torch
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Detects speech in audio using Silero VAD model.
    Uses two-tier silence detection for sentence-level chunking:
    - Short silences (word pauses) mark potential chunk boundaries
    - Longer silences (sentence pauses) trigger actual chunk emission
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.5,  # Legacy, maps to word_silence
        max_speech_duration: float = 15.0,  # Higher default for sentence-based chunking
        speech_pad_duration: float = 0.2,
        # New two-tier silence detection
        word_silence_duration: float = None,  # Short pauses within sentences
        sentence_silence_duration: float = 1.0,  # Longer pauses = sentence boundaries
    ):
        """
        Initialize VAD with sentence-boundary detection.
        
        Args:
            threshold: Speech probability threshold (0-1)
            sample_rate: Audio sample rate (must be 16000 for Silero)
            min_speech_duration: Minimum speech duration in seconds
            min_silence_duration: Legacy param, used as word_silence if not specified
            max_speech_duration: MAXIMUM duration before forcing a chunk
            speech_pad_duration: Padding around detected speech
            word_silence_duration: Short pauses that mark potential boundaries
            sentence_silence_duration: Longer pauses that trigger chunk emission
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self.speech_pad_samples = int(speech_pad_duration * sample_rate)
        
        # Two-tier silence detection
        word_silence = word_silence_duration if word_silence_duration is not None else min_silence_duration
        self.word_silence_samples = int(word_silence * sample_rate)
        self.sentence_silence_samples = int(sentence_silence_duration * sample_rate)
        # Legacy compat
        self.min_silence_samples = self.word_silence_samples
        
        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        self.model.eval()
        
        # State for streaming
        self.audio_buffer: List[np.ndarray] = []
        self.raw_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.is_speech = False
        self.speech_start_idx = 0
        self.silence_samples = 0
        self.speech_samples = 0  # Track how long current speech is
        self.total_samples = 0
        
        # Sentence boundary tracking
        self.potential_boundary_idx: Optional[int] = None  # Index of last potential boundary
        self.potential_boundary_samples: int = 0  # Speech samples at that boundary
        
        logger.info(f"VAD initialized (sentence_silence={sentence_silence_duration}s, max_speech={max_speech_duration}s)")
    
    def reset(self) -> None:
        """Reset VAD state for new segment."""
        self.audio_buffer = []
        self.is_speech = False
        self.speech_start_idx = 0
        self.silence_samples = 0
        self.speech_samples = 0
        self.total_samples = 0
        self.potential_boundary_idx = None
        self.potential_boundary_samples = 0
        self.model.reset_states()
    
    def process_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Process an audio chunk and return complete speech segment if available.
        Forces output at max_speech_duration even if speech continues.
        """
        # Add to raw buffer
        self.raw_buffer = np.concatenate([self.raw_buffer, audio])
        
        vad_window = 512  # Silero requirement for 16k
        results = []
        
        # Process all complete windows
        while len(self.raw_buffer) >= vad_window:
            window = self.raw_buffer[:vad_window]
            self.raw_buffer = self.raw_buffer[vad_window:]
            
            res = self._process_window(window)
            if res is not None:
                results.append(res)
        
        if results:
            return np.concatenate(results)
        
        return None
    
    def _process_window(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single 512-sample window with two-tier silence detection.
        
        Logic:
        1. Speech is being recorded
        2. Short pause (word_silence) → mark potential boundary, continue
        3. Long pause (sentence_silence) → emit chunk (sentence complete)
        4. Max duration reached → emit at last potential boundary
        """
        audio_tensor = torch.from_numpy(audio).float()
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        # Add to buffer
        self.audio_buffer.append(audio)
        self.total_samples += len(audio)
        
        if speech_prob >= self.threshold:
            # Speech detected
            if not self.is_speech:
                self.is_speech = True
                self.speech_start_idx = max(0, len(self.audio_buffer) - 5)
                self.speech_samples = 0
                self.potential_boundary_idx = None
                self.potential_boundary_samples = 0
                logger.debug("Speech started")
            
            self.silence_samples = 0
            self.speech_samples += len(audio)
            
            # CHECK: Force chunk if max duration reached
            if self.speech_samples >= self.max_speech_samples:
                # Try to emit at last potential boundary if available
                if self.potential_boundary_idx is not None:
                    speech_audio = self._extract_speech_to_boundary()
                    dur = len(speech_audio) / self.sample_rate
                    logger.debug(f"Max duration: emitting at boundary ({dur:.2f}s)")
                    self._soft_reset_from_boundary()
                    return speech_audio
                else:
                    # No boundary found, force chunk anyway
                    speech_audio = self._extract_speech()
                    logger.debug(f"Max duration: forced chunk ({len(speech_audio) / self.sample_rate:.2f}s)")
                    self._soft_reset()
                    return speech_audio
                
        else:
            # Silence detected
            if self.is_speech:
                self.silence_samples += len(audio)
                
                # Tier 1: Word pause - mark potential boundary
                if self.silence_samples >= self.word_silence_samples:
                    if self.potential_boundary_idx is None or \
                       len(self.audio_buffer) > self.potential_boundary_idx + 5:
                        # Update boundary to current position
                        self.potential_boundary_idx = len(self.audio_buffer)
                        self.potential_boundary_samples = self.speech_samples
                        logger.debug(f"Potential boundary at {self.speech_samples / self.sample_rate:.2f}s")
                
                # Tier 2: Sentence pause - emit chunk
                if self.silence_samples >= self.sentence_silence_samples:
                    speech_audio = self._extract_speech()
                    
                    if len(speech_audio) >= self.min_speech_samples:
                        dur = len(speech_audio) / self.sample_rate
                        logger.debug(f"Sentence complete (pause): {dur:.2f}s")
                        self.reset()
                        return speech_audio
                    else:
                        logger.debug("Speech too short, discarding")
                        self.reset()
        
        # Prevent buffer growing too large
        max_buffer_chunks = 600  # ~19 seconds (higher for longer sentences)
        if len(self.audio_buffer) > max_buffer_chunks:
            if self.is_speech:
                speech_audio = self._extract_speech()
                logger.debug(f"Buffer full: forced chunk ({len(speech_audio) / self.sample_rate:.2f}s)")
                self._soft_reset()
                return speech_audio
            else:
                self.audio_buffer = self.audio_buffer[-10:]
                self.speech_start_idx = 0
        
        return None
    
    def _extract_speech(self) -> np.ndarray:
        """Extract speech segment from buffer."""
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        
        speech_chunks = self.audio_buffer[self.speech_start_idx:]
        return np.concatenate(speech_chunks)
    
    def _soft_reset(self) -> None:
        """Soft reset - keep some context for continuity."""
        # Keep last bit of audio for context
        keep_chunks = 5
        if len(self.audio_buffer) > keep_chunks:
            self.audio_buffer = self.audio_buffer[-keep_chunks:]
        else:
            self.audio_buffer = []
        
        self.speech_start_idx = 0
        self.silence_samples = 0
        self.speech_samples = 0
        # Keep is_speech = True so we continue immediately
        self.potential_boundary_idx = None
        self.potential_boundary_samples = 0
        self.model.reset_states()
    
    def _extract_speech_to_boundary(self) -> np.ndarray:
        """Extract speech from start to the potential boundary point."""
        if not self.audio_buffer or self.potential_boundary_idx is None:
            return self._extract_speech()
        
        end_idx = min(self.potential_boundary_idx, len(self.audio_buffer))
        speech_chunks = self.audio_buffer[self.speech_start_idx:end_idx]
        if not speech_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(speech_chunks)
    
    def _soft_reset_from_boundary(self) -> None:
        """Reset but keep audio from boundary point onwards for next chunk."""
        if self.potential_boundary_idx is None:
            self._soft_reset()
            return
        
        # Keep audio from boundary onwards
        boundary_idx = min(self.potential_boundary_idx, len(self.audio_buffer))
        if boundary_idx < len(self.audio_buffer):
            self.audio_buffer = self.audio_buffer[boundary_idx:]
        else:
            self.audio_buffer = []
        
        self.speech_start_idx = 0
        self.silence_samples = 0
        # Calculate remaining speech samples
        self.speech_samples = sum(len(chunk) for chunk in self.audio_buffer)
        self.potential_boundary_idx = None
        self.potential_boundary_samples = 0
        # Keep is_speech = True so we continue immediately
        self.model.reset_states()
    
    def get_pending_audio(self) -> Optional[np.ndarray]:
        """Get any pending speech audio (for final flush)."""
        if self.is_speech and self.audio_buffer:
            return self._extract_speech()
        return None


class SimpleVAD:
    """Simple energy-based VAD with max duration."""
    
    def __init__(
        self,
        energy_threshold: float = 0.01,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.5,
        max_speech_duration: float = 5.0,
    ):
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.min_silence_samples = int(min_silence_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        
        self.audio_buffer: List[np.ndarray] = []
        self.is_speech = False
        self.silence_samples = 0
        self.speech_samples = 0
    
    def reset(self) -> None:
        self.audio_buffer = []
        self.is_speech = False
        self.silence_samples = 0
        self.speech_samples = 0
    
    def process_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        energy = np.sqrt(np.mean(audio ** 2))
        self.audio_buffer.append(audio)
        
        if energy >= self.energy_threshold:
            if not self.is_speech:
                self.is_speech = True
                self.speech_samples = 0
            
            self.silence_samples = 0
            self.speech_samples += len(audio)
            
            # Force chunk at max duration
            if self.speech_samples >= self.max_speech_samples:
                speech_audio = np.concatenate(self.audio_buffer)
                self.audio_buffer = self.audio_buffer[-3:]  # Keep context
                self.speech_samples = 0
                return speech_audio
                
        elif self.is_speech:
            self.silence_samples += len(audio)
            
            if self.silence_samples >= self.min_silence_samples:
                speech_audio = np.concatenate(self.audio_buffer)
                self.reset()
                
                if len(speech_audio) >= self.min_speech_samples:
                    return speech_audio
        
        # Limit buffer
        if len(self.audio_buffer) > 100:
            if self.is_speech:
                speech_audio = np.concatenate(self.audio_buffer)
                self.audio_buffer = self.audio_buffer[-3:]
                self.speech_samples = 0
                return speech_audio
            else:
                self.audio_buffer = self.audio_buffer[-5:]
        
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing VAD...")
    vad = VoiceActivityDetector(max_speech_duration=3.0)
    
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    speech = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1  # 10 seconds
    
    test_audio = np.concatenate([silence, speech, silence])
    
    chunk_size = int(sample_rate * 0.1)
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        result = vad.process_chunk(chunk)
        if result is not None:
            print(f"Got speech segment: {len(result) / sample_rate:.2f} seconds")
