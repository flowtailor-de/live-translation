"""
File-based audio capture module.
Simulates real-time audio input from a file.
"""

import numpy as np
import time
import threading
import queue
import logging
import wave
import scipy.signal
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileAudioCapture:
    """Captures audio from a file, simulating a real-time stream."""
    
    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.1,
        loop: bool = False
    ):
        """
        Initialize file audio capture.
        
        Args:
            file_path: Path to the audio file (WAV preferred)
            sample_rate: Target sample rate in Hz
            channels: Target number of channels
            chunk_duration: Duration of each audio chunk in seconds
            loop: Whether to loop the audio file
        """
        self.file_path = Path(file_path)
        self.target_sample_rate = sample_rate
        self.target_channels = channels
        self.chunk_duration = chunk_duration
        self.loop = loop
        
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        self._load_audio()
        
    def _load_audio(self):
        """Load audio file using ffmpeg to ensure support for all formats."""
        import subprocess
        import tempfile
        import os
        import scipy.io.wavfile as wavfile
        
        logger.info(f"Loading audio file: {self.file_path}")
        
        # Create a temp file for the converted wav
        fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        try:
            # Use ffmpeg to convert input to wav with target sample rate and channels
            # -y: overwrite output
            # -i: input file
            # -ac: audio channels
            # -ar: audio sample rate
            # -vn: disable video
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.file_path),
                '-ac', str(self.target_channels),
                '-ar', str(self.target_sample_rate),
                '-vn',
                temp_path
            ]
            
            logger.info(f"Converting with ffmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed with code {result.returncode}")
                
            # Read the converted file
            sr, data = wavfile.read(temp_path)
            
            # Convert to float32 [-1, 1]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0
            elif data.dtype == np.float32:
                pass
            else:
                logger.warning(f"Unexpected dtype: {data.dtype}, normalizing max")
                data = data.astype(np.float32)
                data = data / np.max(np.abs(data))
                
            self.audio_data = data.astype(np.float32)
            logger.info(f"Audio loaded: {len(self.audio_data) / self.target_sample_rate:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            # Create dummy audio (silence)
            self.audio_data = np.zeros(self.target_sample_rate * 5, dtype=np.float32)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
    def _capture_loop(self):
        """Thread to simulate real-time capture."""
        cursor = 0
        total_samples = len(self.audio_data)
        
        logger.info("Starting file playback simulation")
        
        next_chunk_time = time.time()
        
        while self.is_running:
            # Determine chunk size
            remaining = total_samples - cursor
            if remaining <= 0:
                if self.loop:
                    cursor = 0
                    remaining = total_samples
                    logger.info("Looping audio")
                else:
                    # End of file
                    logger.info("End of file reached")
                    # Send silence to keep pipeline alive or break?
                    # Let's send silence to keep it running until manually stopped
                    chunk = np.zeros(self.chunk_size, dtype=np.float32)
                    self.audio_queue.put(chunk)
                    time.sleep(self.chunk_duration)
                    continue
            
            take = min(self.chunk_size, remaining)
            chunk = self.audio_data[cursor : cursor + take]
            
            # Pad if needed (should only happen at very end if not looping)
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            
            self.audio_queue.put(chunk)
            cursor += take
            
            # Precise timing
            next_chunk_time += self.chunk_duration
            sleep_time = next_chunk_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def start(self):
        """Start capturing audio."""
        if self.is_running:
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def stop(self):
        """Stop capturing audio."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
            self.capture_thread = None
            
    def get_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get audio chunk."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
