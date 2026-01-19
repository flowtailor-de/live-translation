"""
Audio capture module for USB audio interface.
Handles real-time audio input from Behringer X32 or similar devices.
"""

import numpy as np
import sounddevice as sd
from typing import Callable, Optional
import threading
import queue
import logging

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from USB audio interface in real-time."""
    
    def __init__(
        self,
        device_name: str = "X32",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.1,
    ):
        """
        Initialize audio capture.
        
        Args:
            device_name: Partial name of the audio device to use
            sample_rate: Sample rate in Hz (16000 recommended for Whisper)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        self.device_id: Optional[int] = None
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        
        self._find_device()
    
    def _find_device(self) -> None:
        """Find the audio device by name."""
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            if self.device_name.lower() in device['name'].lower():
                if device['max_input_channels'] > 0:
                    self.device_id = idx
                    logger.info(f"Found audio device: {device['name']} (id={idx})")
                    return
        
        # List available devices for debugging
        logger.warning(f"Device '{self.device_name}' not found. Available input devices:")
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                logger.warning(f"  [{idx}] {device['name']}")
        
        # Fall back to default input device
        self.device_id = None
        logger.warning("Using default input device")
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to float32 and put in queue
        audio_data = indata.copy().flatten().astype(np.float32)
        self.audio_queue.put(audio_data)
    
    def start(self) -> None:
        """Start capturing audio."""
        if self.is_running:
            return
        
        logger.info(f"Starting audio capture (device={self.device_id}, rate={self.sample_rate}Hz)")
        
        self.stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()
        self.is_running = True
    
    def stop(self) -> None:
        """Stop capturing audio."""
        if not self.is_running:
            return
        
        logger.info("Stopping audio capture")
        self.is_running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio data as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_audio(self) -> np.ndarray:
        """Get all currently queued audio data."""
        chunks = []
        while not self.audio_queue.empty():
            try:
                chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)
    
    @staticmethod
    def list_devices() -> list[dict]:
        """List all available audio devices."""
        devices = []
        for idx, device in enumerate(sd.query_devices()):
            devices.append({
                'id': idx,
                'name': device['name'],
                'inputs': device['max_input_channels'],
                'outputs': device['max_output_channels'],
                'sample_rate': device['default_samplerate'],
            })
        return devices


if __name__ == "__main__":
    # Test audio capture
    logging.basicConfig(level=logging.INFO)
    
    print("Available audio devices:")
    for device in AudioCapture.list_devices():
        if device['inputs'] > 0:
            print(f"  [{device['id']}] {device['name']} (in={device['inputs']}, out={device['outputs']})")
    
    print("\nStarting audio capture test (5 seconds)...")
    capture = AudioCapture()
    capture.start()
    
    import time
    start_time = time.time()
    total_samples = 0
    
    while time.time() - start_time < 5:
        audio = capture.get_audio(timeout=0.5)
        if audio is not None:
            total_samples += len(audio)
            # Calculate audio level
            level = np.abs(audio).mean()
            bars = int(level * 100)
            print(f"\rLevel: {'█' * bars}{' ' * (50 - bars)} {level:.4f}", end="")
    
    capture.stop()
    print(f"\n\nCaptured {total_samples} samples ({total_samples / 16000:.2f} seconds)")
