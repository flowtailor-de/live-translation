"""
Pipeline module for coordinating the translation flow.
Parallel processing with ORDERED delivery and STRICT worker limits.
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of a translation pipeline run."""
    chunk_id: int
    original_text: str
    translated_text: str
    audio_data: np.ndarray
    sample_rate: int
    latency_ms: float
    timestamp: float
    
    def __lt__(self, other):
        return self.chunk_id < other.chunk_id


@dataclass
class SpeechChunk:
    """A speech segment ready for processing."""
    chunk_id: int
    audio: np.ndarray
    created_at: float


class OrderedDeliveryQueue:
    """Ensures results are delivered in chunk_id order."""
    
    def __init__(self):
        self.pending: Dict[int, TranslationResult] = {}
        self.next_to_deliver = 1
        self.lock = threading.Lock()
    
    def add_result(self, result: TranslationResult) -> list[TranslationResult]:
        """Add result and return any that are ready for delivery."""
        ready = []
        with self.lock:
            self.pending[result.chunk_id] = result
            while self.next_to_deliver in self.pending:
                ready.append(self.pending.pop(self.next_to_deliver))
                self.next_to_deliver += 1
        return ready
    
    def skip_chunk(self, chunk_id: int) -> list[TranslationResult]:
        """Skip a chunk (empty result) and return any ready results."""
        ready = []
        with self.lock:
            if chunk_id == self.next_to_deliver:
                self.next_to_deliver += 1
                # Check if any waiting chunks can now be delivered
                while self.next_to_deliver in self.pending:
                    ready.append(self.pending.pop(self.next_to_deliver))
                    self.next_to_deliver += 1
        return ready
    
    def get_status(self) -> dict:
        with self.lock:
            return {
                "next_to_deliver": self.next_to_deliver,
                "buffered_count": len(self.pending),
                "buffered_ids": sorted(self.pending.keys()) if self.pending else [],
            }


class ParallelTranslationPipeline:
    """
    Parallel pipeline with STRICT worker limits and ordered delivery.
    
    Uses a Semaphore to strictly limit concurrent workers.
    """
    
    def __init__(
        self,
        audio_capture,
        vad,
        transcriber,
        translator,
        synthesizer,
        on_result: Optional[Callable[[TranslationResult], None]] = None,
        max_workers: int = 4,
    ):
        self.audio_capture = audio_capture
        self.vad = vad
        self.transcriber = transcriber
        self.translator = translator
        self.synthesizer = synthesizer
        self.on_result = on_result
        self.max_workers = max_workers
        
        self.is_running = False
        self.chunk_counter = 0
        
        # Thread pool - STRICTLY limited
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Semaphore to STRICTLY limit concurrent processing
        self._worker_semaphore: Optional[asyncio.Semaphore] = None
        
        # Queue for speech segments
        self.speech_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        
        # Ordered delivery
        self.delivery_queue = OrderedDeliveryQueue()
        self.results_queue: asyncio.Queue = asyncio.Queue()
        
        # Active worker tracking
        self.active_workers = 0
        self._active_lock = threading.Lock()
        
        # Event loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Stats
        self.stats = {
            "processed_chunks": 0,
            "total_audio_seconds": 0,
            "avg_latency_ms": 0,
            "queued_chunks": 0,
            "active_workers": 0,
        }
    
    async def start(self) -> None:
        """Start the pipeline."""
        if self.is_running:
            return
        
        logger.info(f"Starting pipeline (max {self.max_workers} workers, ordered delivery)...")
        self.is_running = True
        self._loop = asyncio.get_event_loop()
        
        # Create thread pool with STRICT limit
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Semaphore enforces the limit
        self._worker_semaphore = asyncio.Semaphore(self.max_workers)
        
        # Start audio capture
        self.audio_capture.start()
        
        # Run tasks
        await asyncio.gather(
            self._audio_capture_loop(),
            self._processing_dispatcher(),
        )
    
    async def stop(self) -> None:
        """Stop the pipeline."""
        logger.info("Stopping pipeline...")
        self.is_running = False
        self.audio_capture.stop()
        self.vad.reset()
        
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
    
    async def _audio_capture_loop(self) -> None:
        """Capture audio and detect speech."""
        loop = asyncio.get_event_loop()
        
        while self.is_running:
            try:
                audio = await loop.run_in_executor(
                    None,
                    lambda: self.audio_capture.get_audio(timeout=0.1)
                )
                
                if audio is None:
                    continue
                
                speech_segment = self.vad.process_chunk(audio)
                
                if speech_segment is not None:
                    self.chunk_counter += 1
                    chunk = SpeechChunk(
                        chunk_id=self.chunk_counter,
                        audio=speech_segment,
                        created_at=time.time(),
                    )
                    
                    try:
                        self.speech_queue.put_nowait(chunk)
                        logger.info(f"Chunk {chunk.chunk_id}: Queued ({len(speech_segment)/16000:.1f}s)")
                    except asyncio.QueueFull:
                        logger.warning("Queue full, dropping chunk")
                    
                    self.stats["queued_chunks"] = self.speech_queue.qsize()
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                await asyncio.sleep(0.1)
    
    async def _processing_dispatcher(self) -> None:
        """Dispatch chunks with STRICT worker limit using semaphore."""
        loop = asyncio.get_event_loop()
        
        while self.is_running:
            try:
                # Get next chunk
                chunk = await asyncio.wait_for(
                    self.speech_queue.get(),
                    timeout=0.5
                )
                
                # WAIT for a worker slot (this is the key change!)
                await self._worker_semaphore.acquire()
                
                with self._active_lock:
                    self.active_workers += 1
                    self.stats["active_workers"] = self.active_workers
                
                logger.info(f"Chunk {chunk.chunk_id}: Processing (workers: {self.active_workers}/{self.max_workers})")
                
                # Submit to thread pool
                future = loop.run_in_executor(
                    self.executor,
                    self._process_speech_sync,
                    chunk,
                )
                
                # Handle completion
                future.add_done_callback(
                    lambda f, cid=chunk.chunk_id: self._on_complete(f, cid)
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
                await asyncio.sleep(0.1)
    
    def _on_complete(self, future, chunk_id: int) -> None:
        """Handle completed processing."""
        # Release semaphore
        if self._loop and self._worker_semaphore:
            self._loop.call_soon_threadsafe(self._worker_semaphore.release)
        
        with self._active_lock:
            self.active_workers -= 1
            self.stats["active_workers"] = self.active_workers
        
        try:
            result = future.result()
            if result:
                ready = self.delivery_queue.add_result(result)
                for r in ready:
                    self._deliver(r)
                
                status = self.delivery_queue.get_status()
                if status["buffered_count"] > 0:
                    logger.info(f"Chunk {chunk_id}: Done (waiting for {status['buffered_ids']})")
                else:
                    logger.info(f"Chunk {chunk_id}: Done & delivered")
            else:
                # Empty result - skip this chunk in ordering
                ready = self.delivery_queue.skip_chunk(chunk_id)
                for r in ready:
                    self._deliver(r)
                logger.debug(f"Chunk {chunk_id}: Empty, skipped")
                
        except Exception as e:
            logger.error(f"Chunk {chunk_id}: Error: {e}")
            # Skip the failed chunk
            ready = self.delivery_queue.skip_chunk(chunk_id)
            for r in ready:
                self._deliver(r)
    
    def _deliver(self, result: TranslationResult) -> None:
        """Deliver result to callback."""
        self.stats["processed_chunks"] += 1
        if len(result.audio_data) > 0:
            self.stats["total_audio_seconds"] += len(result.audio_data) / result.sample_rate
        self._update_avg_latency(result.latency_ms)
        
        logger.info(f">>> Delivering Chunk {result.chunk_id} (latency: {result.latency_ms:.0f}ms)")
        
        if self.on_result and self._loop:
            self._loop.call_soon_threadsafe(
                lambda r=result: asyncio.create_task(self._async_deliver(r))
            )
        
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda r=result: self.results_queue.put_nowait(r)
            )
    
    async def _async_deliver(self, result: TranslationResult) -> None:
        if self.on_result:
            if asyncio.iscoroutinefunction(self.on_result):
                await self.on_result(result)
            else:
                self.on_result(result)
    
    def _process_speech_sync(self, chunk: SpeechChunk) -> Optional[TranslationResult]:
        """Process: STT -> Translation -> TTS."""
        try:
            start = time.time()
            
            # 1. STT
            stt_start = time.time()
            text, _ = self.transcriber.transcribe(chunk.audio)
            stt_ms = (time.time() - stt_start) * 1000
            
            if not text.strip():
                return None
            
            logger.info(f"Chunk {chunk.chunk_id}: STT ({stt_ms:.0f}ms): '{text[:35]}...'")
            
            # 2. Translation
            trans_start = time.time()
            translated = self.translator.translate(text)
            trans_ms = (time.time() - trans_start) * 1000
            
            if not translated.strip():
                return None
            
            logger.info(f"Chunk {chunk.chunk_id}: Trans ({trans_ms:.0f}ms)")
            
            # 3. TTS
            tts_start = time.time()
            audio, sr = self.synthesizer.synthesize(translated)
            tts_ms = (time.time() - tts_start) * 1000
            
            total_ms = (time.time() - start) * 1000
            latency_ms = (time.time() - chunk.created_at) * 1000
            
            logger.info(f"Chunk {chunk.chunk_id}: TTS ({tts_ms:.0f}ms). Total: {total_ms:.0f}ms")
            
            return TranslationResult(
                chunk_id=chunk.chunk_id,
                original_text=text,
                translated_text=translated,
                audio_data=audio,
                sample_rate=sr,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )
            
        except Exception as e:
            logger.error(f"Chunk {chunk.chunk_id}: Error: {e}")
            return None
    
    def _update_avg_latency(self, latency_ms: float) -> None:
        n = self.stats["processed_chunks"]
        if n > 0:
            self.stats["avg_latency_ms"] += (latency_ms - self.stats["avg_latency_ms"]) / n
    
    async def get_result(self, timeout: float = 1.0) -> Optional[TranslationResult]:
        try:
            return await asyncio.wait_for(self.results_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def get_stats(self) -> dict:
        return {
            **self.stats,
            "is_running": self.is_running,
            "delivery_buffer": self.delivery_queue.get_status()["buffered_count"],
        }


# Alias
TranslationPipeline = ParallelTranslationPipeline


class AudioBuffer:
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 22050):
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = asyncio.Lock()
    
    async def add_audio(self, audio: np.ndarray) -> None:
        async with self.lock:
            self.buffer.extend(audio)
    
    async def get_audio(self, num_samples: int) -> np.ndarray:
        async with self.lock:
            to_read = min(num_samples, len(self.buffer))
            return np.array([self.buffer.popleft() for _ in range(to_read)], dtype=np.float32)
    
    def available(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        self.buffer.clear()
