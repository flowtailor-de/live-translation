"""
Gemini Live Translate pipeline.

A drop-in alternative to ParallelTranslationPipeline that uses Google's
Gemini Live Translate API (gemini-3.5-live-translate-preview) instead of the
local STT -> Translation -> TTS chain.

It honors the same contract as the local pipeline so the server and frontend
need no changes:
  - start() / stop()
  - get_stats()
  - on_result callback emitting a `TranslationResult`

Flow:
  mic audio (16kHz PCM) --> Gemini Live session --> translated audio (24kHz PCM)
  + input/output transcriptions, aggregated per conversational turn into one
  `TranslationResult` and delivered via on_result -> broadcast_result.

Because Gemini connections drop every ~10-15 min, start() wraps the session in
a reconnect loop so a multi-hour live event keeps running.
"""

import asyncio
import contextlib
import logging
import time
from typing import Callable, Optional

import numpy as np

from src.pipeline import TranslationResult

logger = logging.getLogger(__name__)

GEMINI_OUTPUT_SAMPLE_RATE = 24000  # Gemini Live Translate returns 24kHz PCM
GEMINI_INPUT_SAMPLE_RATE = 16000   # We send 16kHz PCM


class GeminiTranslationPipeline:
    """Live translation backed by the Gemini Live Translate API."""

    def __init__(
        self,
        audio_capture,
        api_key: str,
        target_lang: str,
        source_lang: str = "auto",
        model: str = "gemini-3.5-live-translate-preview",
        echo_target_language: bool = True,
        on_result: Optional[Callable[[TranslationResult], None]] = None,
        speech_gate=None,
    ):
        self.audio_capture = audio_capture
        self.api_key = api_key
        self.target_lang = target_lang
        # Source is auto-detected by Gemini; kept so get_stats()/the UI badge work.
        self.source_lang = source_lang
        self.model = model
        self.echo_target_language = echo_target_language
        self.on_result = on_result
        # Optional SpeechGate: suppresses silence so we don't pay to stream it.
        self.speech_gate = speech_gate

        self.is_running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client = None
        self._config = None

        # Per-turn accumulators
        self._turn_counter = 0
        self._orig = ""
        self._trans = ""
        self._audio_buf: list[bytes] = []
        self._turn_start_t: Optional[float] = None

        self.stats = {
            "processed_chunks": 0,
            "total_audio_seconds": 0,
            "avg_latency_ms": 0,
            "queued_chunks": 0,
            "active_workers": 0,
        }

    # ----- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Start capture and run the reconnecting Gemini session loop."""
        if self.is_running:
            return

        # Imported lazily so the local mode never needs google-genai installed.
        from google import genai
        from google.genai import types

        logger.info(
            f"Starting Gemini Live Translate (model={self.model}, "
            f"target={self.target_lang}, echo={self.echo_target_language})"
        )
        self.is_running = True
        self._loop = asyncio.get_event_loop()

        self._client = genai.Client(api_key=self.api_key)
        self._config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=self.target_lang,
                echo_target_language=self.echo_target_language,
            ),
        )

        self.audio_capture.start()

        # Reconnect loop: survives the ~10-15 min Gemini connection cap.
        while self.is_running:
            try:
                await self._run_session(types)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gemini session error: {e}", exc_info=True)

            if self.is_running:
                logger.info("Gemini connection ended; reconnecting in 1s...")
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Stop the pipeline and audio capture."""
        logger.info("Stopping Gemini pipeline...")
        self.is_running = False
        self.audio_capture.stop()

    # ----- session ---------------------------------------------------------

    async def _run_session(self, types) -> None:
        """Open one live session and run the send + receive loops."""
        async with self._client.aio.live.connect(
            model=self.model, config=self._config
        ) as session:
            logger.info("Connected to Gemini Live Translate session")
            send_task = asyncio.create_task(self._send_loop(session, types))
            try:
                await self._receive_loop(session)
            finally:
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task

    async def _send_loop(self, session, types) -> None:
        """Read mic chunks and stream them to Gemini as 16kHz PCM.

        When a speech gate is configured, only audio during speech (plus
        pre-roll/hangover) is sent — silence is dropped so we don't pay to
        stream it.
        """
        loop = asyncio.get_event_loop()
        while self.is_running:
            audio = await loop.run_in_executor(
                None, lambda: self.audio_capture.get_audio(timeout=0.1)
            )
            if audio is None:
                continue

            if self.speech_gate is not None:
                audio = self.speech_gate.process(audio)
                if audio is None:
                    continue  # silence suppressed

            pcm = self._float_to_pcm16(audio)
            await session.send_realtime_input(
                audio=types.Blob(
                    data=pcm,
                    mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
                )
            )

    async def _receive_loop(self, session) -> None:
        """Consume server events, aggregate per turn, emit TranslationResults."""
        self._reset_turn()
        async for response in session.receive():
            # GoAway = server is about to close this connection -> reconnect.
            if getattr(response, "go_away", None) is not None:
                logger.info("Gemini sent GoAway; reconnecting")
                return

            content = response.server_content
            if not content:
                continue

            now = time.time()

            # Process ALL parts of every event (events can carry several at once).
            in_tx = getattr(content, "input_transcription", None)
            if in_tx and in_tx.text:
                if self._turn_start_t is None:
                    self._turn_start_t = now
                self._orig += in_tx.text

            out_tx = getattr(content, "output_transcription", None)
            if out_tx and out_tx.text:
                if self._turn_start_t is None:
                    self._turn_start_t = now
                self._trans += out_tx.text

            model_turn = getattr(content, "model_turn", None)
            if model_turn and model_turn.parts:
                for part in model_turn.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        if self._turn_start_t is None:
                            self._turn_start_t = now
                        self._audio_buf.append(inline.data)

            if getattr(content, "turn_complete", False):
                await self._emit_turn()
                self._reset_turn()

    # ----- delivery --------------------------------------------------------

    async def _emit_turn(self) -> None:
        """Package the accumulated turn into a TranslationResult and deliver it."""
        orig = self._orig.strip()
        trans = self._trans.strip()
        audio_bytes = b"".join(self._audio_buf)

        if not audio_bytes and not trans:
            return  # nothing meaningful in this turn

        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16)
            if audio_bytes
            else np.array([], dtype=np.int16)
        )

        self._turn_counter += 1
        latency_ms = (
            (time.time() - self._turn_start_t) * 1000.0 if self._turn_start_t else 0.0
        )

        result = TranslationResult(
            chunk_id=self._turn_counter,
            original_text=orig,
            translated_text=trans,
            audio_data=audio_np,
            sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

        self.stats["processed_chunks"] += 1
        if len(audio_np) > 0:
            self.stats["total_audio_seconds"] += len(audio_np) / GEMINI_OUTPUT_SAMPLE_RATE
        self._update_avg_latency(latency_ms)

        logger.info(
            f">>> Turn {self._turn_counter}: '{orig[:30]}' -> '{trans[:30]}' "
            f"({len(audio_np) / GEMINI_OUTPUT_SAMPLE_RATE:.1f}s audio, "
            f"{latency_ms:.0f}ms)"
        )

        if self.on_result:
            res = self.on_result(result)
            if asyncio.iscoroutine(res):
                await res

    def _reset_turn(self) -> None:
        self._orig = ""
        self._trans = ""
        self._audio_buf = []
        self._turn_start_t = None

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _float_to_pcm16(audio: np.ndarray) -> bytes:
        """Convert float32 [-1, 1] mono audio to little-endian 16-bit PCM bytes."""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes()

    def _update_avg_latency(self, latency_ms: float) -> None:
        n = self.stats["processed_chunks"]
        if n > 0:
            self.stats["avg_latency_ms"] += (
                latency_ms - self.stats["avg_latency_ms"]
            ) / n

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "is_running": self.is_running,
            "delivery_buffer": 0,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "backend": "gemini-live",
        }
