"""
Main entry point for the Live Translation System.
"""

import asyncio
import os
import yaml
import logging
import argparse
import sys
from pathlib import Path
import warnings

# Suppress benign warning from multiprocessing.resource_tracker on shutdown
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Load environment variables from a local .env file (e.g. GEMINI_API_KEY).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_capture import AudioCapture
from src.vad import VoiceActivityDetector, SimpleVAD
from src.transcriber import Transcriber
from src.translator import create_translator
from src.synthesizer import Synthesizer, MockSynthesizer
from src.server import create_app, broadcast_result

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    
    logger.warning(f"Config file not found: {config_path}, using defaults")
    return {}


def create_audio_capture(config: dict):
    """Create the audio capture source from config (mic or file)."""
    audio_config = config.get("audio", {})
    if audio_config.get('input_file'):
        logger.info(f"Using file input: {audio_config['input_file']}")
        from src.file_audio_capture import FileAudioCapture
        return FileAudioCapture(
            file_path=audio_config['input_file'],
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            chunk_duration=audio_config.get("chunk_duration", 0.1),
            loop=True
        )
    logger.info("Using microphone input")
    return AudioCapture(
        device_name=audio_config.get("device", "X32"),
        sample_rate=audio_config.get("sample_rate", 16000),
        channels=audio_config.get("channels", 1),
        chunk_duration=audio_config.get("chunk_duration", 0.1),
        input_channel=audio_config.get("input_channel"),
    )


def create_components(config: dict):
    """Create all pipeline components from config."""

    # Audio capture
    audio_capture = create_audio_capture(config)

    # Voice Activity Detection with sentence-level chunking
    vad_config = config.get("vad", {})
    try:
        vad = VoiceActivityDetector(
            threshold=vad_config.get("threshold", 0.5),
            min_speech_duration=vad_config.get("min_speech_duration", 0.3),
            min_silence_duration=vad_config.get("min_silence_duration", 0.4),  # Legacy
            max_speech_duration=vad_config.get("max_speech_duration", 15.0),
            speech_pad_duration=vad_config.get("speech_pad_duration", 0.2),
            # Two-tier sentence detection
            word_silence_duration=vad_config.get("word_silence_duration"),  # None = use min_silence
            sentence_silence_duration=vad_config.get("sentence_silence_duration", 1.0),
        )
    except Exception as e:
        logger.warning(f"Could not load Silero VAD: {e}, using simple VAD")
        vad = SimpleVAD(
            energy_threshold=0.01,
            min_speech_duration=vad_config.get("min_speech_duration", 0.3),
            min_silence_duration=vad_config.get("min_silence_duration", 0.4),
            max_speech_duration=vad_config.get("max_speech_duration", 15.0),
        )
    
    # Transcriber (Whisper)
    stt_config = config.get("stt", {})
    transcriber = Transcriber(
        model_size=stt_config.get("model", "medium"),
        device=stt_config.get("device", "auto"),
        compute_type=stt_config.get("compute_type", "float16"),
        language=stt_config.get("language", "de"),
    )
    
    # Translator (NLLB or TranslateGemma)
    trans_config = config.get("translation", {})
    backend = trans_config.get("backend", "nllb")
    
    if backend == "translategemma-mlx":
        tg_config = trans_config.get("translategemma-mlx", {})
        translator = create_translator(
            backend="translategemma-mlx",
            model_name=tg_config.get("model", "mlx-community/translategemma-4b-it-4bit"),
            source_lang=trans_config.get("source_lang", "de"),
            target_lang=trans_config.get("target_lang", "fa"),
            max_new_tokens=tg_config.get("max_new_tokens", 256),
        )
        logger.info(f"Using TranslateGemma MLX backend: {tg_config.get('model')}")
    elif backend == "translategemma":
        tg_config = trans_config.get("translategemma", {})
        translator = create_translator(
            backend="translategemma",
            model_name=tg_config.get("model", "google/translategemma-12b-it"),
            source_lang=trans_config.get("source_lang", "de"),
            target_lang=trans_config.get("target_lang", "fa"),
            device=trans_config.get("device", "mps"),
            max_new_tokens=tg_config.get("max_new_tokens", 256),
        )
        logger.info(f"Using TranslateGemma backend: {tg_config.get('model', 'google/translategemma-12b-it')}")
    else:
        nllb_config = trans_config.get("nllb", {})
        translator = create_translator(
            backend="nllb",
            model_name=nllb_config.get("model", "facebook/nllb-200-distilled-600M"),
            source_lang=trans_config.get("source_lang", "de"),
            target_lang=trans_config.get("target_lang", "fa"),
            device=trans_config.get("device", "mps"),
            max_length=nllb_config.get("max_length", 512),
        )
        logger.info(f"Using NLLB backend: {nllb_config.get('model', 'facebook/nllb-200-distilled-600M')}")
    
    # Synthesizer (Piper)
    tts_config = config.get("tts", {})
    try:
        synthesizer = Synthesizer(
            voice=tts_config.get("model", "fa_IR-amir-medium"),
            speaker_id=tts_config.get("speaker_id", 0),
            length_scale=tts_config.get("length_scale", 1.0),
        )
    except Exception as e:
        logger.warning(f"Could not initialize Piper TTS: {e}, using mock")
        synthesizer = MockSynthesizer()
    
    return audio_capture, vad, transcriber, translator, synthesizer


async def preload_models(transcriber, translator, synthesizer):
    """Preload all models to avoid first-request latency."""
    logger.info("Preloading models...")
    
    import concurrent.futures
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            loop.run_in_executor(executor, transcriber.load_model),
            loop.run_in_executor(executor, translator.load_model),
        ]
        await asyncio.gather(*futures)
    
    logger.info("Models preloaded")


def create_gemini_pipeline(config: dict):
    """Build the Gemini Live Translate pipeline (no local models needed)."""
    from src.gemini_translator import GeminiTranslationPipeline

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini Live mode requires an API key. Set GEMINI_API_KEY in your "
            ".env file (or environment) and restart. Get one at "
            "https://aistudio.google.com/apikey"
        )

    trans_config = config.get("translation", {})
    gemini_config = config.get("gemini", {})

    audio_capture = create_audio_capture(config)

    # Speech gate: only stream audio during speech so we don't pay to send
    # silence. Falls back to streaming everything if Silero VAD can't load.
    speech_gate = None
    if gemini_config.get("vad_gating", True):
        try:
            from src.vad import SpeechGate
            vad_config = config.get("vad", {})
            speech_gate = SpeechGate(
                threshold=vad_config.get("threshold", 0.5),
                sample_rate=config.get("audio", {}).get("sample_rate", 16000),
                pre_roll_duration=gemini_config.get("gate_pre_roll", 0.3),
                hangover_duration=gemini_config.get("gate_hangover", 0.8),
            )
        except Exception as e:
            logger.warning(f"Could not load speech gate ({e}); streaming all audio")
            speech_gate = None

    pipeline = GeminiTranslationPipeline(
        audio_capture=audio_capture,
        api_key=api_key,
        target_lang=trans_config.get("target_lang", "fa"),
        source_lang=trans_config.get("source_lang", "auto"),
        model=gemini_config.get("model", "gemini-3.5-live-translate-preview"),
        echo_target_language=gemini_config.get("echo_target_language", True),
        speech_gate=speech_gate,
    )
    logger.info(
        f"Gemini Live Translate mode enabled (no local models loaded, "
        f"vad_gating={'on' if speech_gate else 'off'})"
    )
    return pipeline


async def run_server(config: dict):
    """Run the translation server."""
    import uvicorn

    backend = config.get("translation", {}).get("backend", "nllb")

    if backend == "gemini-live":
        # Online mode: one Gemini Live session replaces the STT/Trans/TTS stack.
        logger.info("Initializing Gemini Live Translate pipeline...")
        pipeline = create_gemini_pipeline(config)
    else:
        # Local, offline mode: the four-model pipeline.
        logger.info("Initializing components...")
        audio_capture, vad, transcriber, translator, synthesizer = create_components(config)

        # Preload models
        await preload_models(transcriber, translator, synthesizer)

        # Create pipeline with parallel processing
        pipeline_config = config.get("pipeline", {})
        max_workers = pipeline_config.get("max_workers", 3)

        from src.pipeline import ParallelTranslationPipeline
        pipeline = ParallelTranslationPipeline(
            audio_capture=audio_capture,
            vad=vad,
            transcriber=transcriber,
            translator=translator,
            synthesizer=synthesizer,
            max_workers=max_workers,
        )

        logger.info(f"Pipeline configured with {max_workers} parallel workers")

    # Create app
    app = create_app(pipeline)

    # Server config
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    
    logger.info(f"Starting server on http://{host}:{port}")
    
    # Get local IP for easy access
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        logger.info(f"Connect on local network: http://{local_ip}:{port}")
    except Exception:
        pass
    
    # Run server
    config_uvicorn = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config_uvicorn)
    
    # Start pipeline and server
    pipeline_task = asyncio.create_task(pipeline.start())
    server_task = asyncio.create_task(server.serve())
    
    try:
        await asyncio.gather(pipeline_task, server_task)
    except asyncio.CancelledError:
        logger.info("Shutting down...")
        await pipeline.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live Translation System")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--test-audio",
        action="store_true",
        help="Test audio capture and exit"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to audio file for input simulation"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        from src.audio_capture import AudioCapture
        print("\nAvailable audio input devices:")
        for device in AudioCapture.list_devices():
            if device['inputs'] > 0:
                print(f"  [{device['id']}] {device['name']}")
                print(f"       Inputs: {device['inputs']}, Sample Rate: {device['sample_rate']}")
        return
    

    if args.test_audio:
        print("\nTesting audio capture (5 seconds)...")
        import time
        import numpy as np
        
        config = load_config(args.config)
        audio_config = config.get("audio", {})
        
        if args.input_file:
            from src.file_audio_capture import FileAudioCapture
            print(f"Testing with file: {args.input_file}")
            capture = FileAudioCapture(
                file_path=args.input_file,
                sample_rate=audio_config.get("sample_rate", 16000),
                loop=True
            )
        else:
            capture = AudioCapture(
                device_name=audio_config.get("device", "X32"),
                sample_rate=audio_config.get("sample_rate", 16000),
            )
            
        capture.start()
        
        start = time.time()
        while time.time() - start < 5:
            audio = capture.get_audio(timeout=0.5)
            if audio is not None:
                level = np.abs(audio).mean()
                bars = int(level * 100)
                print(f"\rLevel: {'█' * bars}{' ' * (50 - bars)}", end="")
        
        capture.stop()
        print("\n\nAudio test complete!")
        return
    
    # Load config and run
    config = load_config(args.config)
    
    # Inject input file into config if provided via CLI
    if args.input_file:
        config['audio']['input_file'] = args.input_file
    
    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
