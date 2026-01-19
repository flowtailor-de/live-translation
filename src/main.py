"""
Main entry point for the Live Translation System.
"""

import asyncio
import yaml
import logging
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_capture import AudioCapture
from src.vad import VoiceActivityDetector, SimpleVAD
from src.transcriber import Transcriber
from src.translator import Translator
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


def create_components(config: dict):
    """Create all pipeline components from config."""
    
    # Audio capture
    # Audio capture
    audio_config = config.get("audio", {})
    if audio_config.get('input_file'):
        logger.info(f"Using file input: {audio_config['input_file']}")
        from src.file_audio_capture import FileAudioCapture
        audio_capture = FileAudioCapture(
            file_path=audio_config['input_file'],
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            chunk_duration=audio_config.get("chunk_duration", 0.1),
            loop=True
        )
    else:
        logger.info("Using microphone input")
        audio_capture = AudioCapture(
            device_name=audio_config.get("device", "X32"),
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            chunk_duration=audio_config.get("chunk_duration", 0.1),
        )
    
    # Voice Activity Detection
    vad_config = config.get("vad", {})
    try:
        vad = VoiceActivityDetector(
            threshold=vad_config.get("threshold", 0.5),
            min_speech_duration=vad_config.get("min_speech_duration", 0.3),
            min_silence_duration=vad_config.get("min_silence_duration", 0.4),
            max_speech_duration=vad_config.get("max_speech_duration", 5.0),
            speech_pad_duration=vad_config.get("speech_pad_duration", 0.2),
        )
    except Exception as e:
        logger.warning(f"Could not load Silero VAD: {e}, using simple VAD")
        vad = SimpleVAD(
            energy_threshold=0.01,
            min_speech_duration=vad_config.get("min_speech_duration", 0.3),
            min_silence_duration=vad_config.get("min_silence_duration", 0.4),
            max_speech_duration=vad_config.get("max_speech_duration", 5.0),
        )
    
    # Transcriber (Whisper)
    stt_config = config.get("stt", {})
    transcriber = Transcriber(
        model_size=stt_config.get("model", "medium"),
        device=stt_config.get("device", "auto"),
        compute_type=stt_config.get("compute_type", "float16"),
        language=stt_config.get("language", "de"),
    )
    
    # Translator (NLLB)
    trans_config = config.get("translation", {})
    translator = Translator(
        model_name=trans_config.get("model", "facebook/nllb-200-distilled-600M"),
        source_lang=trans_config.get("source_lang", "deu_Latn"),
        target_lang=trans_config.get("target_lang", "pes_Arab"),
        device=trans_config.get("device", "auto"),
        max_length=trans_config.get("max_length", 512),
    )
    
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


async def run_server(config: dict):
    """Run the translation server."""
    import uvicorn
    
    # Create components
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
