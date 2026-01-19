# Live Translation System

Real-time German to Farsi audio translation for church services.

## Hardware Setup
- **Mac**: MacBook Pro M1 Max, 32GB RAM
- **Audio**: Behringer X32 (USB)
- **Network**: Ethernet (recommended)
- **Clients**: 10-15 web browsers

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Download models (first time only)
python -m src.download_models

# Start the server
python -m src.main
```

Then open `http://<mac-ip>:8000` on client devices.

## Architecture

```
Audio In → VAD → Whisper STT → NLLB Translation → Piper TTS → Stream → Clients
```

## Project Structure

```
src/
├── main.py           # Entry point
├── audio_capture.py  # USB audio input
├── vad.py            # Voice activity detection
├── transcriber.py    # German STT
├── translator.py     # DE→FA translation
├── synthesizer.py    # Farsi TTS
├── pipeline.py       # Async orchestration
└── server.py         # Web streaming
```

## Requirements
- Python 3.10+
- M1/M2/M3 Mac (for MPS acceleration)
- ~8GB disk space for models
