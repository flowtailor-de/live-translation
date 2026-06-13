# CLAUDE.md

Guidance for working in this repo. Keep it accurate — update it when the architecture changes.

## What this is

A real-time, low-latency **audio translation** system for live events (default: German → Farsi).
Mic audio in → translated audio + subtitles out to phones over a web server. Built for Apple
Silicon. Backend: Python + FastAPI + AsyncIO. Frontend: React + Vite (TypeScript) listener app.

## The one mental model: 3 stages, swappable middle

The system is an assembly line with **3 stages**:

1. **Capture** — `src/audio_capture.py` (mic / X32) or `src/file_audio_capture.py` (file).
2. **Translate** — the only stage that varies; chosen by `translation.backend` in `config.yaml`.
3. **Broadcast** — `src/server.py` (FastAPI WebSocket) → React app in `ui/` (served built from `web/`).

Both translate engines honor the **same contract** so stages 1 & 3 never change:
`start()`, `stop()`, `get_stats()`, and an `on_result(TranslationResult)` callback.
`TranslationResult` is defined in `src/pipeline.py`. `server.create_app(pipeline)` wires
`on_result` → `broadcast_result` (transcript JSON + a self-contained WAV; the frontend decodes
the WAV by header, so the sample rate just needs to be correct in the WAV).

### Two translate engines

| `backend` | Engine | File | Notes |
| --- | --- | --- | --- |
| `translategemma-mlx` (default), `translategemma`, `nllb` | **Local, offline** pipeline: Silero VAD → Whisper(MLX) → translate → Piper TTS | `src/pipeline.py` (`ParallelTranslationPipeline`) + `src/vad.py`, `src/transcriber.py`, `src/translator.py`, `src/synthesizer.py` | Runs entirely on the Mac. |
| `gemini-live` | **Online** Gemini Live Translate (`gemini-3.5-live-translate-preview`) | `src/gemini_translator.py` (`GeminiTranslationPipeline`) | One bidirectional WebSocket session replaces STT+translate+TTS. |

The engine is chosen in `src/main.py::run_server()` (it branches on `backend` before building
components, so Gemini mode skips loading all local models / `preload_models`).

## Gemini mode specifics

- **Auth**: `GEMINI_API_KEY` read from env (auto-loaded from `.env` via `python-dotenv`).
  Missing key fails loudly at startup. See `.env.example`.
- **Source language is auto-detected** by Gemini → the launcher's Source dropdown is inert in
  this mode. `translation.target_lang` drives the output language.
- **Per-turn aggregation**: the receive loop accumulates `input_transcription` (→ original),
  `output_transcription` (→ translated) and 24 kHz audio until `server_content.turn_complete`,
  then emits one `TranslationResult(sample_rate=24000)`.
- **Reconnect loop**: Gemini connections drop every ~10–15 min; `start()` reconnects so long
  (multi-hour) events keep running. State persists across reconnects (mic stream is continuous).
- **Speech gate** (`src/vad.py::SpeechGate`): only audio during speech is streamed (pre-roll +
  hangover guards against clipping) so we don't pay to stream silence. Billed per second of
  audio; **output audio dominates cost** (~$0.0315/min vs input ~$0.0053/min). Toggle/tune via
  `gemini.vad_gating` / `gate_pre_roll` / `gate_hangover` in `config.yaml`. Falls back to
  streaming everything if Silero can't load.
- Reference: the `gemini-live-api-dev` skill has full Live API details.

## Config

Single source of truth: `config.yaml`. Key knobs: `translation.backend` (engine switch),
`translation.target_lang`, `audio.device`, `gemini.*`. The desktop launcher (`launcher.py`,
customtkinter) reads/writes a subset and exposes `backend` as the "Model Backend" dropdown.

## Run / verify

```bash
./bin/setup.sh                       # venv + deps + model downloads
# or: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python -m src.main                   # backend (uses config.yaml)
cd ui && npm run dev                 # frontend (dev), or build into web/ for prod serving
```

Gemini mode: copy `.env.example` → `.env`, set `GEMINI_API_KEY`, set
`translation.backend: gemini-live`. Open `http://localhost:5173`, Join, speak → hear the
translation + see the transcript. `/api/status` exposes `source_lang`/`target_lang` + stats.

## Conventions / gotchas

- Audio: mic capture is float32 mono 16 kHz. Local pipeline output sample rate varies (Piper);
  Gemini output is 24 kHz. The WAV header carries the rate — don't hardcode it on the client.
- Adding a new translate engine = implement the `start/stop/get_stats/on_result` contract and add
  a branch in `run_server()`; server/frontend need no changes.
- `.env` is gitignored; never commit keys. Commit `.env.example` instead.
- Platform: macOS / Apple Silicon (MLX). The Gemini path is platform-agnostic (no MLX needed).
