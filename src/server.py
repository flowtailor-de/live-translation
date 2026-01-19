"""
Web server for streaming translated audio to clients.
"""

import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import wave
import io
import logging
import os

logger = logging.getLogger(__name__)

app = FastAPI(title="Live Translation Server")

# Mount web directory
web_dir = os.path.join(os.path.dirname(__file__), "..", "web")

# We mount /assets explicitly to ensure it works
assets_dir = os.path.join(web_dir, "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

class ConnectionManager:
    """Manages WebSocket connections to clients."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_audio(self, audio_data: bytes) -> None:
        """Broadcast audio data to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_bytes(audio_data)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_text(self, message: str) -> None:
        """Broadcast text message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


# Global state
manager = ConnectionManager()
pipeline_ref = None
is_streaming = False


def set_pipeline(pipeline) -> None:
    """Set the pipeline reference for the server."""
    global pipeline_ref
    pipeline_ref = pipeline


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio to WAV bytes."""
    # Ensure float32 to int16 conversion
    if audio.dtype == np.float32:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)
    
    # Write to bytes buffer
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    return buffer.getvalue()


@app.get("/")
async def index():
    """Serve the main React app."""
    index_path = os.path.join(web_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return HTMLResponse(content="<h1>Error: Frontend not built. Run 'npm run build' in 'ui/' directory.</h1>")

# Serve other static files from root (like vite.svg) if they exist
@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files from web root if they exist."""
    # Skip API and WS routes (though FastAPI checks regex order, explicit checks are safer if ambiguous)
    if file_path.startswith("api/") or file_path == "ws":
        return JSONResponse({"error": "Not found"}, status_code=404)
        
    full_path = os.path.join(web_dir, file_path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path)
    
    return JSONResponse({"error": "File not found"}, status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive, receive any client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Handle ping/pong or other messages
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text('{"type": "keepalive"}')
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/api/status")
async def get_status():
    """Get server and pipeline status."""
    status = {
        "connected_clients": len(manager.active_connections),
        "is_streaming": is_streaming,
    }
    
    if pipeline_ref:
        status.update(pipeline_ref.get_stats())
    
    return JSONResponse(status)


@app.post("/api/start")
async def start_streaming():
    """Start the translation pipeline."""
    global is_streaming
    
    if pipeline_ref:
        asyncio.create_task(pipeline_ref.start())
        is_streaming = True
        return {"status": "started"}
    
    return {"status": "error", "message": "Pipeline not initialized"}


@app.post("/api/stop")
async def stop_streaming():
    """Stop the translation pipeline."""
    global is_streaming
    
    if pipeline_ref:
        await pipeline_ref.stop()
        is_streaming = False
        return {"status": "stopped"}
    
    return {"status": "error", "message": "Pipeline not initialized"}


async def broadcast_result(result) -> None:
    """Broadcast a translation result to all clients."""
    import json
    
    # Send transcript as JSON
    transcript_msg = json.dumps({
        "type": "transcript",
        "original": result.original_text,
        "translated": result.translated_text,
        "latency": result.latency_ms,
    })
    await manager.broadcast_text(transcript_msg)
    
    # Send audio as binary
    if len(result.audio_data) > 0:
        wav_bytes = audio_to_wav_bytes(result.audio_data, result.sample_rate)
        await manager.broadcast_audio(wav_bytes)


def create_app(pipeline=None):
    """Create and configure the FastAPI app."""
    if pipeline:
        set_pipeline(pipeline)
        # Set the broadcast callback
        pipeline.on_result = lambda r: asyncio.create_task(broadcast_result(r))
    
    return app


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8000)
