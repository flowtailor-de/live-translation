"""
Shared lock for MLX operations.

MLX (Apple Metal) is not thread-safe - only one MLX operation can
run at a time without causing GPU command buffer conflicts.

Both the transcriber (MLX Whisper) and translator (TranslateGemma MLX)
must use this shared lock to serialize GPU operations.
"""

import threading

# Global lock shared across all MLX operations
mlx_lock = threading.Lock()
