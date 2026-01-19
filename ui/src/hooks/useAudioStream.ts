import { useState, useEffect, useRef, useCallback } from 'react';

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

interface TranscriptItem {
    original: string;
    translated: string;
    latency: number;
}

interface AudioStreamState {
    status: ConnectionState;
    transcript: TranscriptItem[];
    latency: number;
    connect: () => void;
    disconnect: () => void;
}

export function useAudioStream(): AudioStreamState {
    const [status, setStatus] = useState<ConnectionState>('disconnected');
    const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
    const [latency, setLatency] = useState(0);

    const wsRef = useRef<WebSocket | null>(null);
    const audioCtxRef = useRef<AudioContext | null>(null);
    const nextStartTimeRef = useRef<number>(0);

    // Initialize AudioContext
    const initAudio = useCallback(() => {
        if (!audioCtxRef.current) {
            const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
            audioCtxRef.current = new AudioContext();
        }
        if (audioCtxRef.current.state === 'suspended') {
            audioCtxRef.current.resume();
        }
    }, []);

    const playAudioChunk = useCallback(async (data: ArrayBuffer) => {
        if (!audioCtxRef.current) return;

        try {
            const audioBuffer = await audioCtxRef.current.decodeAudioData(data);
            const source = audioCtxRef.current.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioCtxRef.current.destination);

            const currentTime = audioCtxRef.current.currentTime;
            // Simple scheduling to prevent overlap/gaps
            // If next start time is in the past, reset to current time
            const startTime = Math.max(currentTime, nextStartTimeRef.current);

            source.start(startTime);
            nextStartTimeRef.current = startTime + audioBuffer.duration;
        } catch (err) {
            console.error('Error decoding audio:', err);
        }
    }, []);

    const connect = useCallback(() => {
        initAudio();
        setStatus('connecting');

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // For development, fallback to localhost:8000 if running on port 5173
        const host = window.location.port === '5173' ? 'localhost:8000' : window.location.host;
        const wsUrl = `${protocol}//${host}/ws`;

        const ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            setStatus('connected');
            console.log('Connected to WebSocket');
        };

        ws.onmessage = async (event) => {
            if (typeof event.data === 'string') {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'transcript') {
                        setTranscript(prev => [{
                            original: msg.original,
                            translated: msg.translated,
                            latency: msg.latency
                        }, ...prev].slice(0, 50)); // Keep last 50 items
                        setLatency(msg.latency);
                    }
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            } else if (event.data instanceof ArrayBuffer) {
                await playAudioChunk(event.data);
            }
        };

        ws.onclose = () => {
            setStatus('disconnected');
            wsRef.current = null;
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            setStatus('error');
        };

        wsRef.current = ws;
    }, [initAudio, playAudioChunk]);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setStatus('disconnected');
        // Close audio context to stop playing
        if (audioCtxRef.current) {
            audioCtxRef.current.close();
            audioCtxRef.current = null;
        }
        nextStartTimeRef.current = 0;
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (audioCtxRef.current) {
                audioCtxRef.current.close();
            }
        };
    }, []);

    return { status, transcript, latency, connect, disconnect };
}
