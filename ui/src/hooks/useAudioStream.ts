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

    // Initialize AudioContext safely and unlock for iOS
    const initAudio = useCallback(async () => {
        try {
            if (!audioCtxRef.current) {
                const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
                audioCtxRef.current = new AudioContext();
            }

            // Unlock audio on iOS by playing a short silent buffer
            if (audioCtxRef.current) {
                const buffer = audioCtxRef.current.createBuffer(1, 1, 22050);
                const source = audioCtxRef.current.createBufferSource();
                source.buffer = buffer;
                source.connect(audioCtxRef.current.destination);
                source.start(0);

                if (audioCtxRef.current.state === 'suspended') {
                    await audioCtxRef.current.resume();
                }
            }
        } catch (e) {
            console.error('Audio Init Error:', e);
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
            const startTime = Math.max(currentTime, nextStartTimeRef.current);
            source.start(startTime);
            nextStartTimeRef.current = startTime + audioBuffer.duration;
        } catch (err) {
            // console.error('Error decoding audio:', err); 
            // Silent for now to avoid spam
        }
    }, []);

    const connect = useCallback(async () => {
        addLog('Connect called');

        // Initialize AudioContext immediately on user gesture (click)
        try {
            await initAudio();
        } catch (e) {
            addLog(`Audio init error: ${e}`);
        }

        setStatus('connecting');

        // Connect via Vite Proxy (same host/port as UI)
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host; // e.g., 192.168.1.x:5173
        const wsUrl = `${protocol}//${host}/ws`;

        addLog(`Attempting WS: ${wsUrl}`);

        let ws: WebSocket;
        try {
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer'; // Set binary type immediately
            addLog('WebSocket instance created');
        } catch (e) {
            addLog(`WS Creation Error: ${e}`);
            alert(`WS Creation Error: ${e}`);
            setStatus('error');
            return;
        }

        ws.onopen = () => {
            addLog('WS Open event fired!');
            // alert('WS OPENED!'); // Uncomment if needed
            setStatus('connected');

            // Send a ping message immediately to start traffic
            try {
                ws.send('ping');
                addLog('Ping sent');
            } catch (e) {
                addLog(`Ping error: ${e}`);
            }
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
                        }, ...prev].slice(0, 50));
                        setLatency(msg.latency);
                    }
                } catch (e) { }
            } else if (event.data instanceof ArrayBuffer) {
                await playAudioChunk(event.data);
            }
        };

        ws.onclose = (event) => {
            addLog(`WS Closed: ${event.code} ${event.reason}`);
            // alert(`WS CLOSED: ${event.code} ${event.reason}`);
            setStatus('disconnected');
            wsRef.current = null;
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addLog('WS Error occurred');
            // alert('WS ERROR!');
            setStatus('error');
        };

        wsRef.current = ws;
    }, [initAudio, playAudioChunk, addLog]);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setStatus('disconnected');

        if (audioCtxRef.current) {
            // Don't close context, just suspend to allow reuse? 
            // Better to close for full reset
            audioCtxRef.current.close().catch(e => console.error(e));
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
                audioCtxRef.current.close().catch(() => { });
            }
        };
    }, []);

    // Global error handler for crashes
    useEffect(() => {
        const handleError = (event: ErrorEvent) => {
            console.error('Global Error Event:', event);
        };
        const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
            console.error('Unhandled Rejection Event:', event);
        };

        window.addEventListener('error', handleError);
        window.addEventListener('unhandledrejection', handleUnhandledRejection);

        return () => {
            window.removeEventListener('error', handleError);
            window.removeEventListener('unhandledrejection', handleUnhandledRejection);
        };
    }, []);

    return { status, transcript, latency, connect, disconnect };
}
