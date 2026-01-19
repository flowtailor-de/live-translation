import { Mic, Loader2, StopCircle } from 'lucide-react';
import './JoinButton.css';

interface JoinButtonProps {
    status: 'disconnected' | 'connecting' | 'connected' | 'error';
    onClick: () => void;
}

export function JoinButton({ status, onClick }: JoinButtonProps) {
    const getContent = () => {
        switch (status) {
            case 'connecting':
                return (
                    <>
                        <Loader2 className="icon spin" size={32} />
                        <span>Connecting...</span>
                    </>
                );
            case 'connected':
                return (
                    <>
                        <StopCircle className="icon" size={32} />
                        <span>Leave Stream</span>
                    </>
                );
            case 'error':
                return (
                    <>
                        <Mic className="icon" size={32} />
                        <span>Retry Connection</span>
                    </>
                );
            default:
                return (
                    <>
                        <Mic className="icon" size={32} />
                        <span>Join Stream</span>
                    </>
                );
        }
    };

    return (
        <div className="button-container">
            {status === 'connected' && <div className="pulse-ring" />}
            <button
                className={`join-button ${status}`}
                onClick={onClick}
                disabled={status === 'connecting'}
            >
                {getContent()}
            </button>
        </div>
    );
}
