import './JoinButton.css';

interface JoinButtonProps {
    status: 'disconnected' | 'connecting' | 'connected' | 'error';
    onClick: () => void;
}

export function JoinButton({ status, onClick }: JoinButtonProps) {
    const isActive = status === 'connected' || status === 'connecting';

    const getLabel = () => {
        switch (status) {
            case 'connecting':
                return (
                    <div className="stream-button__spinner animate-spin" />
                );
            case 'connected':
                return (
                    <>
                        <span className="stream-button__label">leave.</span>
                        <span className="stream-button__label">stream.</span>
                    </>
                );
            case 'error':
                return (
                    <>
                        <span className="stream-button__label">retry.</span>
                        <span className="stream-button__label">stream.</span>
                    </>
                );
            default:
                return (
                    <>
                        <span className="stream-button__label">join.</span>
                        <span className="stream-button__label">stream.</span>
                    </>
                );
        }
    };

    return (
        <div className="stream-button-container">
            {/* Pulse rings – always rendered, animated only when active */}
            {isActive && (
                <>
                    <div className="pulse-ring pulse-ring--inner animate-pulse-ring" />
                    <div
                        className="pulse-ring pulse-ring--outer animate-pulse-ring"
                        style={{ animationDelay: '1s' }}
                    />
                </>
            )}

            <button
                className="stream-button"
                onClick={onClick}
                disabled={status === 'connecting'}
            >
                {getLabel()}
            </button>
        </div>
    );
}
