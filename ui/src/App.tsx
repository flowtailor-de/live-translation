import { useAudioStream } from './hooks/useAudioStream';
import { JoinButton } from './components/JoinButton';
import { Activity, Radio, Signal } from 'lucide-react';
import './App.css';

function App() {
  const { status, transcript, latency, connect, disconnect } = useAudioStream();

  const handleToggle = () => {
    if (status === 'connected' || status === 'connecting') {
      disconnect();
    } else {
      connect();
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="live-badge">
          <div className={`status-dot ${status === 'connected' ? 'active' : ''}`} />
          <span>{status === 'connected' ? 'LIVE' : 'OFFLINE'}</span>
        </div>
        <h1>Live Translation</h1>
      </header>

      <main className="main-content">
        <div className="visualizer-placeholder">
          {status === 'connected' ? (
            <div className="waves">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="wave-bar" style={{ animationDelay: `${i * 0.1}s` }} />
              ))}
            </div>
          ) : (
            <Radio className="placeholder-icon" size={48} />
          )}
        </div>

        <div className="control-area">
          <JoinButton status={status} onClick={handleToggle} />

          <div className="stats-row">
            {status === 'connected' && (
              <>
                <div className="stat-item">
                  <Signal size={16} />
                  <span>{Math.round(latency)}ms</span>
                </div>
                <div className="stat-item">
                  <Activity size={16} />
                  <span>Good Quality</span>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="transcript-area">
          <h3>Latest Translations</h3>
          <div className="transcript-list">
            {transcript.length === 0 ? (
              <p className="empty-text">Waiting for speech...</p>
            ) : (
              transcript.map((item, idx) => (
                <div key={idx} className="transcript-item">
                  <p className="original">{item.original}</p>
                  <p className="translated">{item.translated}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
