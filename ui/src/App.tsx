import { useAudioStream } from './hooks/useAudioStream';
import { JoinButton } from './components/JoinButton';
import './App.css';

function App() {
  const { status, transcript, languages, connect, disconnect } = useAudioStream();

  const handleToggle = () => {
    if (status === 'connected' || status === 'connecting') {
      disconnect();
    } else {
      connect();
    }
  };

  // Format transcript text in the brand-dot style: word-by-word with dots
  const formatBrandDot = (text: string) => {
    return text
      .split(/\s+/)
      .filter(Boolean)
      .map(w => w.toLowerCase())
      .join('. ')
      .concat('.');
  };

  return (
    <div className="app-container">
      {/* ─── Header ─── */}
      <header className="app-header">
        <div className="app-header__brand">
          open.translate.live
        </div>
        <div className={`app-header__badge ${status === 'connected' ? 'app-header__badge--live' : ''}`}>
          <div className="app-header__badge-dot" />
          <span className="app-header__badge-text">
            {status === 'connected' ? 'Live' : 'Offline'}
          </span>
        </div>
      </header>

      {/* ─── Main ─── */}
      <main className="app-main">
        {/* Stream button */}
        <div className="app-main__button-area">
          <JoinButton status={status} onClick={handleToggle} />
        </div>

        {/* Transcript area */}
        <div className="app-main__transcript">
          {/* Top gradient fade */}
          <div className="app-main__transcript-fade app-main__transcript-fade--top" />

          <div className="app-main__transcript-scroll custom-scrollbar">
            {transcript.length === 0 ? (
              <p className="brand-dot-text app-main__transcript-empty">
                {status === 'connected'
                  ? 'waiting. for. speech...'
                  : 'tap. the. button. to. join. the. live. stream.'}
              </p>
            ) : (
              <>
                {transcript.map((item, idx) => (
                  <div
                    key={idx}
                    className={`brand-dot-text animate-slide-in ${idx === 0
                      ? 'app-main__transcript-item--latest'
                      : 'app-main__transcript-item--older'
                      }`}
                  >
                    {formatBrandDot(item.translated)}
                  </div>
                ))}

                {/* Bouncing dots – listening indicator */}
                {status === 'connected' && (
                  <div className="app-main__listening-dots">
                    <div className="app-main__dot animate-bounce-dot" />
                    <div className="app-main__dot animate-bounce-dot" style={{ animationDelay: '0.2s' }} />
                    <div className="app-main__dot animate-bounce-dot" style={{ animationDelay: '0.4s' }} />
                  </div>
                )}
              </>
            )}
          </div>

          {/* Bottom gradient fade */}
          <div className="app-main__transcript-fade app-main__transcript-fade--bottom" />
        </div>
      </main>

      {/* ─── Footer ─── */}
      <footer className="app-footer">
        <div className="app-footer__content">
          {/* Language selector */}
          <div className="app-footer__languages">
            <div className="app-footer__lang">
              <span className="app-footer__lang-label">Source</span>
              <div className="app-footer__lang-pill app-footer__lang-pill--source">
                <span>{languages.sourceLang}</span>
              </div>
            </div>

            <span className="material-symbols-outlined app-footer__arrow">arrow_forward</span>

            <div className="app-footer__lang">
              <span className="app-footer__lang-label app-footer__lang-label--target">Target</span>
              <div className="app-footer__lang-pill app-footer__lang-pill--target">
                <span>{languages.targetLang}</span>
              </div>
            </div>
          </div>

          {/* Divider */}
          <div className="app-footer__divider" />

          {/* Action buttons */}
          <div className="app-footer__actions">
            <button className="app-footer__action-btn">
              <span className="material-symbols-outlined">volume_up</span>
            </button>
            <button className="app-footer__action-btn">
              <span className="material-symbols-outlined">settings</span>
            </button>
          </div>
        </div>

        {/* Home indicator */}
        <div className="app-footer__indicator">
          <div className="app-footer__indicator-bar" />
        </div>
      </footer>
    </div>
  );
}

export default App;
