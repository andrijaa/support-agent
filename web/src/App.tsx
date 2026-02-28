import { useState, useRef, useEffect } from 'react';
import { AudioBridgeClient, ConnectionState, ConversationData } from './AudioBridgeClient';

type AppStatus = 'idle' | 'connecting' | 'active' | 'completed';

function App() {
  const [status, setStatus] = useState<AppStatus>('idle');
  const [conversationData, setConversationData] = useState<ConversationData | null>(null);
  const clientRef = useRef<AudioBridgeClient | null>(null);
  const audioRefs = useRef<Map<string, HTMLAudioElement>>(new Map());

  const handleStart = async () => {
    if (status === 'active' || status === 'connecting') {
      clientRef.current?.disconnect();
      clientRef.current = null;
      setStatus('idle');
      return;
    }

    setStatus('connecting');
    setConversationData(null);

    const clientId = `web-${Math.random().toString(36).slice(2, 8)}`;
    const client = new AudioBridgeClient(clientId);
    clientRef.current = client;

    client.setCallbacks({
      onConnectionStateChange: (state: ConnectionState) => {
        if (state === 'connected') setStatus('active');
        else if (state === 'failed') setStatus('idle');
        else if (state === 'disconnected' && clientRef.current) {
          // Only go idle if we weren't completed
          setStatus(prev => prev === 'completed' ? 'completed' : 'idle');
        }
      },
      onAudioTrack: (_peerId, track) => {
        console.log('Received audio track from agent');
        let audio = audioRefs.current.get('agent');
        if (!audio) {
          audio = new Audio();
          audio.autoplay = true;
          audioRefs.current.set('agent', audio);
        }
        audio.srcObject = new MediaStream([track]);
        audio.play().catch(err => console.error('Audio play failed:', err));
      },
      onConversationData: (data: ConversationData) => {
        setConversationData(data);
        setStatus('completed');
      },
      onError: (error) => {
        console.error('Connection error:', error);
        setStatus('idle');
      },
    });

    try {
      await client.connect('ai-room');
    } catch {
      setStatus('idle');
    }
  };

  useEffect(() => {
    return () => {
      clientRef.current?.disconnect();
    };
  }, []);

  const statusText = {
    idle: 'Ready to start',
    connecting: 'Connecting...',
    active: 'Conversation active',
    completed: 'Conversation complete',
  }[status];

  const buttonText = {
    idle: 'Start Conversation',
    connecting: 'Connecting...',
    active: 'End Conversation',
    completed: 'Start New Conversation',
  }[status];

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Customer Support</h1>

      <div style={styles.center}>
        {status === 'active' && (
          <div style={styles.pulseWrapper}>
            <div className="pulse-ring" />
            <div style={styles.pulseCore} />
          </div>
        )}

        <button
          style={{
            ...styles.button,
            backgroundColor: status === 'active' ? '#ef4444' :
                             status === 'completed' ? '#22c55e' : '#3b82f6',
            opacity: status === 'connecting' ? 0.6 : 1,
          }}
          onClick={handleStart}
          disabled={status === 'connecting'}
        >
          {buttonText}
        </button>

        <p style={{
          ...styles.statusText,
          color: status === 'active' ? '#22c55e' :
                 status === 'completed' ? '#3b82f6' : '#9ca3af',
        }}>
          {statusText}
        </p>
      </div>

      {conversationData && (
        <div style={styles.resultCard}>
          <h2 style={styles.resultTitle}>Support Ticket</h2>
          <pre style={styles.json}>
            {JSON.stringify(conversationData.ticket, null, 2)}
          </pre>
          <p style={styles.summary}>{conversationData.summary}</p>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: '500px',
    margin: '0 auto',
    padding: '40px 20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    minHeight: '100vh',
  },
  title: {
    textAlign: 'center',
    marginBottom: '48px',
    color: '#1f2937',
    fontSize: '28px',
    fontWeight: '600',
  },
  center: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px',
  },
  pulseWrapper: {
    position: 'relative',
    width: '48px',
    height: '48px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pulseCore: {
    width: '16px',
    height: '16px',
    borderRadius: '50%',
    backgroundColor: '#22c55e',
  },
  button: {
    padding: '16px 48px',
    fontSize: '18px',
    fontWeight: '600',
    color: '#fff',
    border: 'none',
    borderRadius: '12px',
    cursor: 'pointer',
    transition: 'background-color 0.2s, opacity 0.2s',
  },
  statusText: {
    fontSize: '14px',
    margin: 0,
  },
  resultCard: {
    marginTop: '40px',
    backgroundColor: '#fff',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  },
  resultTitle: {
    margin: '0 0 16px 0',
    fontSize: '18px',
    color: '#374151',
    fontWeight: '600',
  },
  json: {
    backgroundColor: '#1f2937',
    color: '#d1d5db',
    borderRadius: '8px',
    padding: '16px',
    fontSize: '13px',
    overflow: 'auto',
    margin: '0 0 16px 0',
  },
  summary: {
    fontSize: '14px',
    color: '#6b7280',
    margin: 0,
    fontStyle: 'italic',
  },
};

export default App;
