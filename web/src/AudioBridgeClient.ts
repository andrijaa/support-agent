export interface SignalMessage {
  type: string;
  room?: string;
  client_id?: string;
  sdp?: string;
  candidate?: string;
  ticket?: Record<string, unknown>;
  summary?: string;
  conversation_history?: Array<{ role: string; content: string }>;
}

export interface ConversationData {
  ticket: Record<string, unknown>;
  summary: string;
  conversation_history: Array<{ role: string; content: string }>;
}

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'failed';

export interface AudioBridgeCallbacks {
  onConnectionStateChange?: (state: ConnectionState) => void;
  onPeerJoined?: (peerId: string) => void;
  onPeerLeft?: (peerId: string) => void;
  onAudioTrack?: (peerId: string, track: MediaStreamTrack) => void;
  onError?: (error: string) => void;
  onConversationData?: (data: ConversationData) => void;
}

export class AudioBridgeClient {
  private ws: WebSocket | null = null;
  private pc: RTCPeerConnection | null = null;
  private localStream: MediaStream | null = null;
  private callbacks: AudioBridgeCallbacks = {};
  private clientId: string;
  private serverUrl: string;

  constructor(clientId: string, serverUrl: string = 'ws://localhost:8080/ws') {
    this.clientId = clientId;
    this.serverUrl = serverUrl;
  }

  setCallbacks(callbacks: AudioBridgeCallbacks) {
    this.callbacks = callbacks;
  }

  async connect(room: string): Promise<void> {
    this.callbacks.onConnectionStateChange?.('connecting');

    try {
      // Get microphone access
      this.localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      // Connect WebSocket
      this.ws = new WebSocket(this.serverUrl);

      await new Promise<void>((resolve, reject) => {
        this.ws!.onopen = () => resolve();
        this.ws!.onerror = () => reject(new Error('WebSocket connection failed'));
        setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
      });

      // Create PeerConnection
      this.pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });

      // Add local audio track
      this.localStream.getAudioTracks().forEach(track => {
        this.pc!.addTrack(track, this.localStream!);
      });

      // Handle ICE candidates
      this.pc.onicecandidate = (event) => {
        if (event.candidate) {
          this.sendMessage({
            type: 'candidate',
            candidate: event.candidate.candidate,
          });
        }
      };

      // Handle incoming tracks
      this.pc.ontrack = (event) => {
        console.log('Received track:', event.track.kind, event.streams);
        if (event.track.kind === 'audio') {
          // Extract peer ID from stream ID (format: stream-peerID)
          const streamId = event.streams[0]?.id || 'unknown';
          const peerId = streamId.startsWith('stream-') ? streamId.slice(7) : streamId;
          this.callbacks.onAudioTrack?.(peerId, event.track);
        }
      };

      // Handle connection state
      this.pc.onconnectionstatechange = () => {
        console.log('Connection state:', this.pc?.connectionState);
        switch (this.pc?.connectionState) {
          case 'connected':
            this.callbacks.onConnectionStateChange?.('connected');
            break;
          case 'failed':
            this.callbacks.onConnectionStateChange?.('failed');
            break;
          case 'disconnected':
          case 'closed':
            this.callbacks.onConnectionStateChange?.('disconnected');
            break;
        }
      };

      // Handle WebSocket messages
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        this.handleSignalMessage(msg);
      };

      this.ws.onclose = () => {
        this.callbacks.onConnectionStateChange?.('disconnected');
      };

      // Join the room
      this.sendMessage({
        type: 'join',
        room: room,
        client_id: this.clientId,
      });

    } catch (error) {
      this.callbacks.onError?.(error instanceof Error ? error.message : 'Connection failed');
      this.callbacks.onConnectionStateChange?.('failed');
      throw error;
    }
  }

  private async handleSignalMessage(msg: SignalMessage) {
    console.log('Received signal:', msg.type);

    switch (msg.type) {
      case 'offer':
        await this.handleOffer(msg);
        break;
      case 'answer':
        await this.handleAnswer(msg);
        break;
      case 'candidate':
        await this.handleCandidate(msg);
        break;
      case 'peer_joined':
        this.callbacks.onPeerJoined?.(msg.client_id || 'unknown');
        break;
      case 'peer_left':
        this.callbacks.onPeerLeft?.(msg.client_id || 'unknown');
        break;
      case 'conversation_data':
        if (msg.ticket && msg.summary !== undefined && msg.conversation_history) {
          this.callbacks.onConversationData?.({
            ticket: msg.ticket,
            summary: msg.summary,
            conversation_history: msg.conversation_history,
          });
        }
        break;
    }
  }

  private async handleOffer(msg: SignalMessage) {
    if (!this.pc || !msg.sdp) return;

    try {
      await this.pc.setRemoteDescription({
        type: 'offer',
        sdp: msg.sdp,
      });

      const answer = await this.pc.createAnswer();
      await this.pc.setLocalDescription(answer);

      this.sendMessage({
        type: 'answer',
        sdp: answer.sdp,
      });
    } catch (error) {
      console.error('Failed to handle offer:', error);
    }
  }

  private async handleAnswer(msg: SignalMessage) {
    if (!this.pc || !msg.sdp) return;

    try {
      await this.pc.setRemoteDescription({
        type: 'answer',
        sdp: msg.sdp,
      });
    } catch (error) {
      console.error('Failed to handle answer:', error);
    }
  }

  private async handleCandidate(msg: SignalMessage) {
    if (!this.pc || !msg.candidate) return;

    try {
      await this.pc.addIceCandidate({
        candidate: msg.candidate,
      });
    } catch (error) {
      console.error('Failed to add ICE candidate:', error);
    }
  }

  private sendMessage(msg: SignalMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  disconnect() {
    this.localStream?.getTracks().forEach(track => track.stop());
    this.pc?.close();
    this.ws?.close();
    this.localStream = null;
    this.pc = null;
    this.ws = null;
    this.callbacks.onConnectionStateChange?.('disconnected');
  }

  // Mute/unmute microphone
  setMuted(muted: boolean) {
    this.localStream?.getAudioTracks().forEach(track => {
      track.enabled = !muted;
    });
  }
}
