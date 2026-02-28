# Python Conversational Customer Support Agent

A real-time voice-based customer support agent that conducts natural conversations over WebRTC. The agent collects support ticket information through spoken dialogue — order number, problem category, description, and urgency — then creates a structured ticket once the customer confirms.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Setup](#setup)
- [Running](#running)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Audio Pipeline](#audio-pipeline)
- [Conversation Flow](#conversation-flow)
- [Conversation Logs](#conversation-logs)
- [Potential Improvements](#potential-improvements)

## Architecture Overview

The system is split into three independently-running processes that communicate over WebRTC and WebSockets:

```
┌─────────────────┐       WebSocket        ┌──────────────────────┐
│                 │◄──── signaling ────────►│                      │
│   Browser UI    │                         │   SFU Server         │
│   (React/Vite)  │◄──── WebRTC ──────────►│   (FastAPI + aiortc) │
│                 │    audio + data         │                      │
└─────────────────┘                         └──────────┬───────────┘
                                                       │ WebRTC
                                                       │ audio + data
                                            ┌──────────▼───────────┐
                                            │   Support Agent      │
                                            │   (Python)           │
                                            │                      │
                                            │   ┌──────────────┐   │
                                            │   │ Deepgram STT │   │
                                            │   │  (nova-2)    │   │
                                            │   └──────┬───────┘   │
                                            │          ▼           │
                                            │   ┌──────────────┐   │
                                            │   │ OpenAI LLM   │   │
                                            │   │  (gpt-4o)    │   │
                                            │   └──────┬───────┘   │
                                            │          ▼           │
                                            │   ┌──────────────┐   │
                                            │   │ ElevenLabs   │   │
                                            │   │  TTS         │   │
                                            │   └──────────────┘   │
                                            └──────────────────────┘
```

**SFU Server** — A Selective Forwarding Unit built on FastAPI and aiortc. It manages WebSocket signaling (offer/answer/ICE candidates) and relays audio between peers using `MediaRelay`. The server doesn't decode or process audio — it forwards RTP packets as-is between the browser and agent.

**Support Agent** — A Python process that joins the SFU as a WebRTC peer. It receives the user's audio, runs it through a streaming STT → LLM → TTS pipeline, and sends synthesized speech back. It also extracts structured ticket data from the conversation and sends it to the UI via WebSocket when complete.

**Browser UI** — A React + TypeScript app that captures the user's microphone, establishes a WebRTC connection through the SFU, plays back the agent's audio, and displays the final support ticket.

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- API keys for [Deepgram](https://deepgram.com), [OpenAI](https://platform.openai.com), and [ElevenLabs](https://elevenlabs.io)

### Python Environment

```bash
cd conversational_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

| Variable | Service | Used For |
|----------|---------|----------|
| `DEEPGRAM_API_KEY` | Deepgram | Streaming speech-to-text (nova-2) |
| `OPENAI_API_KEY` | OpenAI | Conversational LLM (gpt-4o) and ticket extraction (gpt-4o-mini) |
| `ELEVENLABS_API_KEY` | ElevenLabs | Text-to-speech (eleven_turbo_v2_5, Rachel voice) |

### Web UI

```bash
cd web
npm install
```

## Running

Start all three processes in separate terminals:

**Terminal 1 — SFU Server:**
```bash
python -m server.main
# WebSocket: ws://localhost:8080/ws
# Health check: curl localhost:8080/health
```

**Terminal 2 — Support Agent:**
```bash
python -m agent.main --id support-agent --room ai-room
```

**Terminal 3 — React UI:**
```bash
cd web
npm run dev
# Open http://localhost:3000
```

Click "Start Conversation" in the browser. The agent will greet you and begin collecting ticket information.

## Project Structure

```
conversational_agent/
├── server/                         # SFU signaling + audio relay
│   ├── main.py                     # FastAPI app, /ws WebSocket, /health
│   ├── models.py                   # SignalMessage Pydantic model
│   ├── room.py                     # Room + RoomManager
│   └── peer.py                     # Peer (RTCPeerConnection + MediaRelay)
│
├── agent/                          # Voice support agent
│   ├── main.py                     # CLI entrypoint (arg parsing, wiring)
│   ├── agent.py                    # SupportAgent orchestrator (STT→LLM→TTS)
│   ├── client.py                   # AgentClient (WebRTC + WS signaling)
│   ├── stt.py                      # Deepgram streaming STT client
│   ├── llm.py                      # OpenAI LLM + SupportTicketData extraction
│   ├── tts.py                      # ElevenLabs WebSocket streaming TTS
│   └── audio.py                    # TTSAudioTrack + PCM conversion utilities
│
├── web/                            # React UI
│   └── src/
│       ├── App.tsx                 # Main UI (connection states, ticket display)
│       └── AudioBridgeClient.ts    # WebRTC peer + signaling client
│
├── config/
│   └── support_agent.json          # Agent persona + system prompt
│
├── conversations/                  # Auto-saved conversation JSON logs
├── requirements.txt
└── .env.example
```

## Key Design Decisions

### SFU over Peer-to-Peer

The server acts as a Selective Forwarding Unit rather than having the browser connect directly to the agent. This keeps the agent's network address private, allows multiple peers in a room, and gives the server a natural place to relay metadata (like ticket data) alongside audio. The `MediaRelay` in aiortc handles efficient RTP packet forwarding without decoding.

### Streaming Everything End-to-End

Every stage of the pipeline is streaming: Deepgram streams transcripts as the user speaks, OpenAI streams LLM tokens as they're generated, and ElevenLabs converts those tokens to audio in real-time via WebSocket. This means the agent starts speaking before the full LLM response is generated, significantly reducing perceived latency. The alternative — waiting for the full response before starting TTS — would add seconds of dead air on every turn.

### Transcript-Based Barge-In (Not VAD)

When the user starts speaking while the agent is talking, the agent should stop and listen. The naive approach is to use Deepgram's Voice Activity Detection (VAD) `speech_started` event to trigger interruption. However, the agent's own TTS audio loops back through the user's microphone, causing VAD to fire on echo and produce false barge-ins.

Instead, barge-in is triggered only when Deepgram produces an actual **final transcript** while the agent is speaking. Deepgram's speech model can distinguish real speech from echo/noise far more reliably than raw VAD energy detection. This eliminates phantom interruptions at the cost of a small additional delay (~200-500ms for Deepgram to produce a transcript vs. instant VAD), which is an acceptable trade-off for conversation stability.

### Dual-Model Ticket Extraction

The conversational LLM (gpt-4o) handles dialogue, while a separate gpt-4o-mini call extracts structured ticket data into JSON after each turn. This separation means the conversational model doesn't need to juggle both natural dialogue and structured output in the same response. The extraction model uses `response_format: json_object` for reliable parsing. Extraction runs concurrently after each turn completes.

### Confirmation Before Ticket Creation

The agent doesn't end the conversation the moment all ticket fields are populated. Instead, a `_pending_confirmation` flag is set, and the LLM naturally summarizes the collected details and asks the customer to confirm. Only after the customer's next response (confirming the details) does the ticket get created and the session end. This prevents premature closure when the extraction model happens to fill all fields before the customer is ready.

### Interrupt Flag on Audio Track

When barge-in is detected, simply clearing the audio queue isn't enough — the TTS WebSocket may still be streaming audio that gets pushed to the queue between the cancel signal and actual task cancellation. The `TTSAudioTrack` uses an `_interrupted` flag that causes `push_pcm()` to drop all incoming audio and `recv()` to return silence immediately, ensuring the user hears silence within one 20ms frame of the interrupt.

### 20ms Frame Pacing

`TTSAudioTrack.recv()` paces output at real-time rate (one 960-sample frame per 20ms) using wall-clock timing. Without this, aiortc would pull frames as fast as possible, sending RTP packets in bursts that overwhelm the receiver's jitter buffer and cause audio glitches.

## Audio Pipeline

**Inbound (browser → agent → Deepgram):**
```
Browser mic → Opus (WebRTC) → aiortc decode → av.AudioFrame (48kHz stereo)
  → av_frame_to_pcm_bytes() → interleaved int16 PCM → Deepgram STT
```

**Outbound (ElevenLabs → agent → browser):**
```
ElevenLabs WebSocket → PCM int16 22050Hz mono
  → resample_pcm() 22050→48000 (scipy polyphase) → stereo
  → 20ms av.AudioFrame chunks → TTSAudioTrack queue
  → aiortc Opus encode → RTP → WebRTC → browser speaker
```

## Conversation Flow

1. **Greeting** — Agent sends a pre-written greeting via TTS asking for the order number.
2. **Information Gathering** — Each user utterance triggers: Deepgram transcription → OpenAI streaming response → ElevenLabs TTS. After each turn, gpt-4o-mini extracts ticket fields from the full conversation history.
3. **Confirmation** — Once all four fields are populated (order number, category, description, urgency), the agent summarizes and asks the customer to confirm. The session continues until the customer explicitly confirms.
4. **Ticket Creation** — On confirmation, the ticket JSON and conversation history are sent to the UI via WebSocket and saved to `conversations/`.
5. **Shutdown** — The agent says goodbye, drains remaining audio, and signals the done event for graceful shutdown.

## Conversation Logs

Conversations are auto-saved to `conversations/` as JSON on ticket completion or Ctrl+C:

```json
{
  "timestamp": "2026-02-25T10:30:00",
  "room": "ai-room",
  "agent_id": "support-agent",
  "ticket": {
    "order_number": "ABC123456",
    "problem_category": "shipping",
    "problem_description": "Package has not arrived after 2 weeks",
    "urgency_level": "high",
    "resolved": true
  },
  "conversation_history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "summary": "Order #ABC123456 — shipping issue — urgency: high"
}
```

## Potential Improvements

### General Improvements
- no echo cancellations,  

### Latency Optimization

- **TTS warm-up**: Open the ElevenLabs WebSocket connection before the LLM starts generating, so the first token can be sent immediately.
- **Edge STT**: Deepgram's on-prem or edge deployment would eliminate the round-trip to their cloud for VAD and transcription events.

### Robustness

- **Reconnection logic**: If the Deepgram, OpenAI, or ElevenLabs WebSocket drops mid-conversation, the agent should reconnect and resume rather than crash.
- **Timeout handling**: If the user goes silent for an extended period, the agent could prompt them or gracefully end the session.

### Production Readiness

- **Authentication**: The WebSocket endpoint is currently open. Add token-based auth for both the browser client and agent.
- **Horizontal scaling**: There is a lot more work to be done for scaling this
- **Observability**: Add structured logging, metrics (turn latency, STT accuracy, barge-in rate), and tracing across the STT → LLM → TTS pipeline.
- **Conversation storage**: Replace local JSON files with a database for searchable, durable ticket storage.
