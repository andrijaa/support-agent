# Python Conversational Customer Support Agent

## Architecture

```
Browser (React UI)  ←WebRTC/WS→    Python SFU Server (FastAPI + aiortc)
                                        ↕ WebRTC
                                   Python Support Agent
                                        ↓ PCM audio
                                   Deepgram STT
                                        ↓ transcript
                                   OpenAI 
                                        ↓ response text
                                   ElevenLabs TTS
                                        ↓ PCM audio
                                   WebRTC → Browser
```

## Setup

```bash
cd conversational_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

## Running

**Terminal 1 — SFU Server:**
```bash
python -m server.main
# Runs on ws://localhost:8080/ws
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
# Room: ai-room
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPGRAM_API_KEY` | Deepgram API key for STT |
| `OPENAI_API_KEY` | OpenAI API key for LLM |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for TTS |

## Project Structure

```
conversational_agent/
├── server/
│   ├── main.py        # FastAPI app, /ws WebSocket, /health
│   ├── models.py      # SignalMessage Pydantic model
│   ├── room.py        # Room + RoomManager
│   └── peer.py        # Peer (WebRTC + relay)
├── agent/
│   ├── main.py        # CLI entrypoint
│   ├── agent.py       # SupportAgent orchestrator
│   ├── client.py      # WebRTC WebSocket client
│   ├── stt.py         # Deepgram streaming STT
│   ├── llm.py         # OpenAI LLM client + SupportTicketData
│   ├── tts.py         # ElevenLabs TTS client
│   └── audio.py       # TTSAudioTrack + PCM utilities
├── config/
│   └── support_agent.json   # System prompt + persona
├── conversations/           # Saved conversation JSON logs
├── requirements.txt
└── .env.example
```

## Audio Pipeline

**Inbound (browser → agent):**
```
Browser mic → Opus (WebRTC) → aiortc decode → av.AudioFrame (float32 48kHz stereo)
→ interleave → int16 PCM → Deepgram STT
```

**Outbound (agent → browser):**
```
ElevenLabs → PCM (int16 22050Hz mono)
→ resample to 48kHz → stereo → av.AudioFrame chunks
→ TTSAudioTrack → aiortc Opus → WebRTC → browser speaker
```

## Conversation Logs

When a support ticket is complete, or on Ctrl+C, the conversation is saved to `conversations/`:

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
  "conversation_history": [...],
  "summary": "Order #ABC123456 — shipping issue — urgency: high"
}
```
