"""ElevenLabs TTS client — real-time WebSocket streaming."""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import os
from typing import AsyncGenerator, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.audio import TTSAudioTrack

logger = logging.getLogger(__name__)

ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_MODEL = "eleven_turbo_v2_5"
TTS_SAMPLE_RATE = 22050
_WS_URL = (
    f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input"
    f"?model_id={ELEVENLABS_MODEL}&output_format=pcm_22050&optimize_streaming_latency=3"
)


class TTSClient:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")

    async def stream_to_track(
        self,
        text_gen: AsyncGenerator[str, None],
        track: "TTSAudioTrack",
        on_first_audio: Optional[Callable] = None,
    ) -> None:
        """Open ElevenLabs WebSocket, stream text from text_gen, push audio chunks to track.

        on_first_audio is called (once) when the first audio chunk arrives.
        """
        import websockets

        async with websockets.connect(_WS_URL) as ws:
            # Begin-of-stream with voice settings and API key
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
                "xi_api_key": self._api_key,
            }))

            async def _send() -> None:
                async for chunk in text_gen:
                    if chunk:
                        await ws.send(json.dumps({"text": chunk}))
                # End-of-stream triggers final audio flush
                await ws.send(json.dumps({"text": ""}))

            send_task = asyncio.create_task(_send())
            first_chunk = True
            chunk_count = 0
            total_bytes = 0
            try:
                async for raw in ws:
                    data = json.loads(raw)
                    if data.get("audio"):
                        if first_chunk:
                            first_chunk = False
                            logger.info("TTS: first audio chunk received from ElevenLabs")
                            if on_first_audio:
                                on_first_audio()
                        pcm = base64.b64decode(data["audio"])
                        chunk_count += 1
                        total_bytes += len(pcm)
                        await track.push_pcm(pcm, sample_rate=TTS_SAMPLE_RATE)
                    elif data.get("error"):
                        logger.error("ElevenLabs error: %s", data["error"])
            finally:
                logger.info("TTS: stream ended, %d chunks, %d bytes total", chunk_count, total_bytes)
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
