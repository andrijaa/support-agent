"""Deepgram streaming STT client — deepgram-sdk v6 API."""
from __future__ import annotations
import asyncio
import logging
import os
from typing import Callable, Optional, Awaitable

logger = logging.getLogger(__name__)

STT_SAMPLE_RATE = 48000
STT_CHANNELS = 2
STT_ENCODING = "linear16"
STT_MODEL = "nova-2"
UTTERANCE_END_MS = "1000"
KEEPALIVE_INTERVAL = 8  # seconds — Deepgram times out after ~10s of silence


class STTClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        on_transcript: Optional[Callable[[str, bool], Awaitable[None]]] = None,
        on_utterance_end: Optional[Callable[[], Awaitable[None]]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.on_transcript = on_transcript
        self.on_utterance_end = on_utterance_end
        self.on_speech_started = on_speech_started

        self._connection = None
        self._listen_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._running = False
        self._cm = None

    async def start(self) -> None:
        """Open streaming connection to Deepgram and start listen loop."""
        from deepgram import AsyncDeepgramClient
        from deepgram.listen import ListenV1Results, ListenV1UtteranceEnd, ListenV1SpeechStarted
        from deepgram.core.events import EventType

        dg = AsyncDeepgramClient(api_key=self._api_key)

        self._cm = dg.listen.v1.connect(
            model=STT_MODEL,
            encoding=STT_ENCODING,
            sample_rate=str(STT_SAMPLE_RATE),
            channels=str(STT_CHANNELS),
            interim_results="true",
            utterance_end_ms=UTTERANCE_END_MS,
            vad_events="true",
            smart_format="true",
            language="en-US",
        )
        self._connection = await self._cm.__aenter__()
        self._running = True

        async def _on_message(msg):
            if isinstance(msg, ListenV1Results):
                try:
                    transcript = msg.channel.alternatives[0].transcript
                    is_final = msg.is_final
                    if transcript and self.on_transcript:
                        await self.on_transcript(transcript, is_final)
                except Exception as e:
                    logger.warning("STT message parse error: %s", e)
            elif isinstance(msg, ListenV1UtteranceEnd):
                if self.on_utterance_end:
                    await self.on_utterance_end()
            elif isinstance(msg, ListenV1SpeechStarted):
                if self.on_speech_started:
                    await self.on_speech_started()

        async def _on_error(error):
            logger.error("Deepgram error: %s", error)

        self._connection.on(EventType.MESSAGE, _on_message)
        self._connection.on(EventType.ERROR, _on_error)

        self._listen_task = asyncio.create_task(self._connection.start_listening())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info("Deepgram STT connection started (v6 API)")

    async def _keepalive_loop(self) -> None:
        """Send KeepAlive every few seconds to prevent Deepgram timeout on silence."""
        try:
            while self._running:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                if self._connection and self._running:
                    await self._connection.send_keep_alive()
        except asyncio.CancelledError:
            pass

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send raw PCM audio bytes to Deepgram."""
        if self._connection and self._running and pcm_bytes:
            await self._connection.send_media(pcm_bytes)

    async def stop(self) -> None:
        """Close Deepgram connection."""
        self._running = False
        for task in (self._keepalive_task, self._listen_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._connection:
            try:
                await self._connection.send_close_stream()
            except Exception:
                pass
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._connection = None
        logger.info("Deepgram STT connection closed")
