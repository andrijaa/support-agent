"""
SupportAgent: orchestrates STT -> LLM -> TTS pipeline.

- Receives audio via on_audio_frame() from WebRTC track -> Deepgram STT
- On speech_started (VAD): interrupts current agent speech (barge-in)
- On utterance_end: sends transcript to OpenAI LLM -> ElevenLabs TTS -> WebRTC
- On ticket complete: saves JSON conversation log and sends data via WebSocket
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Awaitable

import av

from agent.audio import av_frame_to_pcm_bytes, TTSAudioTrack
from agent.stt import STTClient
from agent.llm import LLMClient, SupportTicketData
from agent.tts import TTSClient

logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path(__file__).parent.parent / "conversations"


class SupportAgent:
    def __init__(
        self,
        client_id: str,
        room: str,
        system_prompt: str,
        tts_track: TTSAudioTrack,
        ws_send: Optional[Callable[[dict], Awaitable[None]]] = None,
        deepgram_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
    ):
        self.client_id = client_id
        self.room = room
        self._tts_track = tts_track
        self._ws_send = ws_send

        self._stt = STTClient(
            api_key=deepgram_api_key,
            on_transcript=self._on_transcript,
            on_utterance_end=self._on_utterance_end,
            on_speech_started=self._on_speech_started,
        )
        self._llm = LLMClient(api_key=openai_api_key, system_prompt=system_prompt)
        self._tts = TTSClient(api_key=elevenlabs_api_key)

        self._transcript_buffer: list[str] = []
        self._llm_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        self._started = False
        self._conversation_ended = False
        self._done_event = asyncio.Event()
        self._speech_start_time: float = 0.0  # when first audio chunk started playing

    async def start(self) -> None:
        """Start STT connection and send greeting."""
        await self._stt.start()
        self._started = True
        logger.info("SupportAgent started")

        await self._respond("Hello! Thank you for contacting customer support. "
                            "My name is Alex and I'm here to help you today. "
                            "Could you please start by giving me your order number?")

    async def on_audio_frame(self, frame: av.AudioFrame) -> None:
        """Forward incoming WebRTC audio to Deepgram STT."""
        if not self._started:
            return
        pcm_bytes = av_frame_to_pcm_bytes(frame)
        await self._stt.send_audio(pcm_bytes)

    async def _on_speech_started(self) -> None:
        """Deepgram VAD detected the user started speaking — barge-in."""
        import time
        if not self._is_speaking:
            return
        if time.monotonic() - self._speech_start_time < 0.5:
            return
        logger.info("Barge-in detected (VAD) — interrupting agent")
        self._interrupt()

    async def _on_transcript(self, text: str, is_final: bool) -> None:
        if is_final and text.strip():
            # If agent is speaking and we get a real transcript, treat as barge-in.
            # This catches cases where SpeechStarted fired during the grace period.
            if self._is_speaking:
                logger.info("Barge-in detected (transcript) — interrupting agent")
                self._interrupt()
            self._transcript_buffer.append(text.strip())
            logger.info("STT final: %s", text)

    async def _on_utterance_end(self) -> None:
        if not self._transcript_buffer:
            return
        if self._conversation_ended:
            return
        full_text = " ".join(self._transcript_buffer)
        self._transcript_buffer.clear()
        logger.info("Utterance ended: %s", full_text)

        self._interrupt()
        self._llm_task = asyncio.create_task(self._process_turn(full_text))

    def _interrupt(self) -> None:
        """Cancel in-progress LLM/TTS and clear audio queue."""
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
        self._tts_track.clear_queue()
        self._is_speaking = False

    def _on_first_audio(self) -> None:
        """Called when the first TTS audio chunk actually reaches the track."""
        import time
        self._is_speaking = True
        self._speech_start_time = time.monotonic()
        logger.debug("Agent speech started playing")

    async def _process_turn(self, user_text: str) -> None:
        """LLM -> TTS streaming pipeline for one conversation turn."""
        try:
            logger.info("Processing turn: %s", user_text)
            await self._tts.stream_to_track(
                self._llm.stream_chat(user_text),
                self._tts_track,
                on_first_audio=self._on_first_audio,
            )
            # Wait for extraction to finish before checking completeness
            await self._llm.await_extraction()
            if self._llm.ticket.is_complete() and not self._llm.ticket.resolved:
                self._llm.ticket.resolved = True
                await self._on_ticket_complete()
        except asyncio.CancelledError:
            logger.info("Turn processing cancelled (barge-in)")
        except Exception as e:
            logger.error("Error processing turn: %s", e)
        finally:
            self._is_speaking = False

    async def _respond(self, text: str) -> None:
        """Stream a static text response to the WebRTC track via ElevenLabs WebSocket."""
        async def _single_text():
            yield text

        try:
            await self._tts.stream_to_track(
                _single_text(),
                self._tts_track,
                on_first_audio=self._on_first_audio,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("TTS error: %s", e)
        finally:
            self._is_speaking = False

    async def _on_ticket_complete(self) -> None:
        print("\n[SUPPORT TICKET CREATED]")
        print(f"Order: {self._llm.ticket.order_number}")
        print(f"Category: {self._llm.ticket.problem_category}")
        print(f"Urgency: {self._llm.ticket.urgency_level}")
        print(f"Description: {self._llm.ticket.problem_description}")
        print()
        await self._send_conversation_data()
        await self.save_conversation()

        # Say goodbye, wait for audio to finish, then signal done
        self._conversation_ended = True
        await self._respond(
            "I've created a support ticket for your issue. "
            "Our team will look into it and get back to you as soon as possible. "
            "Thank you for contacting us, and have a great day! Goodbye."
        )
        await self._tts_track.drain()
        await asyncio.sleep(0.5)  # small buffer for last RTP packets
        logger.info("Conversation ended, signalling shutdown")
        self._done_event.set()

    async def _send_conversation_data(self) -> None:
        """Send ticket data to the UI via WebSocket."""
        if not self._ws_send:
            return
        data = {
            "type": "conversation_data",
            "ticket": self._llm.ticket.to_dict(),
            "summary": self._llm.get_summary(),
            "conversation_history": self._llm.history,
        }
        try:
            await self._ws_send(data)
            logger.info("Sent conversation_data to UI")
        except Exception as e:
            logger.warning("Failed to send conversation_data: %s", e)

    async def save_conversation(self) -> None:
        CONVERSATIONS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().isoformat()
        filename = f"{timestamp.replace(':', '-').replace('.', '-')}-{self.room}.json"
        data = {
            "timestamp": timestamp,
            "room": self.room,
            "agent_id": self.client_id,
            "ticket": self._llm.ticket.to_dict(),
            "conversation_history": self._llm.history,
            "summary": self._llm.get_summary(),
        }
        filepath = CONVERSATIONS_DIR / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Conversation saved to %s", filepath)

    async def stop(self) -> None:
        self._interrupt()
        await self._stt.stop()
        if self._llm.history:
            await self.save_conversation()
        logger.info("SupportAgent stopped")
