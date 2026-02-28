"""Integration tests — full STT→LLM→TTS pipeline with all services mocked."""
import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import av
import pytest

from agent.audio import TTSAudioTrack


class AsyncIterFromList:
    """Wrap a list into a proper async iterator."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _make_audio_frame():
    """Create a valid s16 stereo audio frame."""
    data = np.zeros((1, 1920), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(data, format="s16", layout="stereo")
    frame.sample_rate = 48000
    return frame


class TestMultiTurnConversation:
    """Simulate a multi-turn conversation ending in ticket completion."""

    @patch("websockets.connect")
    @patch("agent.agent.TTSClient")
    @patch("agent.llm.LLMClient._get_sync_client")
    @patch("openai.AsyncOpenAI")
    @patch("deepgram.AsyncDeepgramClient")
    async def test_multi_turn_ticket_extraction(
        self, MockDG, MockAsyncOAI, mock_sync_client, MockTTSClient, mock_ws_connect
    ):
        # --- STT mock: immediately ready, no real listening ---
        mock_connection = AsyncMock()
        mock_connection.on = MagicMock()
        mock_connection.start_listening = AsyncMock()
        mock_connection.send_media = AsyncMock()
        mock_connection.send_keep_alive = AsyncMock()
        mock_connection.send_close_stream = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_cm.__aexit__ = AsyncMock()

        mock_dg_instance = MagicMock()
        mock_dg_instance.listen.v1.connect = MagicMock(return_value=mock_cm)
        MockDG.return_value = mock_dg_instance

        # --- LLM mock: streaming response ---
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1

            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = f"Response {call_count}"

            return AsyncIterFromList([chunk])

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = mock_create
        MockAsyncOAI.return_value = mock_async_client

        # --- LLM extraction mock ---
        extraction_response = json.dumps({
            "order_number": "ORD999",
            "problem_category": "shipping",
            "problem_description": "My package was delivered to the wrong address and I need it redirected",
            "urgency_level": "high",
        })
        mock_sync = MagicMock()
        mock_sync.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=extraction_response))]
        )
        mock_sync_client.return_value = mock_sync

        # --- TTS mock: consume the generator so stream_chat runs to completion ---
        async def mock_stream_to_track(text_gen, track, on_first_audio=None):
            async for _ in text_gen:
                pass

        mock_tts_instance = MagicMock()
        mock_tts_instance.stream_to_track = mock_stream_to_track
        MockTTSClient.return_value = mock_tts_instance

        # --- Build agent ---
        from agent.agent import SupportAgent

        mock_track = MagicMock(spec=TTSAudioTrack)
        mock_track.interrupt = MagicMock()
        mock_track.reset_interrupt = MagicMock()
        mock_track.clear_queue = MagicMock()
        mock_track.drain = AsyncMock()

        ws_send = AsyncMock()

        agent = SupportAgent(
            client_id="agent-1",
            room="integration-test",
            system_prompt="You are a support agent",
            tts_track=mock_track,
            ws_send=ws_send,
            deepgram_api_key="test",
            openai_api_key="test",
            elevenlabs_api_key="test",
        )

        await agent.start()

        # Simulate user turn
        await agent._on_transcript("My order ORD999 was delivered to wrong address", is_final=True)
        await agent._on_utterance_end()

        # Wait for processing
        if agent._llm_task:
            await agent._llm_task

        # Verify extraction was triggered
        assert agent._llm.ticket.order_number == "ORD999"
        assert agent._llm.ticket.problem_category == "shipping"


class TestBargeIn:
    """Test that user interruption cancels in-progress speech."""

    @patch("websockets.connect")
    @patch("agent.agent.TTSClient")
    @patch("openai.AsyncOpenAI")
    @patch("deepgram.AsyncDeepgramClient")
    async def test_bargein_interrupts_speech(
        self, MockDG, MockAsyncOAI, MockTTSClient, mock_ws_connect
    ):
        # STT mock
        mock_connection = AsyncMock()
        mock_connection.on = MagicMock()
        mock_connection.start_listening = AsyncMock()
        mock_connection.send_media = AsyncMock()
        mock_connection.send_keep_alive = AsyncMock()
        mock_connection.send_close_stream = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_cm.__aexit__ = AsyncMock()

        mock_dg_instance = MagicMock()
        mock_dg_instance.listen.v1.connect = MagicMock(return_value=mock_cm)
        MockDG.return_value = mock_dg_instance

        # LLM mock — slow streaming to allow interruption
        async def slow_create(**kwargs):
            async def slow_iter():
                for i in range(10):
                    await asyncio.sleep(0.05)
                    chunk = MagicMock()
                    chunk.choices = [MagicMock()]
                    chunk.choices[0].delta.content = f"word{i} "
                    yield chunk

            mock_resp = MagicMock()
            mock_resp.__aiter__ = lambda self: slow_iter()
            return mock_resp

        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = slow_create
        MockAsyncOAI.return_value = mock_async_client

        # TTS mock — actually consume the text generator to simulate streaming
        async def mock_stream_to_track(text_gen, track, on_first_audio=None):
            if on_first_audio:
                on_first_audio()
            async for chunk in text_gen:
                await asyncio.sleep(0.01)

        mock_tts_instance = AsyncMock()
        mock_tts_instance.stream_to_track = mock_stream_to_track
        MockTTSClient.return_value = mock_tts_instance

        mock_track = MagicMock(spec=TTSAudioTrack)
        mock_track.interrupt = MagicMock()
        mock_track.reset_interrupt = MagicMock()
        mock_track.clear_queue = MagicMock()
        mock_track.drain = AsyncMock()

        from agent.agent import SupportAgent

        agent = SupportAgent(
            client_id="agent-1",
            room="test",
            system_prompt="test",
            tts_track=mock_track,
            deepgram_api_key="test",
            openai_api_key="test",
            elevenlabs_api_key="test",
        )

        await agent.start()

        # Start first turn
        agent._transcript_buffer = ["Tell me about shipping"]
        await agent._on_utterance_end()

        # Let it start processing
        await asyncio.sleep(0.1)
        assert agent._is_speaking or agent._llm_task is not None

        # Simulate barge-in while agent is speaking
        agent._is_speaking = True
        await agent._on_transcript("actually never mind", is_final=True)

        # interrupt should have been called
        mock_track.interrupt.assert_called()


class TestSaveConversation:
    """Test that conversation is saved to JSON file."""

    @patch("websockets.connect")
    @patch("agent.agent.TTSClient")
    @patch("openai.AsyncOpenAI")
    @patch("deepgram.AsyncDeepgramClient")
    async def test_save_conversation_writes_json(
        self, MockDG, MockAsyncOAI, MockTTSClient, mock_ws_connect
    ):
        # Minimal mocks
        mock_connection = AsyncMock()
        mock_connection.on = MagicMock()
        mock_connection.start_listening = AsyncMock()
        mock_connection.send_keep_alive = AsyncMock()
        mock_connection.send_close_stream = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_cm.__aexit__ = AsyncMock()

        mock_dg_instance = MagicMock()
        mock_dg_instance.listen.v1.connect = MagicMock(return_value=mock_cm)
        MockDG.return_value = mock_dg_instance

        mock_tts_instance = AsyncMock()
        mock_tts_instance.stream_to_track = AsyncMock()
        MockTTSClient.return_value = mock_tts_instance

        mock_track = MagicMock(spec=TTSAudioTrack)
        mock_track.interrupt = MagicMock()
        mock_track.reset_interrupt = MagicMock()
        mock_track.clear_queue = MagicMock()
        mock_track.drain = AsyncMock()

        from agent.agent import SupportAgent, CONVERSATIONS_DIR

        agent = SupportAgent(
            client_id="agent-1",
            room="save-test",
            system_prompt="test",
            tts_track=mock_track,
            deepgram_api_key="test",
            openai_api_key="test",
            elevenlabs_api_key="test",
        )

        # Manually set some history
        agent._llm.history = [
            {"role": "user", "content": "I have a problem"},
            {"role": "assistant", "content": "How can I help?"},
        ]
        agent._llm.ticket.order_number = "ORD123"
        agent._llm.ticket.problem_category = "billing"

        # Use a temp dir to avoid polluting the real conversations dir
        with tempfile.TemporaryDirectory() as tmpdir:
            import agent.agent as agent_module
            original_dir = agent_module.CONVERSATIONS_DIR
            agent_module.CONVERSATIONS_DIR = Path(tmpdir)
            try:
                await agent.save_conversation()

                files = list(Path(tmpdir).glob("*.json"))
                assert len(files) == 1

                with open(files[0]) as f:
                    data = json.load(f)
                assert data["room"] == "save-test"
                assert data["agent_id"] == "agent-1"
                assert len(data["conversation_history"]) == 2
            finally:
                agent_module.CONVERSATIONS_DIR = original_dir
