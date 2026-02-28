"""Tests for agent.tts — TTSClient with mocked ElevenLabs WebSocket."""
import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from agent.tts import TTSClient, TTS_SAMPLE_RATE


class MockElevenLabsWS:
    """Mock WebSocket that supports async iteration with a yield point."""
    def __init__(self, responses):
        self.send = AsyncMock()
        self._responses = responses

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        # Yield control to let the concurrent _send task execute first
        await asyncio.sleep(0.05)
        for item in self._responses:
            yield item


class TestTTSClient:
    def _make_mock_ws(self, audio_responses):
        """Create a mock websocket that yields audio_responses as async iterable."""
        return MockElevenLabsWS(audio_responses)

    @patch("websockets.connect")
    async def test_sends_voice_settings_on_connect(self, mock_connect):
        mock_ws = self._make_mock_ws([])
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock()
        mock_connect.return_value = mock_cm

        client = TTSClient(api_key="test-key")
        track = AsyncMock()
        track.push_pcm = AsyncMock()

        async def text_gen():
            yield "Hello"

        await client.stream_to_track(text_gen(), track)

        # First send should be the voice settings
        first_call = mock_ws.send.call_args_list[0]
        data = json.loads(first_call[0][0])
        assert "voice_settings" in data
        assert data["xi_api_key"] == "test-key"

    @patch("websockets.connect")
    async def test_streams_text_chunks(self, mock_connect):
        mock_ws = self._make_mock_ws([])
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock()
        mock_connect.return_value = mock_cm

        client = TTSClient(api_key="test-key")
        track = AsyncMock()
        track.push_pcm = AsyncMock()

        async def text_gen():
            yield "Hello "
            yield "world"

        await client.stream_to_track(text_gen(), track)

        # Calls should include: voice_settings, "Hello ", "world", "" (end marker)
        send_calls = mock_ws.send.call_args_list
        texts = [json.loads(c[0][0]).get("text") for c in send_calls]
        assert "Hello " in texts
        assert "world" in texts
        assert "" in texts  # end of stream

    @patch("websockets.connect")
    async def test_decodes_audio_and_pushes_to_track(self, mock_connect):
        pcm_data = b"\x00\x01" * 100
        audio_b64 = base64.b64encode(pcm_data).decode()
        audio_msg = json.dumps({"audio": audio_b64})

        mock_ws = self._make_mock_ws([audio_msg])
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock()
        mock_connect.return_value = mock_cm

        client = TTSClient(api_key="test-key")
        track = AsyncMock()
        track.push_pcm = AsyncMock()

        async def text_gen():
            yield "Hi"

        await client.stream_to_track(text_gen(), track)

        track.push_pcm.assert_awaited_once_with(pcm_data, sample_rate=TTS_SAMPLE_RATE)

    @patch("websockets.connect")
    async def test_calls_on_first_audio_callback(self, mock_connect):
        pcm_data = b"\x00\x01" * 50
        audio_b64 = base64.b64encode(pcm_data).decode()
        audio_msg1 = json.dumps({"audio": audio_b64})
        audio_msg2 = json.dumps({"audio": audio_b64})

        mock_ws = self._make_mock_ws([audio_msg1, audio_msg2])
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_cm.__aexit__ = AsyncMock()
        mock_connect.return_value = mock_cm

        client = TTSClient(api_key="test-key")
        track = AsyncMock()
        track.push_pcm = AsyncMock()

        on_first_audio = MagicMock()

        async def text_gen():
            yield "Hi"

        await client.stream_to_track(text_gen(), track, on_first_audio=on_first_audio)

        # Should be called exactly once (on first chunk only)
        on_first_audio.assert_called_once()
