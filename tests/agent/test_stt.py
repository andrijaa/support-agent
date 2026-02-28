"""Tests for agent.stt — STTClient with mocked Deepgram SDK."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSTTClient:
    @patch("deepgram.AsyncDeepgramClient")
    async def test_start_opens_connection(self, MockDG):
        mock_connection = AsyncMock()
        mock_connection.on = MagicMock()
        mock_connection.start_listening = AsyncMock()
        mock_connection.send_keep_alive = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_cm.__aexit__ = AsyncMock()

        mock_dg_instance = MagicMock()
        mock_dg_instance.listen.v1.connect = MagicMock(return_value=mock_cm)
        MockDG.return_value = mock_dg_instance

        from agent.stt import STTClient
        stt = STTClient(api_key="test-key")
        await stt.start()

        assert stt._running is True
        assert stt._connection is mock_connection
        mock_connection.on.assert_called()

        # Cleanup
        await stt.stop()

    @patch("deepgram.AsyncDeepgramClient")
    async def test_send_audio_forwards_bytes(self, MockDG):
        mock_connection = AsyncMock()
        mock_connection.on = MagicMock()
        mock_connection.start_listening = AsyncMock()
        mock_connection.send_media = AsyncMock()
        mock_connection.send_keep_alive = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_cm.__aexit__ = AsyncMock()

        mock_dg_instance = MagicMock()
        mock_dg_instance.listen.v1.connect = MagicMock(return_value=mock_cm)
        MockDG.return_value = mock_dg_instance

        from agent.stt import STTClient
        stt = STTClient(api_key="test-key")
        await stt.start()

        audio_data = b"\x00" * 3200
        await stt.send_audio(audio_data)
        mock_connection.send_media.assert_awaited_once_with(audio_data)

        await stt.stop()

    async def test_send_audio_noop_when_not_running(self):
        from agent.stt import STTClient
        stt = STTClient(api_key="test-key")
        # Not started → send_audio should be no-op
        await stt.send_audio(b"\x00" * 100)
        # No error should occur

    @patch("deepgram.AsyncDeepgramClient")
    async def test_stop_cancels_tasks(self, MockDG):
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

        from agent.stt import STTClient
        stt = STTClient(api_key="test-key")
        await stt.start()

        assert stt._running is True
        await stt.stop()

        assert stt._running is False
        assert stt._connection is None
