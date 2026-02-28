"""Tests for agent.agent — SupportAgent orchestration."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from agent.audio import TTSAudioTrack


class TestSupportAgent:
    @pytest.fixture
    def mock_deps(self):
        """Set up mocked STT, LLM, TTS and TTSAudioTrack."""
        with patch("agent.agent.STTClient") as MockSTT, \
             patch("agent.agent.LLMClient") as MockLLM, \
             patch("agent.agent.TTSClient") as MockTTS:

            mock_stt = AsyncMock()
            mock_stt.start = AsyncMock()
            mock_stt.stop = AsyncMock()
            mock_stt.send_audio = AsyncMock()
            MockSTT.return_value = mock_stt

            mock_llm = MagicMock()
            mock_llm.history = []
            mock_llm.ticket = MagicMock()
            mock_llm.ticket.is_complete.return_value = False
            mock_llm.ticket.resolved = False
            mock_llm.ticket.to_dict.return_value = {}
            mock_llm.get_summary.return_value = "test summary"
            mock_llm.await_extraction = AsyncMock()

            async def mock_stream_chat(text):
                yield "Response"
            mock_llm.stream_chat = mock_stream_chat
            MockLLM.return_value = mock_llm

            mock_tts = AsyncMock()
            mock_tts.stream_to_track = AsyncMock()
            MockTTS.return_value = mock_tts

            mock_track = MagicMock(spec=TTSAudioTrack)
            mock_track.interrupt = MagicMock()
            mock_track.reset_interrupt = MagicMock()
            mock_track.clear_queue = MagicMock()
            mock_track.drain = AsyncMock()

            yield {
                "MockSTT": MockSTT, "stt": mock_stt,
                "MockLLM": MockLLM, "llm": mock_llm,
                "MockTTS": MockTTS, "tts": mock_tts,
                "track": mock_track,
            }

    def _make_agent(self, deps, ws_send=None):
        from agent.agent import SupportAgent
        agent = SupportAgent(
            client_id="agent-1",
            room="test-room",
            system_prompt="You are a support agent",
            tts_track=deps["track"],
            ws_send=ws_send,
            deepgram_api_key="test",
            openai_api_key="test",
            elevenlabs_api_key="test",
        )
        return agent

    async def test_start_starts_stt_and_sends_greeting(self, mock_deps):
        agent = self._make_agent(mock_deps)
        await agent.start()

        mock_deps["stt"].start.assert_awaited_once()
        assert agent._started is True
        # TTS should have been called for the greeting
        mock_deps["tts"].stream_to_track.assert_awaited_once()

    async def test_on_audio_frame_forwards_pcm(self, mock_deps):
        agent = self._make_agent(mock_deps)
        await agent.start()

        # Create a simple av.AudioFrame
        import numpy as np
        import av
        data = np.zeros((1, 1920), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(data, format="s16", layout="stereo")
        frame.sample_rate = 48000

        await agent.on_audio_frame(frame)
        mock_deps["stt"].send_audio.assert_awaited_once()

    async def test_on_audio_frame_noop_before_start(self, mock_deps):
        agent = self._make_agent(mock_deps)
        # Don't call start()

        import numpy as np
        import av
        data = np.zeros((1, 1920), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(data, format="s16", layout="stereo")
        frame.sample_rate = 48000

        await agent.on_audio_frame(frame)
        mock_deps["stt"].send_audio.assert_not_awaited()

    async def test_on_transcript_bargein_interrupts(self, mock_deps):
        agent = self._make_agent(mock_deps)
        await agent.start()
        agent._is_speaking = True

        await agent._on_transcript("stop talking", is_final=True)

        mock_deps["track"].interrupt.assert_called()
        assert "stop talking" in agent._transcript_buffer

    async def test_on_utterance_end_triggers_process_turn(self, mock_deps):
        agent = self._make_agent(mock_deps)
        await agent.start()
        agent._transcript_buffer = ["Hello", "I need help"]

        await agent._on_utterance_end()

        # Should have cleared buffer and created llm_task
        assert agent._transcript_buffer == []
        assert agent._llm_task is not None
        # Wait for task to complete
        await asyncio.sleep(0.1)

    async def test_process_turn_streams_llm_to_tts(self, mock_deps):
        agent = self._make_agent(mock_deps)
        await agent.start()

        # Reset the call count from greeting
        mock_deps["tts"].stream_to_track.reset_mock()

        await agent._process_turn("I need help with order 12345")

        mock_deps["track"].reset_interrupt.assert_called()
        mock_deps["tts"].stream_to_track.assert_awaited_once()
        mock_deps["llm"].await_extraction.assert_awaited()

    async def test_send_conversation_data(self, mock_deps):
        ws_send = AsyncMock()
        agent = self._make_agent(mock_deps, ws_send=ws_send)
        await agent.start()

        mock_deps["llm"].ticket.to_dict.return_value = {"order_number": "123"}
        mock_deps["llm"].get_summary.return_value = "Order #123"
        mock_deps["llm"].history = [{"role": "user", "content": "test"}]

        await agent._send_conversation_data()

        ws_send.assert_awaited_once()
        sent_data = ws_send.call_args[0][0]
        assert sent_data["type"] == "conversation_data"
        assert sent_data["ticket"]["order_number"] == "123"
        assert sent_data["summary"] == "Order #123"
