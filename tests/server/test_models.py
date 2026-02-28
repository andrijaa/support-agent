"""Tests for server.models.SignalMessage."""
import pytest
from pydantic import ValidationError

from server.models import SignalMessage


class TestSignalMessage:
    def test_valid_join_message_minimal(self):
        msg = SignalMessage(type="join")
        assert msg.type == "join"
        assert msg.room is None
        assert msg.client_id is None

    def test_full_message_all_fields(self):
        msg = SignalMessage(
            type="join",
            room="test-room",
            client_id="agent-1",
            sdp="v=0\r\n...",
            candidate="candidate:1 1 udp 2130706431 ...",
            sdp_mid="audio",
            sdp_mline_index=0,
        )
        assert msg.type == "join"
        assert msg.room == "test-room"
        assert msg.client_id == "agent-1"
        assert msg.sdp == "v=0\r\n..."
        assert msg.candidate == "candidate:1 1 udp 2130706431 ..."
        assert msg.sdp_mid == "audio"
        assert msg.sdp_mline_index == 0

    def test_ice_candidate_message(self):
        msg = SignalMessage(
            type="candidate",
            candidate="candidate:1 1 udp 2130706431 192.168.1.1 5000 typ host",
            sdp_mid="0",
            sdp_mline_index=0,
        )
        assert msg.type == "candidate"
        assert msg.sdp_mid == "0"
        assert msg.sdp_mline_index == 0

    def test_answer_message_sdp_roundtrip(self):
        sdp_text = "v=0\r\no=- 123 1 IN IP4 0.0.0.0\r\ns=-\r\n"
        msg = SignalMessage(type="answer", sdp=sdp_text)
        assert msg.sdp == sdp_text
        assert msg.type == "answer"

    def test_missing_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SignalMessage()  # type is required
