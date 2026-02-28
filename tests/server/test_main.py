"""Tests for server.main — FastAPI WebSocket endpoint."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from server.main import app


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestWebSocketEndpoint:
    @patch("server.main.Peer")
    @patch("server.main.room_manager")
    def test_join_creates_room_and_sends_offer(self, mock_rm, MockPeer):
        mock_room = MagicMock()
        mock_room.get_others = MagicMock(return_value=[])
        mock_room.add = AsyncMock()
        mock_room.broadcast = AsyncMock()
        mock_room.remove = AsyncMock()
        mock_room.peers = {}
        mock_rm.get_or_create = AsyncMock(return_value=mock_room)
        mock_rm.cleanup = AsyncMock()

        mock_pc = MagicMock()
        mock_pc.addTransceiver = MagicMock()
        mock_pc.createOffer = AsyncMock(return_value=MagicMock(sdp="test-sdp", type="offer"))
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.localDescription = MagicMock(sdp="test-sdp")
        mock_pc.close = AsyncMock()

        mock_peer_instance = MagicMock()
        mock_peer_instance.client_id = "browser-1"
        mock_peer_instance.pc = mock_pc
        mock_peer_instance.close = AsyncMock()
        MockPeer.return_value = mock_peer_instance

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "join", "room": "r1", "client_id": "browser-1"}))
            resp = ws.receive_json()
            assert resp["type"] == "offer"
            assert resp["sdp"] == "test-sdp"

    def test_invalid_json_is_ignored(self):
        """Server should not crash on invalid JSON."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not valid json {{{")
            # Connection should stay open — send a valid close-inducing action
            # Just verify no crash by closing gracefully
            ws.close()

    @patch("server.main.Peer")
    @patch("server.main.room_manager")
    def test_conversation_data_broadcast(self, mock_rm, MockPeer):
        mock_room = MagicMock()
        mock_room.get_others = MagicMock(return_value=[])
        mock_room.add = AsyncMock()
        mock_room.broadcast = AsyncMock()
        mock_room.remove = AsyncMock()
        mock_room.peers = {}
        mock_rm.get_or_create = AsyncMock(return_value=mock_room)
        mock_rm.cleanup = AsyncMock()

        mock_pc = MagicMock()
        mock_pc.addTransceiver = MagicMock()
        mock_pc.createOffer = AsyncMock(return_value=MagicMock(sdp="sdp", type="offer"))
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.localDescription = MagicMock(sdp="sdp")
        mock_pc.close = AsyncMock()

        mock_peer_instance = MagicMock()
        mock_peer_instance.client_id = "agent-1"
        mock_peer_instance.pc = mock_pc
        mock_peer_instance.close = AsyncMock()
        MockPeer.return_value = mock_peer_instance

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # First join
            ws.send_text(json.dumps({"type": "join", "room": "r1", "client_id": "agent-1"}))
            ws.receive_json()  # offer

            # Send conversation_data
            conv_data = {"type": "conversation_data", "ticket": {"order": "123"}}
            ws.send_text(json.dumps(conv_data))

            # Verify broadcast was called with conversation_data (excluding sender)
            # Need to give the async event loop a moment
            import time
            time.sleep(0.1)

        # After disconnect, broadcast should have been called for conversation_data
        calls = mock_room.broadcast.call_args_list
        conv_calls = [c for c in calls if c[0][0].get("type") == "conversation_data"]
        assert len(conv_calls) >= 1
        assert conv_calls[0][1]["exclude_id"] == "agent-1"

    @patch("server.main.Peer")
    @patch("server.main.room_manager")
    def test_answer_sets_remote_description(self, mock_rm, MockPeer):
        mock_room = MagicMock()
        mock_room.get_others = MagicMock(return_value=[])
        mock_room.add = AsyncMock()
        mock_room.broadcast = AsyncMock()
        mock_room.remove = AsyncMock()
        mock_room.peers = {}
        mock_rm.get_or_create = AsyncMock(return_value=mock_room)
        mock_rm.cleanup = AsyncMock()

        mock_pc = MagicMock()
        mock_pc.addTransceiver = MagicMock()
        mock_pc.createOffer = AsyncMock(return_value=MagicMock(sdp="sdp", type="offer"))
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.setRemoteDescription = AsyncMock()
        mock_pc.localDescription = MagicMock(sdp="sdp")
        mock_pc.close = AsyncMock()

        mock_peer_instance = MagicMock()
        mock_peer_instance.client_id = "b1"
        mock_peer_instance.pc = mock_pc
        mock_peer_instance.close = AsyncMock()
        MockPeer.return_value = mock_peer_instance

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "join", "room": "r1", "client_id": "b1"}))
            ws.receive_json()  # offer

            ws.send_text(json.dumps({"type": "answer", "sdp": "answer-sdp"}))
            import time
            time.sleep(0.1)

        mock_pc.setRemoteDescription.assert_awaited_once()

    @patch("server.main.Peer")
    @patch("server.main.room_manager")
    def test_disconnect_triggers_cleanup(self, mock_rm, MockPeer):
        mock_room = MagicMock()
        mock_room.get_others = MagicMock(return_value=[])
        mock_room.add = AsyncMock()
        mock_room.broadcast = AsyncMock()
        mock_room.remove = AsyncMock()
        mock_room.peers = {}
        mock_rm.get_or_create = AsyncMock(return_value=mock_room)
        mock_rm.cleanup = AsyncMock()

        mock_pc = MagicMock()
        mock_pc.addTransceiver = MagicMock()
        mock_pc.createOffer = AsyncMock(return_value=MagicMock(sdp="sdp", type="offer"))
        mock_pc.setLocalDescription = AsyncMock()
        mock_pc.localDescription = MagicMock(sdp="sdp")
        mock_pc.close = AsyncMock()

        mock_peer_instance = MagicMock()
        mock_peer_instance.client_id = "b1"
        mock_peer_instance.pc = mock_pc
        mock_peer_instance.close = AsyncMock()
        MockPeer.return_value = mock_peer_instance

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "join", "room": "r1", "client_id": "b1"}))
            ws.receive_json()  # offer

        # After disconnect
        mock_room.remove.assert_awaited_with("b1")
        mock_peer_instance.close.assert_awaited_once()
        mock_rm.cleanup.assert_awaited_with("r1")

        # Should have broadcast peer_left
        broadcast_calls = mock_room.broadcast.call_args_list
        peer_left_calls = [c for c in broadcast_calls if c[0][0].get("type") == "peer_left"]
        assert len(peer_left_calls) >= 1
