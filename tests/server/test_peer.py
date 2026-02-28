"""Tests for server.peer — Peer class and renegotiate helper."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from server.room import Room


class TestPeer:
    @pytest.fixture
    def mock_pc(self):
        pc = MagicMock()
        pc.close = AsyncMock()
        pc.createOffer = AsyncMock(return_value=MagicMock(sdp="offer-sdp", type="offer"))
        pc.setLocalDescription = AsyncMock()
        pc.addTrack = MagicMock()
        pc.addTransceiver = MagicMock()
        pc.connectionState = "connected"
        pc.localDescription = MagicMock(sdp="offer-sdp")
        pc.on = MagicMock()
        return pc

    @pytest.fixture
    def mock_ws(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.fixture
    def room(self):
        return Room("test-room")

    @patch("server.peer.RTCPeerConnection")
    def test_init_sets_up_fields(self, MockPC, mock_ws, room):
        mock_instance = MagicMock()
        mock_instance.on = MagicMock()
        MockPC.return_value = mock_instance

        from server.peer import Peer
        peer = Peer(client_id="agent-1", room=room, ws=mock_ws)

        assert peer.client_id == "agent-1"
        assert peer.room is room
        assert peer.ws is mock_ws
        assert peer.relay is None

    @patch("server.peer.RTCPeerConnection")
    async def test_subscribe_to_with_relay(self, MockPC, mock_ws, room):
        mock_instance = MagicMock()
        mock_instance.on = MagicMock()
        mock_instance.addTrack = MagicMock()
        MockPC.return_value = mock_instance

        from server.peer import Peer
        peer = Peer(client_id="p1", room=room, ws=mock_ws)

        source = MagicMock()
        source.relay = MagicMock()
        source._relay_track = MagicMock()
        source.relay.subscribe = MagicMock(return_value=MagicMock())

        await peer.subscribe_to(source)
        source.relay.subscribe.assert_called_once_with(source._relay_track)
        mock_instance.addTrack.assert_called_once()

    @patch("server.peer.RTCPeerConnection")
    async def test_subscribe_to_no_relay_is_noop(self, MockPC, mock_ws, room):
        mock_instance = MagicMock()
        mock_instance.on = MagicMock()
        mock_instance.addTrack = MagicMock()
        MockPC.return_value = mock_instance

        from server.peer import Peer
        peer = Peer(client_id="p1", room=room, ws=mock_ws)

        source = MagicMock()
        source.relay = None
        source._relay_track = None

        await peer.subscribe_to(source)
        mock_instance.addTrack.assert_not_called()

    @patch("server.peer.RTCPeerConnection")
    async def test_close_calls_pc_close(self, MockPC, mock_ws, room):
        mock_instance = MagicMock()
        mock_instance.on = MagicMock()
        mock_instance.close = AsyncMock()
        MockPC.return_value = mock_instance

        from server.peer import Peer
        peer = Peer(client_id="p1", room=room, ws=mock_ws)
        await peer.close()
        mock_instance.close.assert_awaited_once()


class TestRenegotiate:
    @patch("server.peer.RTCPeerConnection")
    async def test_renegotiate_sends_offer(self, MockPC):
        mock_instance = MagicMock()
        mock_instance.on = MagicMock()
        mock_instance.connectionState = "connected"
        mock_instance.createOffer = AsyncMock(return_value=MagicMock(sdp="v=0", type="offer"))
        mock_instance.setLocalDescription = AsyncMock()
        mock_instance.localDescription = MagicMock(sdp="v=0")
        MockPC.return_value = mock_instance

        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        from server.peer import Peer, renegotiate
        room = Room("test")
        peer = Peer(client_id="p1", room=room, ws=mock_ws)
        await renegotiate(peer)

        mock_ws.send_json.assert_awaited_with({"type": "offer", "sdp": "v=0"})

    @patch("server.peer.RTCPeerConnection")
    async def test_renegotiate_skips_closed(self, MockPC):
        mock_instance = MagicMock()
        mock_instance.on = MagicMock()
        mock_instance.connectionState = "closed"
        mock_instance.createOffer = AsyncMock()
        MockPC.return_value = mock_instance

        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        from server.peer import Peer, renegotiate
        room = Room("test")
        peer = Peer(client_id="p1", room=room, ws=mock_ws)
        await renegotiate(peer)

        mock_instance.createOffer.assert_not_awaited()
        mock_ws.send_json.assert_not_awaited()
