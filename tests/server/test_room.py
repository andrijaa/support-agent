"""Tests for server.room — Room and RoomManager."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from server.room import Room, RoomManager


class TestRoom:
    @pytest.fixture
    def room(self):
        return Room("test-room")

    @pytest.fixture
    def make_peer(self):
        def _make(client_id: str):
            peer = MagicMock()
            peer.client_id = client_id
            peer.ws = MagicMock()
            peer.ws.send_json = AsyncMock()
            return peer
        return _make

    async def test_add_inserts_peer(self, room, make_peer):
        peer = make_peer("p1")
        await room.add(peer)
        assert "p1" in room.peers
        assert room.peers["p1"] is peer

    async def test_remove_deletes_peer(self, room, make_peer):
        peer = make_peer("p1")
        await room.add(peer)
        await room.remove("p1")
        assert "p1" not in room.peers

    async def test_remove_nonexistent_is_noop(self, room):
        await room.remove("nonexistent")  # should not raise
        assert len(room.peers) == 0

    async def test_get_others_excludes_self(self, room, make_peer):
        p1 = make_peer("p1")
        p2 = make_peer("p2")
        p3 = make_peer("p3")
        await room.add(p1)
        await room.add(p2)
        await room.add(p3)
        others = room.get_others("p1")
        ids = [p.client_id for p in others]
        assert "p1" not in ids
        assert "p2" in ids
        assert "p3" in ids

    async def test_get_others_empty_for_solo(self, room, make_peer):
        p1 = make_peer("p1")
        await room.add(p1)
        assert room.get_others("p1") == []

    async def test_broadcast_sends_to_all(self, room, make_peer):
        p1 = make_peer("p1")
        p2 = make_peer("p2")
        await room.add(p1)
        await room.add(p2)
        msg = {"type": "test"}
        await room.broadcast(msg)
        p1.ws.send_json.assert_awaited_with(msg)
        p2.ws.send_json.assert_awaited_with(msg)

    async def test_broadcast_with_exclude(self, room, make_peer):
        p1 = make_peer("p1")
        p2 = make_peer("p2")
        await room.add(p1)
        await room.add(p2)
        msg = {"type": "test"}
        await room.broadcast(msg, exclude_id="p1")
        p1.ws.send_json.assert_not_awaited()
        p2.ws.send_json.assert_awaited_with(msg)

    async def test_broadcast_handles_send_failure(self, room, make_peer):
        p1 = make_peer("p1")
        p2 = make_peer("p2")
        p1.ws.send_json = AsyncMock(side_effect=Exception("connection lost"))
        await room.add(p1)
        await room.add(p2)
        msg = {"type": "test"}
        await room.broadcast(msg)  # should not raise
        p2.ws.send_json.assert_awaited_with(msg)


class TestRoomManager:
    @pytest.fixture
    def manager(self):
        return RoomManager()

    async def test_get_or_create_creates_new(self, manager):
        room = await manager.get_or_create("room-1")
        assert room.name == "room-1"
        assert isinstance(room, Room)

    async def test_get_or_create_returns_same_instance(self, manager):
        room1 = await manager.get_or_create("room-1")
        room2 = await manager.get_or_create("room-1")
        assert room1 is room2

    async def test_cleanup_removes_empty_room(self, manager):
        await manager.get_or_create("room-1")
        await manager.cleanup("room-1")
        # Next call should create a fresh room
        room = await manager.get_or_create("room-1")
        assert len(room.peers) == 0

    async def test_cleanup_keeps_nonempty_room(self, manager, mock_peer):
        room = await manager.get_or_create("room-1")
        peer = mock_peer("p1")
        await room.add(peer)
        await manager.cleanup("room-1")
        # Room should still exist with the peer
        room2 = await manager.get_or_create("room-1")
        assert room2 is room
        assert "p1" in room2.peers

    async def test_cleanup_nonexistent_is_noop(self, manager):
        await manager.cleanup("nonexistent")  # should not raise
