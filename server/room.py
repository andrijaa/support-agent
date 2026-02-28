from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.peer import Peer

logger = logging.getLogger(__name__)


class Room:
    def __init__(self, name: str):
        self.name = name
        self.peers: dict[str, "Peer"] = {}
        self._lock = asyncio.Lock()

    async def add(self, peer: "Peer") -> None:
        async with self._lock:
            self.peers[peer.client_id] = peer
        logger.info("Room %s: peer %s joined (total=%d)", self.name, peer.client_id, len(self.peers))

    async def remove(self, client_id: str) -> None:
        async with self._lock:
            self.peers.pop(client_id, None)
        logger.info("Room %s: peer %s left (total=%d)", self.name, client_id, len(self.peers))

    def get_others(self, client_id: str) -> list["Peer"]:
        return [p for pid, p in self.peers.items() if pid != client_id]

    async def broadcast(self, message: dict, exclude_id: str | None = None) -> None:
        for peer in self.peers.values():
            if exclude_id and peer.client_id == exclude_id:
                continue
            try:
                await peer.ws.send_json(message)
            except Exception as e:
                logger.warning("Broadcast to %s failed: %s", peer.client_id, e)


class RoomManager:
    def __init__(self):
        self._rooms: dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, name: str) -> Room:
        async with self._lock:
            if name not in self._rooms:
                self._rooms[name] = Room(name)
            return self._rooms[name]

    async def cleanup(self, name: str) -> None:
        async with self._lock:
            room = self._rooms.get(name)
            if room and not room.peers:
                del self._rooms[name]
                logger.info("Room %s removed (empty)", name)
