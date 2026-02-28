from __future__ import annotations
import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from aiortc import RTCPeerConnection
from aiortc.contrib.media import MediaRelay
from aiortc.sdp import candidate_to_sdp
from fastapi import WebSocket

if TYPE_CHECKING:
    from server.room import Room

logger = logging.getLogger(__name__)


class Peer:
    def __init__(self, client_id: str, room: "Room", ws: WebSocket):
        self.client_id = client_id
        self.room = room
        self.ws = ws
        self.pc = RTCPeerConnection()
        self.relay: Optional[MediaRelay] = None
        self._relay_track = None
        self._setup_pc_callbacks()

    def _setup_pc_callbacks(self) -> None:
        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                await self.ws.send_json({
                    "type": "candidate",
                    "candidate": candidate_to_sdp(candidate),
                    "sdp_mid": candidate.sdpMid,
                    "sdp_mline_index": candidate.sdpMLineIndex,
                })

        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info("Peer %s connection state: %s", self.client_id, self.pc.connectionState)

        @self.pc.on("track")
        async def on_track(track):
            logger.info("Peer %s received track: kind=%s", self.client_id, track.kind)
            if track.kind == "audio":
                await self._handle_audio_track(track)

    async def _handle_audio_track(self, track) -> None:
        self.relay = MediaRelay()
        self._relay_track = track

        others = self.room.get_others(self.client_id)
        for other in others:
            if other.pc.connectionState == "closed":
                continue
            try:
                proxy = self.relay.subscribe(track)
                other.pc.addTrack(proxy)
                await renegotiate(other)
                logger.info("Relaying audio from %s to %s", self.client_id, other.client_id)
            except Exception as e:
                logger.warning("Failed to relay audio to %s: %s", other.client_id, e)

    async def subscribe_to(self, source_peer: "Peer") -> None:
        """Subscribe this peer to receive audio from source_peer's relay."""
        if source_peer.relay and source_peer._relay_track:
            proxy = source_peer.relay.subscribe(source_peer._relay_track)
            self.pc.addTrack(proxy)
            logger.info("Peer %s subscribed to audio from %s", self.client_id, source_peer.client_id)

    async def close(self) -> None:
        await self.pc.close()


async def renegotiate(peer: Peer) -> None:
    """Send a new offer to trigger renegotiation after adding tracks."""
    if peer.pc.connectionState == "closed":
        return
    try:
        offer = await peer.pc.createOffer()
        await peer.pc.setLocalDescription(offer)
        await peer.ws.send_json({
            "type": "offer",
            "sdp": peer.pc.localDescription.sdp,
        })
    except Exception as e:
        logger.error("Renegotiation failed for peer %s: %s", peer.client_id, e)
