"""
WebRTC WebSocket client — mirrors Go client pkg.

Connects to ws://localhost:8080/ws, creates RTCPeerConnection with
TTSAudioTrack added upfront, handles signaling (join/offer/answer/candidate).
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Callable, Optional, Awaitable

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
import websockets

from agent.audio import TTSAudioTrack

logger = logging.getLogger(__name__)


class AgentClient:
    def __init__(
        self,
        server_url: str,
        client_id: str,
        room: str,
        on_track: Optional[Callable] = None,
        on_peer_event: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ):
        self._server_url = server_url
        self._client_id = client_id
        self._room = room
        self.on_track = on_track
        self.on_peer_event = on_peer_event

        self.pc = RTCPeerConnection()
        self.tts_track = TTSAudioTrack()
        self._ws = None
        self._running = False

    async def connect(self) -> None:
        """Connect to signaling server and start the WebRTC flow."""
        self.pc.addTrack(self.tts_track)
        self._setup_pc_callbacks()

        logger.info("Connecting to %s as %s in room %s", self._server_url, self._client_id, self._room)
        self._ws = await websockets.connect(self._server_url)
        self._running = True

        # Send join message
        await self._send({"type": "join", "room": self._room, "client_id": self._client_id})

        # Process incoming messages
        asyncio.create_task(self._recv_loop())

    def _setup_pc_callbacks(self) -> None:
        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                await self._send({
                    "type": "candidate",
                    "candidate": candidate_to_sdp(candidate),
                    "sdp_mid": candidate.sdpMid,
                    "sdp_mline_index": candidate.sdpMLineIndex,
                })

        @self.pc.on("track")
        async def on_track(track):
            logger.info("Received track: kind=%s", track.kind)
            if self.on_track:
                await self.on_track(track)

        @self.pc.on("connectionstatechange")
        async def on_connection_state():
            logger.info("PC connection state: %s", self.pc.connectionState)

    async def _recv_loop(self) -> None:
        """Main message receive loop."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON: %s", raw)
                    continue

                msg_type = msg.get("type")

                if msg_type == "offer":
                    await self._handle_offer(msg)
                elif msg_type == "candidate":
                    await self._handle_candidate(msg)
                elif msg_type in ("peer_joined", "peer_left"):
                    if self.on_peer_event:
                        await self.on_peer_event(msg.get("client_id", ""), msg_type)
                else:
                    logger.debug("Unhandled message type: %s", msg_type)
        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error("Recv loop error: %s", e)
        finally:
            self._running = False

    async def _handle_offer(self, msg: dict) -> None:
        sdp = msg.get("sdp", "")
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self._send({
            "type": "answer",
            "sdp": self.pc.localDescription.sdp,
        })

    async def _handle_candidate(self, msg: dict) -> None:
        candidate_str = msg.get("candidate")
        if candidate_str:
            candidate = candidate_from_sdp(candidate_str)
            candidate.sdpMid = msg.get("sdp_mid")
            sdp_mline = msg.get("sdp_mline_index")
            candidate.sdpMLineIndex = sdp_mline if sdp_mline is not None else 0
            await self.pc.addIceCandidate(candidate)

    async def write_audio(self, pcm_bytes: bytes, sample_rate: int = 22050) -> None:
        """Push TTS PCM audio to the WebRTC track."""
        await self.tts_track.push_pcm(pcm_bytes, sample_rate=sample_rate)

    async def send_json(self, msg: dict) -> None:
        """Send an arbitrary JSON message through the WebSocket."""
        await self._send(msg)

    async def _send(self, msg: dict) -> None:
        if self._ws:
            await self._ws.send(json.dumps(msg))

    async def close(self) -> None:
        self._running = False
        await self.pc.close()
        if self._ws:
            await self._ws.close()
