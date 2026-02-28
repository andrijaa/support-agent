import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from aiortc import RTCSessionDescription
from aiortc.sdp import candidate_from_sdp

from server.models import SignalMessage
from server.room import RoomManager
from server.peer import Peer, renegotiate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Conversational Agent SFU")
room_manager = RoomManager()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    peer: Peer | None = None
    room_name: str | None = None

    try:
        async for raw in ws.iter_text():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON: %s", e)
                continue

            # Forward conversation_data messages without Pydantic validation
            if data.get("type") == "conversation_data" and peer and room_name:
                room = await room_manager.get_or_create(room_name)
                await room.broadcast(data, exclude_id=peer.client_id)
                continue

            try:
                msg = SignalMessage(**data)
            except Exception as e:
                logger.warning("Invalid message: %s", e)
                continue

            if msg.type == "join":
                room_name = msg.room or "default"
                client_id = msg.client_id or "unknown"
                room = await room_manager.get_or_create(room_name)
                peer = Peer(client_id=client_id, room=room, ws=ws)

                # Subscribe to existing peers' audio before sending offer
                for other in room.get_others(client_id):
                    await peer.subscribe_to(other)

                await room.add(peer)

                # Notify others that a new peer joined
                await room.broadcast(
                    {"type": "peer_joined", "client_id": client_id},
                    exclude_id=client_id,
                )

                # Add sendrecv transceiver and send offer
                peer.pc.addTransceiver("audio", direction="sendrecv")
                offer = await peer.pc.createOffer()
                await peer.pc.setLocalDescription(offer)
                await ws.send_json({
                    "type": "offer",
                    "sdp": peer.pc.localDescription.sdp,
                })

            elif msg.type == "answer" and peer:
                desc = RTCSessionDescription(sdp=msg.sdp, type="answer")
                await peer.pc.setRemoteDescription(desc)

            elif msg.type == "candidate" and peer:
                if msg.candidate:
                    candidate = candidate_from_sdp(msg.candidate)
                    candidate.sdpMid = msg.sdp_mid
                    # Browser may omit both fields; default to first m-line
                    candidate.sdpMLineIndex = msg.sdp_mline_index if msg.sdp_mline_index is not None else 0
                    await peer.pc.addIceCandidate(candidate)

    except WebSocketDisconnect:
        logger.info("Peer %s disconnected", peer.client_id if peer else "unknown")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        if peer and room_name:
            room = await room_manager.get_or_create(room_name)
            await room.remove(peer.client_id)
            await peer.close()
            await room.broadcast(
                {"type": "peer_left", "client_id": peer.client_id},
            )
            await room_manager.cleanup(room_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8080, reload=False)
