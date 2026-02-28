"""
CLI entrypoint for the customer support agent.

Usage:
    python -m agent.main
    python -m agent.main --id support-agent --room ai-room
    python -m agent.main --server ws://localhost:8080/ws
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> None:
    import json
    from agent.client import AgentClient
    from agent.agent import SupportAgent

    # Load system prompt from config
    config_path = Path(__file__).parent.parent / "config" / "support_agent.json"
    with open(config_path) as f:
        config = json.load(f)
    system_prompt = config["system_prompt"]

    # Create WebRTC client
    client = AgentClient(
        server_url=args.server,
        client_id=args.id,
        room=args.room,
    )

    # Connect WebRTC client
    await client.connect()

    # Create support agent (needs tts_track from client)
    agent = SupportAgent(
        client_id=args.id,
        room=args.room,
        system_prompt=system_prompt,
        tts_track=client.tts_track,
        ws_send=client.send_json,
        deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY"),
    )

    # Wire audio track -> agent
    async def on_track(track):
        if track.kind == "audio":
            asyncio.create_task(_consume_audio_track(track, agent))

    client.on_track = on_track

    # Start the agent (sends greeting)
    await agent.start()

    print(f"Support agent '{args.id}' connected to room '{args.room}'")
    print("Press Ctrl+C to stop.\n")

    # Wait until interrupted or conversation ends naturally
    stop_event = asyncio.Event()

    def _handle_signal():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    # Complete when either Ctrl+C or agent finishes the conversation
    done = asyncio.create_task(agent._done_event.wait())
    stop = asyncio.create_task(stop_event.wait())
    await asyncio.wait([done, stop], return_when=asyncio.FIRST_COMPLETED)
    done.cancel()
    stop.cancel()

    print("\nShutting down...")
    await agent.stop()
    await client.close()


async def _consume_audio_track(track, agent: "SupportAgent") -> None:
    """Continuously read frames from WebRTC audio track and send to agent."""
    from agent.agent import SupportAgent
    logger.info("Started consuming audio track: %s", track.kind)
    try:
        while True:
            frame = await track.recv()
            await agent.on_audio_frame(frame)
    except Exception as e:
        logger.info("Audio track ended: %s", e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Customer support agent")
    parser.add_argument("--id", default="support-agent", help="Agent client ID")
    parser.add_argument("--room", default="ai-room", help="Room to join")
    parser.add_argument("--server", default="ws://localhost:8080/ws", help="SFU server WebSocket URL")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
