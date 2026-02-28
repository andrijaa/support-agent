"""Shared fixtures for the test suite."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_peer():
    """Create a mock Peer with client_id and async ws.send_json."""
    def _make(client_id: str = "peer-1"):
        peer = MagicMock()
        peer.client_id = client_id
        peer.ws = MagicMock()
        peer.ws.send_json = AsyncMock()
        peer.relay = None
        peer._relay_track = None
        peer.pc = MagicMock()
        peer.pc.connectionState = "connected"
        return peer
    return _make


@pytest.fixture
def mock_websocket():
    """Create a mock FastAPI WebSocket."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    ws.accept = AsyncMock()
    return ws
