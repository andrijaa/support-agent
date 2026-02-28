"""
Audio utilities: TTSAudioTrack + PCM conversion helpers.

Pipeline:
  WebRTC audio (Opus → aiortc → av.AudioFrame s16 interleaved 48kHz stereo)
    → raw int16 bytes → Deepgram

  ElevenLabs PCM (22050Hz mono)
    → resample 22050→48000 (scipy)
    → stereo (duplicate channel)
    → av.AudioFrame → TTSAudioTrack queue → aiortc Opus → WebRTC
"""
from __future__ import annotations
import asyncio
import logging
from fractions import Fraction

import numpy as np
import av
from aiortc.mediastreams import AudioStreamTrack

logger = logging.getLogger(__name__)

TTS_INPUT_SAMPLE_RATE = 22050
WEBRTC_SAMPLE_RATE = 48000
WEBRTC_CHANNELS = 2
FRAME_DURATION_MS = 20  # aiortc standard
SAMPLES_PER_FRAME = WEBRTC_SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960


def av_frame_to_pcm_bytes(frame: av.AudioFrame) -> bytes:
    """Convert av.AudioFrame → interleaved int16 PCM bytes for Deepgram (48kHz stereo).

    aiortc decodes incoming Opus to s16 (interleaved int16, shape (1, samples*ch)).
    Some tracks may be fltp (float32 planar, shape (ch, samples)).
    """
    fmt = frame.format.name
    arr = frame.to_ndarray()

    if fmt == 's16':
        # Already interleaved int16: shape (1, samples*channels) → flatten
        return arr.reshape(-1).tobytes()

    if fmt == 's16p':
        # Planar int16: shape (channels, samples) → interleave
        if arr.shape[0] == 1:
            arr = np.stack([arr[0], arr[0]], axis=1)  # mono → stereo (samples, 2)
        else:
            arr = arr[:2].T  # (channels, samples) → (samples, channels)
        return arr.astype(np.int16).reshape(-1).tobytes()

    # Float formats (fltp, flt, dbl, dblp): shape (channels, samples) or (1, samples)
    arr = arr.astype(np.float32)
    if arr.ndim == 1 or arr.shape[0] == 1:
        mono = arr.reshape(-1)
        interleaved = np.stack([mono, mono], axis=1).reshape(-1)
    else:
        interleaved = arr[:2].T.reshape(-1)
    interleaved = np.clip(interleaved, -1.0, 1.0)
    return (interleaved * 32767).astype(np.int16).tobytes()


def resample_pcm(pcm_bytes: bytes, src_rate: int, dst_rate: int, channels: int = 1) -> bytes:
    """Resample PCM int16 bytes from src_rate to dst_rate."""
    from scipy.signal import resample_poly
    from math import gcd

    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        arr = arr.reshape(-1, channels)

    g = gcd(dst_rate, src_rate)
    up = dst_rate // g
    down = src_rate // g

    if channels > 1:
        resampled = np.stack([resample_poly(arr[:, c], up, down) for c in range(channels)], axis=1)
        resampled = resampled.reshape(-1)
    else:
        resampled = resample_poly(arr, up, down)

    resampled = np.clip(resampled, -1.0, 1.0)
    return (resampled * 32767).astype(np.int16).tobytes()


class TTSAudioTrack(AudioStreamTrack):
    """
    AudioStreamTrack that plays back TTS audio pushed via push_pcm().
    When queue is empty, outputs silence so the WebRTC connection stays alive.
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue()
        self._pts = 0
        self._start_time: float | None = None
        self._interrupted = False
        # aiortc Opus encoder requires interleaved s16 format
        self._resampler = av.AudioResampler(format="s16", layout="stereo", rate=WEBRTC_SAMPLE_RATE)

    async def recv(self) -> av.AudioFrame:
        """Called by aiortc to get the next audio frame.

        Paces output at real-time rate (one 20ms frame per 20ms) so that
        RTP packets are not sent in bursts that overwhelm the receiver.
        """
        loop = asyncio.get_event_loop()
        if self._start_time is None:
            self._start_time = loop.time()

        # Wait until this frame's wall-clock delivery time
        target_time = self._start_time + (self._pts / WEBRTC_SAMPLE_RATE)
        wait = target_time - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)

        if self._interrupted:
            frame = self._silence_frame()
        else:
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                frame = self._silence_frame()

        # Convert fltp → s16 (required by aiortc Opus encoder)
        frames = self._resampler.resample(frame)
        out = frames[0] if frames else self._silence_frame()
        out.pts = self._pts
        out.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        out.sample_rate = WEBRTC_SAMPLE_RATE
        self._pts += SAMPLES_PER_FRAME
        return out

    def _silence_frame(self) -> av.AudioFrame:
        silence = np.zeros((WEBRTC_CHANNELS, SAMPLES_PER_FRAME), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(silence, format="fltp", layout="stereo")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        return frame

    async def push_pcm(self, pcm_bytes: bytes, sample_rate: int = TTS_INPUT_SAMPLE_RATE,
                        channels: int = 1) -> None:
        """Resample TTS output, convert to stereo 48kHz, chunk into frames, enqueue."""
        if not pcm_bytes or self._interrupted:
            return

        # Resample to 48kHz
        if sample_rate != WEBRTC_SAMPLE_RATE:
            pcm_bytes = resample_pcm(pcm_bytes, sample_rate, WEBRTC_SAMPLE_RATE, channels)

        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if channels == 1:
            # mono → stereo
            arr_stereo = np.stack([arr, arr], axis=0)  # (2, N)
        else:
            arr_stereo = arr.reshape(-1, channels).T  # (channels, N)

        total_samples = arr_stereo.shape[1]
        offset = 0
        while offset < total_samples:
            chunk = arr_stereo[:, offset:offset + SAMPLES_PER_FRAME]
            if chunk.shape[1] < SAMPLES_PER_FRAME:
                pad = np.zeros((WEBRTC_CHANNELS, SAMPLES_PER_FRAME - chunk.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=1)
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            await self._queue.put(frame)
            offset += SAMPLES_PER_FRAME

    async def drain(self) -> None:
        """Wait until all queued audio frames have been sent via recv()."""
        while not self._queue.empty():
            await asyncio.sleep(0.05)

    def interrupt(self) -> None:
        """Set interrupted flag and clear queue for immediate silence."""
        self._interrupted = True
        self.clear_queue()

    def reset_interrupt(self) -> None:
        """Reset interrupted flag to allow audio playback again."""
        self._interrupted = False

    def clear_queue(self) -> None:
        """Discard queued audio (for interruption handling)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
