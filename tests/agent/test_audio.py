"""Tests for agent.audio — av_frame_to_pcm_bytes, resample_pcm, TTSAudioTrack."""
import asyncio
import struct

import numpy as np
import av
import pytest

from agent.audio import (
    av_frame_to_pcm_bytes,
    resample_pcm,
    TTSAudioTrack,
    WEBRTC_SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    WEBRTC_CHANNELS,
    TTS_INPUT_SAMPLE_RATE,
)


class TestAvFrameToPcmBytes:
    def test_s16_format_correct_length(self):
        """s16 interleaved: (1, samples*channels) → bytes length = samples*channels*2."""
        samples = 960
        channels = 2
        data = np.zeros((1, samples * channels), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(data, format="s16", layout="stereo")
        frame.sample_rate = 48000

        result = av_frame_to_pcm_bytes(frame)
        assert len(result) == samples * channels * 2

    def test_s16p_mono_duplicated_to_stereo(self):
        """s16p mono: (1, samples) → duplicated to stereo → bytes = samples*2*2."""
        samples = 960
        data = np.ones((1, samples), dtype=np.int16) * 1000
        frame = av.AudioFrame.from_ndarray(data, format="s16p", layout="mono")
        frame.sample_rate = 48000

        result = av_frame_to_pcm_bytes(frame)
        # Each sample duplicated to stereo: samples * 2 channels * 2 bytes
        assert len(result) == samples * 2 * 2
        # Check first stereo pair: both channels should be 1000
        left, right = struct.unpack_from("<hh", result, 0)
        assert left == 1000
        assert right == 1000

    def test_fltp_stereo_to_interleaved_int16(self):
        """fltp stereo: float32 (2, samples) → int16 interleaved."""
        samples = 480
        left = np.full(samples, 0.5, dtype=np.float32)
        right = np.full(samples, -0.5, dtype=np.float32)
        data = np.stack([left, right], axis=0)
        frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
        frame.sample_rate = 48000

        result = av_frame_to_pcm_bytes(frame)
        assert len(result) == samples * 2 * 2  # stereo int16

        # Check first pair
        l_val, r_val = struct.unpack_from("<hh", result, 0)
        assert abs(l_val - 16383) < 2  # 0.5 * 32767 ≈ 16383
        assert abs(r_val - (-16383)) < 2

    def test_float_values_clipped(self):
        """Float values > 1.0 should be clipped to 1.0."""
        samples = 10
        data = np.full((2, samples), 2.0, dtype=np.float32)  # above 1.0
        frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
        frame.sample_rate = 48000

        result = av_frame_to_pcm_bytes(frame)
        arr = np.frombuffer(result, dtype=np.int16)
        # All values should be clipped to 32767 (int16 max)
        assert np.all(arr == 32767)


class TestResamplePcm:
    def test_identity_resample(self):
        """Same rate → output ≈ same length."""
        samples = 1000
        pcm = np.zeros(samples, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 48000, 48000, channels=1)
        # Should be approximately the same length (may differ slightly due to filter)
        assert abs(len(result) - len(pcm)) < 20

    def test_upsample_22050_to_48000(self):
        """22050→48000 should produce ~2.177x samples."""
        samples = 1000
        pcm = np.zeros(samples, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 22050, 48000, channels=1)
        expected_samples = int(samples * 48000 / 22050)
        actual_samples = len(result) // 2  # int16 = 2 bytes
        assert abs(actual_samples - expected_samples) < 10

    def test_multichannel_resample(self):
        """Multi-channel resampling should work without error."""
        samples_per_ch = 500
        channels = 2
        pcm = np.zeros(samples_per_ch * channels, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 22050, 48000, channels=2)
        assert len(result) > len(pcm)  # Upsampled


class TestTTSAudioTrack:
    async def test_recv_returns_silence_when_empty(self):
        track = TTSAudioTrack()
        frame = await track.recv()
        assert frame is not None
        assert frame.sample_rate == WEBRTC_SAMPLE_RATE

    async def test_push_pcm_and_recv(self):
        track = TTSAudioTrack()

        # Push enough PCM for one frame (960 samples stereo at 48kHz = 960*2*2 bytes)
        # But push_pcm resamples from 22050 mono, so we need to provide mono 22050 data
        # that after resampling gives us at least one frame
        samples = 960  # enough at 22050 to produce frames at 48000
        pcm = np.zeros(samples, dtype=np.int16).tobytes()
        await track.push_pcm(pcm, sample_rate=TTS_INPUT_SAMPLE_RATE)

        # Queue should not be empty after push
        assert not track._queue.empty()

    async def test_interrupt_clears_queue(self):
        track = TTSAudioTrack()

        # Push some data
        pcm = np.zeros(960, dtype=np.int16).tobytes()
        await track.push_pcm(pcm, sample_rate=TTS_INPUT_SAMPLE_RATE)
        assert not track._queue.empty()

        track.interrupt()
        assert track._queue.empty()
        assert track._interrupted is True

    async def test_push_pcm_noop_when_interrupted(self):
        track = TTSAudioTrack()
        track.interrupt()

        pcm = np.zeros(960, dtype=np.int16).tobytes()
        await track.push_pcm(pcm, sample_rate=TTS_INPUT_SAMPLE_RATE)
        assert track._queue.empty()

    async def test_reset_interrupt_allows_playback(self):
        track = TTSAudioTrack()
        track.interrupt()
        assert track._interrupted is True

        track.reset_interrupt()
        assert track._interrupted is False

        pcm = np.zeros(960, dtype=np.int16).tobytes()
        await track.push_pcm(pcm, sample_rate=TTS_INPUT_SAMPLE_RATE)
        assert not track._queue.empty()
