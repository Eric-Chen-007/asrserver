"""Silero VAD end-of-turn detection for the public realtime stream.

The engine's energy gate still splits utterances; this only decides when a
speaker who has already talked stayed silent long enough to close the stream.
"""

import asyncio
import logging

import numpy as np
import torch

from .protocol import SAMPLE_RATE

logger = logging.getLogger(__name__)

WINDOW_SAMPLES = 512  # Silero v5+ accepts exactly 512 samples at 16 kHz.
MIN_SPEECH_SAMPLES = 8 * WINDOW_SAMPLES  # 256 ms, so a click cannot arm the stop.


def load_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


class SilenceStop:
    """Silero state for one connection; an inference failure only disables it."""

    def __init__(self, model, silence_ms: int, threshold: float = 0.5) -> None:
        self.model = model  # Recurrent state: never share across connections.
        self.limit = silence_ms * SAMPLE_RATE // 1000
        self.threshold = threshold
        self.pending = np.empty(0, dtype=np.float32)
        self.speech = 0
        self.silence = 0

    async def feed(self, pcm: bytes) -> bool:
        """Return True once speech has been followed by the configured silence."""
        if self.model is None:
            return False
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.pending = np.concatenate((self.pending, audio))
        count = len(self.pending) // WINDOW_SAMPLES
        if not count:
            return False
        windows = self.pending[: count * WINDOW_SAMPLES].reshape(count, WINDOW_SAMPLES)
        self.pending = self.pending[count * WINDOW_SAMPLES :]
        try:
            probabilities = await asyncio.to_thread(self._infer, windows)
        except Exception:
            logger.exception("Silero VAD failed; realtime auto stop is disabled")
            self.model = None
            return False
        for probability in probabilities:
            if probability >= self.threshold:
                self.speech += WINDOW_SAMPLES
                self.silence = 0
            elif self.speech >= MIN_SPEECH_SAMPLES and (
                # Hysteresis: once counting, only real speech resets the pause.
                self.silence or probability < self.threshold - 0.15
            ):
                self.silence += WINDOW_SAMPLES
        return self.speech >= MIN_SPEECH_SAMPLES and self.silence >= self.limit

    def _infer(self, windows: np.ndarray) -> list[float]:
        with torch.inference_mode():
            return [
                float(self.model(torch.from_numpy(window.copy()), SAMPLE_RATE))
                for window in windows
            ]


async def open_silence_stop(silence_ms: int) -> SilenceStop | None:
    try:
        return SilenceStop(await asyncio.to_thread(load_model), silence_ms)
    except Exception:
        logger.exception("Silero VAD is unavailable; realtime auto stop is disabled")
        return None
