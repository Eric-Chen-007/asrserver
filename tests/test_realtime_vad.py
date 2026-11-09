import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.realtime.gateway import handle_stream
from app.services.realtime.protocol import StreamConfig
from app.services.realtime.vad import SilenceStop

FRAME = 2560  # 160 ms client frames.


class LoudIsSpeech:
    """Silero stand-in: a window is speech when its peak is clearly audible."""

    def __call__(self, window, sample_rate):
        return torch.tensor([[0.9 if window.abs().max() > 0.01 else 0.05]])


def pcm(value, seconds):
    return np.full(int(seconds * 16000), value, dtype="<i2").tobytes()


def frames(data):
    return [data[i : i + 2 * FRAME] for i in range(0, len(data), 2 * FRAME)]


async def feed(stop, data):
    fired = False
    for frame in frames(data):
        fired = await stop.feed(frame) or fired
    return fired


class SilenceStopTest(unittest.IsolatedAsyncioTestCase):
    async def test_fires_after_speech_then_configured_silence(self):
        stop = SilenceStop(LoudIsSpeech(), 3000)
        self.assertFalse(await feed(stop, pcm(3000, 1)))
        self.assertFalse(await feed(stop, pcm(0, 2.9)))
        self.assertTrue(await feed(stop, pcm(0, 0.2)))

    async def test_speech_resets_the_pause_and_silence_alone_never_fires(self):
        stop = SilenceStop(LoudIsSpeech(), 3000)
        self.assertFalse(await feed(stop, pcm(0, 5)))
        self.assertFalse(await feed(stop, pcm(3000, 0.1)))  # A click does not arm it.
        self.assertFalse(await feed(stop, pcm(0, 5)))
        self.assertFalse(await feed(stop, pcm(3000, 1)))
        self.assertFalse(await feed(stop, pcm(0, 2)))
        self.assertFalse(await feed(stop, pcm(3000, 0.5)))
        self.assertFalse(await feed(stop, pcm(0, 2.9)))
        self.assertTrue(await feed(stop, pcm(0, 0.2)))

    async def test_inference_failure_disables_auto_stop(self):
        def broken(window, sample_rate):
            raise RuntimeError("boom")

        stop = SilenceStop(broken, 3000)
        self.assertFalse(await feed(stop, pcm(3000, 1)))
        self.assertIsNone(stop.model)
        self.assertFalse(await feed(stop, pcm(0, 5)))


class FakeSpeakers:
    model = None
    ended = False

    def feed(self, data):
        pass

    async def advance(self):
        return []

    def ready(self):
        return []

    def utterance(self, *args):
        pass


class FakeUpstream:
    def __init__(self):
        self.sent = []
        self.events = asyncio.Queue()

    async def send(self, data):
        self.sent.append(data)
        if data == "end":
            await self.events.put(
                json.dumps(
                    {
                        "delta": "",
                        "audio_ms": 0,
                        "done": True,
                        "utterance": 0,
                        "utterance_end": True,
                        "text": "你好",
                    }
                )
            )

    async def recv(self):
        return await self.events.get()


class GatewayAutoStopTest(unittest.TestCase):
    def setUp(self):
        self.opened = []

        @asynccontextmanager
        async def open_stream(config):
            upstream = FakeUpstream()
            self.opened.append((config, upstream))
            yield upstream, {"ready": True, "max_session_seconds": 3600}

        async def open_silence_stop(silence_ms):
            return SilenceStop(LoudIsSpeech(), silence_ms)

        for target, value in (
            ("open_stream", open_stream),
            ("open_silence_stop", open_silence_stop),
            ("SpeakerStream", FakeSpeakers),
        ):
            patcher = patch(f"app.services.realtime.gateway.{target}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.add_api_websocket_route("/v1/stream", handle_stream)
        self.client = TestClient(app)

    def test_pause_ends_the_stream_and_late_input_is_ignored(self):
        with self.client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"context": "热词", "auto_stop_silence_ms": 3000})
            self.assertEqual(ws.receive_json()["auto_stop_silence_ms"], 3000)
            for frame in frames(pcm(3000, 1) + pcm(0, 4)):
                ws.send_bytes(frame)
            ws.send_text("end")
            # Speech ends in the window at 1.024 s; 3 s later lands in frame 26.
            self.assertEqual(ws.receive_json(), {"auto_stop": "silence", "audio_ms": 4160})
            self.assertEqual(ws.receive_json()["text"], "你好")
        config, upstream = self.opened[0]
        self.assertEqual(config, StreamConfig(context="热词"))
        self.assertEqual(upstream.sent.count("end"), 1)
        self.assertEqual(sum(len(d) for d in upstream.sent if d != "end"), 2 * 4160 * 16)

    def test_server_default_and_disabled_sessions_wait_for_end(self):
        with patch("app.services.realtime.gateway.settings.REALTIME_AUTO_STOP_MS", 3000):
            with self.client.websocket_connect("/v1/stream") as ws:
                ws.send_json({"auto_stop_silence_ms": 0})
                self.assertEqual(ws.receive_json()["auto_stop_silence_ms"], 0)
                for frame in frames(pcm(3000, 1) + pcm(0, 4)):
                    ws.send_bytes(frame)
                ws.send_text("end")
                self.assertTrue(ws.receive_json()["done"])
            with self.client.websocket_connect("/v1/stream") as ws:
                ws.send_json({})
                self.assertEqual(ws.receive_json()["auto_stop_silence_ms"], 3000)
                ws.send_text("end")
                self.assertTrue(ws.receive_json()["done"])
            with self.client.websocket_connect("/v1/stream") as ws:
                ws.send_json({"auto_stop_silence_ms": -1})
                self.assertEqual(ws.receive_json()["code"], "invalid_config")


class SileroModelTest(unittest.IsolatedAsyncioTestCase):
    async def test_packaged_model_runs_and_ignores_silence(self):
        try:
            from app.services.realtime.vad import load_model
            model = load_model()
        except ImportError:
            self.skipTest("silero-vad is not installed")
        stop = SilenceStop(model, 500)
        self.assertFalse(await feed(stop, pcm(0, 2)))
        self.assertIsNotNone(stop.model)
        self.assertEqual(stop.speech, 0)


if __name__ == "__main__":
    unittest.main()
