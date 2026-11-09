"""Public native WebSocket relay; no local models or text postprocessors."""

import asyncio
import logging
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError

from app.core.config import settings
from app.core.security import validate_websocket_token

from .client import open_stream, receive_event
from .protocol import (
    MAX_SECONDS,
    SAMPLE_RATE,
    StreamConfig,
    StreamError,
    supervise,
    validate_pcm,
)
from .speakers import SpeakerStream
from .vad import open_silence_stop

logger = logging.getLogger(__name__)


class GatewayStreamConfig(StreamConfig):
    # Speech followed by this much Silero non-speech ends the stream; 0 disables.
    auto_stop_silence_ms: int = Field(
        default_factory=lambda: settings.REALTIME_AUTO_STOP_MS, ge=0, le=60000
    )


async def handle_stream(websocket: WebSocket):
    valid, _ = validate_websocket_token(websocket)
    if not valid:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        config = GatewayStreamConfig.model_validate(
            await asyncio.wait_for(websocket.receive_json(), 10)
        )
        speakers = SpeakerStream()
        silence = (
            await open_silence_stop(config.auto_stop_silence_ms)
            if config.auto_stop_silence_ms
            else None
        )
        async with open_stream(StreamConfig(context=config.context)) as (
            upstream,
            ready,
        ):
            lock = asyncio.Lock()

            async def send(*events):
                async with lock:  # Audio and result tasks both emit labels.
                    for event in events:
                        await asyncio.wait_for(websocket.send_json(event), 5)

            await send(
                dict(
                    ready,
                    speaker_diarization=speakers.model is not None,
                    auto_stop_silence_ms=config.auto_stop_silence_ms if silence else 0,
                )
            )

            async def send_audio():
                samples = 0
                ended = auto_stopped = False
                while True:
                    message = await asyncio.wait_for(websocket.receive(), 30)
                    if message["type"] == "websocket.disconnect":
                        return
                    if auto_stopped:
                        continue  # Sent before the client saw auto_stop.
                    if ended:
                        raise StreamError(
                            "session_finished", "No messages are allowed after end"
                        )
                    data = message.get("bytes")
                    if data is not None:
                        samples += validate_pcm(data)
                        if (
                            samples
                            > min(MAX_SECONDS, ready["max_session_seconds"])
                            * SAMPLE_RATE
                        ):
                            raise StreamError(
                                "session_limit", "Session audio duration limit exceeded"
                            )
                        await asyncio.wait_for(upstream.send(data), 5)
                        speakers.feed(data)
                        await send(*await speakers.advance())
                        if silence and await silence.feed(data):
                            ended = auto_stopped = True
                            await send(
                                {
                                    "auto_stop": "silence",
                                    "audio_ms": samples * 1000 // SAMPLE_RATE,
                                }
                            )
                            await asyncio.wait_for(upstream.send("end"), 5)
                    elif message.get("text") == "end":
                        ended = True
                        await asyncio.wait_for(upstream.send("end"), 5)
                    else:
                        raise StreamError(
                            "invalid_message", "Send int16 PCM frames or the text end"
                        )

            async def send_results():
                start_ms, spoken = 0, False
                while True:
                    event = await receive_event(upstream)
                    spoken = spoken or bool(event["delta"].strip())
                    if event["utterance_end"]:
                        if spoken:
                            speakers.utterance(
                                event["utterance"], start_ms, event["audio_ms"]
                            )
                        start_ms, spoken = event["audio_ms"], False
                    if event["done"]:
                        # Every label precedes done, so clients can close on it.
                        speakers.ended = True
                        await send(*await speakers.advance(), event)
                        return
                    await send(event, *speakers.ready())

            await supervise(send_audio(), send_results())
    except WebSocketDisconnect:
        pass
    except Exception as error:
        if isinstance(error, StreamError):
            code, message = error.code, str(error)
        elif isinstance(error, (ValidationError, ValueError)):
            code, message = "invalid_config", "Invalid stream configuration"
        elif isinstance(error, asyncio.TimeoutError):
            code, message = (
                "session_timeout",
                "Session timed out or consumer is too slow",
            )
        else:
            logger.exception("Realtime gateway failed")
            code, message = "stream_failed", "Realtime session failed"
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(
                websocket.send_json({"error": message, "code": code}), 2
            )
    finally:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(websocket.close(), 2)
