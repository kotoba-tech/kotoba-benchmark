"""OpenAI Realtime Translation backend.

Routes audio through OpenAI's Realtime Translation WebSocket
(`wss://api.openai.com/v1/realtime/translations?model=...`). Source language is
auto-detected by the server; target language is set via `session.update`.
Audio is paced at wall-clock rate; output transcript text is collected as the
"translation text" returned to the partner (target language).

Docs: https://developers.openai.com/api/docs/guides/realtime-translation
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections.abc import AsyncIterator

import aiohttp

from kotoba_benchmark.stages.translate import TranslateChunk, register

logger = logging.getLogger(__name__)


_DEFAULT_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"


@register("openai-realtime")
class OpenAIRealtimeBackend:
    """Streams source audio through OpenAI's Realtime Translation API.

    Defaults to the minimum session config: just the target language. Source ASR
    (`input_transcription_model`) and `noise_reduction` are both optional — both
    add server-side work without affecting what the benchmark measures, since we
    do source transcription via Gemini in the transcribe stage. Override either
    via TOML if you want them enabled.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        chunk_ms: int = 40,
        input_transcription_model: str | None = None,
        noise_reduction: str | None = None,
    ) -> None:
        self.url = url or _DEFAULT_URL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the openai-realtime backend"
            )
        self.chunk_ms = chunk_ms
        self.input_transcription_model = input_transcription_model
        self.noise_reduction = noise_reduction

    @staticmethod
    async def _await_session_updated(ws: aiohttp.ClientWebSocketResponse) -> None:
        """Block until the server emits `session.updated`. Raises on `error` events."""
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            msg_type = payload.get("type")
            if msg_type == "session.updated":
                return
            if msg_type == "error":
                raise RuntimeError(
                    f"OpenAI Realtime error during session.update handshake: {payload}"
                )

    async def translate(
        self,
        *,
        pcm16: bytes,
        sample_rate: int,
        source_lang: str,  # noqa: ARG002 — server auto-detects source
        target_lang: str,
    ) -> AsyncIterator[TranslateChunk]:
        chunk_bytes = int(sample_rate * (self.chunk_ms / 1000.0)) * 2
        chunk_interval = self.chunk_ms / 1000.0

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(self.url, headers=headers, autoping=True) as ws:
                session_cfg: dict = {
                    "audio": {"output": {"language": target_lang}},
                }
                if self.input_transcription_model or self.noise_reduction:
                    input_cfg: dict = {}
                    if self.input_transcription_model:
                        input_cfg["transcription"] = {
                            "model": self.input_transcription_model
                        }
                    if self.noise_reduction:
                        input_cfg["noise_reduction"] = {"type": self.noise_reduction}
                    session_cfg["audio"]["input"] = input_cfg
                await ws.send_str(
                    json.dumps({"type": "session.update", "session": session_cfg})
                )

                # Wait for the server to acknowledge session.update before
                # sending audio. Without this, early audio chunks can race
                # the handshake and arrive before the target_lang is configured,
                # causing the server to emit ~silence for a substantial prefix.
                try:
                    await asyncio.wait_for(
                        self._await_session_updated(ws), timeout=10.0,
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        "Timed out waiting for session.updated from OpenAI Realtime"
                    ) from exc

                async def feeder() -> None:
                    try:
                        for i in range(0, len(pcm16), chunk_bytes):
                            await ws.send_str(
                                json.dumps(
                                    {
                                        "type": "session.input_audio_buffer.append",
                                        "audio": base64.b64encode(
                                            pcm16[i : i + chunk_bytes]
                                        ).decode("ascii"),
                                    }
                                )
                            )
                            await asyncio.sleep(chunk_interval)
                        await ws.send_str(json.dumps({"type": "session.close"}))
                    except (aiohttp.ClientError, ConnectionError) as exc:
                        logger.warning("openai-realtime feeder aborted: %s", exc)

                feed_task = asyncio.create_task(feeder())
                try:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        received_at = time.monotonic()
                        try:
                            payload = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue
                        msg_type = payload.get("type")

                        if msg_type == "session.output_audio.delta":
                            try:
                                audio = base64.b64decode(payload.get("delta", ""))
                            except Exception:  # noqa: BLE001
                                continue
                            if audio:
                                yield TranslateChunk(
                                    audio=audio,
                                    sample_rate=sample_rate,
                                    partial_source=None,
                                    received_at_monotonic=received_at,
                                )
                        elif msg_type == "session.output_transcript.delta":
                            # Target-language translated text. The runner accumulates
                            # this into `<id>_translation.txt`.
                            text = payload.get("delta", "")
                            if text:
                                yield TranslateChunk(
                                    audio=b"",
                                    sample_rate=sample_rate,
                                    partial_source=text,
                                    received_at_monotonic=received_at,
                                )
                        elif msg_type in {"session.closed", "session.output_audio.done"}:
                            break
                        elif msg_type == "error":
                            raise RuntimeError(f"OpenAI Realtime API error: {payload}")
                finally:
                    if not feed_task.done():
                        feed_task.cancel()
                        try:
                            await feed_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
