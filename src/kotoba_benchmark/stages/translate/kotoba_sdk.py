"""kotoba-sdk translate backend (the default)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from kotoba_benchmark.stages.translate import TranslateChunk, register

# Number of leading audio_chunk events to discard. The Kotoba S2ST server emits
# a few setup/silence chunks at the start of every session; downstream metrics
# treat those as if they were translated output, so drop them.
_LEADING_AUDIO_CHUNKS_TO_SKIP = 5


@register("kotoba-sdk")
class KotobaSdkBackend:
    """Routes audio through a Kotoba S2ST WebSocket endpoint via kotoba-sdk.

    The Kotoba server is realtime-streaming: input chunks must be paced at
    wall-clock rate (≈ chunk_ms apart), and the first few output chunks are
    silence/intro that should be skipped.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        chunk_ms: int = 40,
        delay: int | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.chunk_ms = chunk_ms
        self.delay = delay

    async def translate(
        self,
        *,
        pcm16: bytes,
        sample_rate: int,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[TranslateChunk]:
        from kotoba import AsyncKotobaClient

        client = AsyncKotobaClient(api_key=self.api_key)
        chunk_bytes = int(sample_rate * (self.chunk_ms / 1000.0)) * 2
        chunk_interval = self.chunk_ms / 1000.0

        async with client.s2st.stream(
            src=source_lang,
            tgt=target_lang,
            sample_rate=sample_rate,
            url=self.url,
            delay=self.delay,
        ) as session:

            async def feeder() -> None:
                for i in range(0, len(pcm16), chunk_bytes):
                    await session.send_audio(pcm16[i : i + chunk_bytes])
                    await asyncio.sleep(chunk_interval)
                await session.commit()

            feed_task = asyncio.create_task(feeder())
            seen_audio_chunks = 0
            try:
                async for event in session:
                    received_at = time.monotonic()
                    if event.type == "audio_chunk" and event.audio:
                        seen_audio_chunks += 1
                        if seen_audio_chunks <= _LEADING_AUDIO_CHUNKS_TO_SKIP:
                            continue
                        yield TranslateChunk(
                            audio=event.audio,
                            sample_rate=sample_rate,
                            partial_source=None,
                            received_at_monotonic=received_at,
                        )
                    elif event.type == "partial_transcript" and event.text:
                        yield TranslateChunk(
                            audio=b"",
                            sample_rate=sample_rate,
                            partial_source=event.text,
                            received_at_monotonic=received_at,
                        )
                    elif event.type == "done":
                        break
            finally:
                if not feed_task.done():
                    feed_task.cancel()
                    try:
                        await feed_task
                    except (asyncio.CancelledError, Exception):
                        pass
