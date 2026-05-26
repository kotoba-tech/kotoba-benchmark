# Extending kotoba-benchmark

## Adding a translate backend

The translate stage is the only vendor-tied part of the pipeline. The repo ships two built-in backends:

- `kotoba-sdk` — Kotoba S2ST endpoint (default).
- `openai-realtime` — OpenAI's Realtime Translation API.

To benchmark another STS system (in-house servers, other providers), implement the `TranslateBackend` protocol:

```python
import time
from kotoba_benchmark.stages.translate import TranslateChunk, register

@register("my-backend")   # optional registry name; otherwise reference by dotted path
class MyBackend:
    def __init__(self, *, url: str | None = None, api_key: str | None = None, **kwargs):
        # Backend-specific kwargs are passed through from [translate] config fields.
        self.url = url
        self.api_key = api_key

    async def translate(self, *, pcm16, sample_rate, source_lang, target_lang):
        # Open your WS / HTTP session, stream input audio in, yield TranslateChunk
        # as the server emits output. pcm16 is PCM16 LE mono at `sample_rate` Hz.
        async for event in my_session(pcm16, sample_rate, source_lang, target_lang):
            yield TranslateChunk(
                audio=event.pcm16 or b"",
                sample_rate=sample_rate,
                partial_source=event.translated_text,   # any target-language transcript text
                received_at_monotonic=time.monotonic(),
            )
```

Reference it from TOML:

```toml
[translate]
backend = "my-backend"                       # registered name
# backend = "mypkg.module.MyBackend"          # OR an importable dotted path
# any other [translate].* fields get forwarded to MyBackend.__init__
url = "wss://my-stuff/realtime"
```

Or pass an instance from Python:

```python
from kotoba_benchmark import evaluate, Config
result = evaluate(Config(..., translate_backend=MyBackend(url="...")))
```

The rest of the pipeline (transcribe → align → score) doesn't care which backend produced the audio — anything that yields `TranslateChunk`s is fair game.

### Notes

- **Pacing.** If the upstream service is realtime-streaming (e.g. simultaneous translation models), you'll usually want to sleep `chunk_ms` between `send_audio` calls. See `stages/translate/kotoba_sdk.py` for the pattern.
- **`partial_source`** in `TranslateChunk` is the partial transcript text the runner accumulates into `<id>_translation.txt`. For both built-in backends this is the target-language (translated) text. Use whichever transcript your server emits and that you want surfaced to partners.
- **Audio.** `chunk.audio` is appended to a PCM16 buffer and written as a `<id>_output.wav` per clip. Empty `chunk.audio` is fine for text-only events.

## Adding a metric

The metric interface is not yet stable. The current per-chunk scores
(accuracy / fluency / conciseness via OpenAI or Gemini) and latency
metrics are computed directly in `stages/score.py`. If you want to add
something like a referenceless audio-quality MOS (e.g. NISQA) or an
ASR-WER-against-reference metric, the right place is a new stage between
`align` and `report`, with the result written as a new column on the
dataset and surfaced in the summary writer.
