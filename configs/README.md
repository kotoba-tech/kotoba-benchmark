# Config reference

A run config is a TOML file that fully describes one evaluation. CLI and Python API both materialize this same `Config` object — runs are reproducible from the file alone.

## Minimal config

```toml
wav_dir = "./my_wavs"
source_lang = "en"
target_lang = "ja"
```

Everything else has sensible defaults. See [`docs/config-reference.md`](../docs/config-reference.md) for the full field list.

## Layout

```toml
# Inputs (one of wav_dir or dataset is required)
wav_dir = "./my_wavs"                  # directory of .wav files
# dataset = "kotoba-speech/eval-en"    # OR an HF dataset name / local save_to_disk path

source_lang = "en"
target_lang = "ja"
output_dir = "./out"
write_summary = "json+md+html"         # one of: none, json, json+md, json+md+html

[translate]                            # speech-to-speech stage
backend = "kotoba-sdk"                 # built-in: "kotoba-sdk" (default), "openai-realtime". Others: docs/extending.md
# url = "wss://.../v1/realtime_voice"  # optional override of KOTOBA_S2ST_<SRC>_<TGT>_URL
max_concurrency = 4
max_retries = 3
retry_interval_seconds = 5.0
sample_rate = 24000
chunk_ms = 40
# label = "partner_en2ja_staging"      # free-form, used in output filenames
# delay = 10                           # backend-specific kwarg (kotoba-sdk only); see note below

[transcribe]                           # Gemini source + target transcription
model = "gemini-2.5-flash"
max_workers = 32

[align]                                # OpenAI segment alignment
model = "gpt-5"
# version = "v1.7"                     # pin a specific prompt; default is latest pair-specific
max_workers = 32

[evaluate]                             # OpenAI per-chunk scoring
model = "gpt-5"
# version = "v1.1"
max_workers = 32
```

### Backend-specific keys

Any extra key under `[translate]` is forwarded as a kwarg to the chosen backend's constructor. Unknown keys for that backend raise `TypeError` at startup — typos and cross-backend mix-ups fail loud rather than getting silently dropped.

- `kotoba-sdk` accepts `delay` (int) — forwarded to `AsyncKotobaClient.s2st.stream(delay=...)` to tune server-side pacing.
- `openai-realtime` accepts `input_transcription_model` and `noise_reduction`; both are off by default (see [`docs/config-reference.md`](../docs/config-reference.md)).

## Examples

- [`en2ja_smoke.toml`](en2ja_smoke.toml) — minimal partner smoke test on the bundled English clip; targets a Kotoba S2ST endpoint.
- [`partner_staging.toml.example`](partner_staging.toml.example) — partner template with placeholders for the Kotoba backend.
- [`openai_realtime_en2ja.toml`](openai_realtime_en2ja.toml) — evaluate OpenAI's Realtime Translation API end-to-end through the same pipeline.
