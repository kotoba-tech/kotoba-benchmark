# Config reference

A `Config` describes one evaluation run. CLI loads it from TOML; the Python API takes a `Config` object directly. Both paths share the same fields.

## Top-level

| Field | Type | Default | Purpose |
|---|---|---|---|
| `wav_dir` | path | — | Directory of audio files (one row per file). Mutually exclusive with `dataset`. |
| `dataset` | str / path | — | HF dataset name (e.g. `"kotoba-speech/eval-en"`) or local `save_to_disk` path. Must have an `audio_<source_lang>` column. |
| `source_lang` | str | — | Source language code, e.g. `"en"`. **Required.** |
| `target_lang` | str | — | Target language code, e.g. `"ja"`. **Required.** |
| `output_dir` | path | `./out` | Where summary files, translated WAVs, and stage caches land. |
| `write_summary` | str | `"json+md+html"` | One of `none`, `json`, `json+md`, `json+md+html`. |

## `[translate]`

The translate stage talks to a speech-to-speech translation server.

| Field | Default | Purpose |
|---|---|---|
| `backend` | `"kotoba-sdk"` | Built-in backends: `"kotoba-sdk"` (default), `"openai-realtime"`. For other STS systems see [`extending.md`](extending.md). |
| `url` | env-resolved | `kotoba-sdk` reads `KOTOBA_S2ST_<SRC>_<TGT>_URL` when omitted; `openai-realtime` defaults to `wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate`. |
| `max_concurrency` | `4` | How many WAVs are streamed concurrently. |
| `max_retries` | `3` | Retry attempts per WAV on connection / timeout failure. |
| `retry_interval_seconds` | `5.0` | Fixed sleep between retries. |
| `sample_rate` | `24000` | Send sample rate. Input audio is resampled if needed. |
| `chunk_ms` | `40` | Audio chunk size sent to the WS server. |
| `label` | — | Free-form tag used in output filenames (e.g. `"partner_en2ja_staging"`). |

Backend-specific keys are passed through to the backend's constructor.

### Built-in backends

| Name | API key env var | What it talks to |
|---|---|---|
| `kotoba-sdk` (default) | `KOTOBA_API_KEY` | Kotoba S2ST WebSocket endpoint via the `kotoba-sdk` package. Accepts an optional `delay` (int, currently 0–25) under `[translate]` that is forwarded to the server as `session.delay`. |
| `openai-realtime` | `OPENAI_API_KEY` | OpenAI's Realtime Translation API. Source language is auto-detected; only `target_lang` is sent. Realtime API requires an OpenAI account with Realtime access. Optional extras (both off by default): `input_transcription_model` (e.g. `"gpt-realtime-whisper"`) and `noise_reduction` (e.g. `"near_field"`). We don't enable either by default — the benchmark already runs Gemini-based source transcription in the transcribe stage, and enabling them costs server-side compute without affecting what we measure. |

## `[transcribe]`

Gemini source + target transcription with structured timestamps.

| Field | Default | Purpose |
|---|---|---|
| `model` | `"gemini-2.5-flash"` | Gemini model. |
| `temperature` | `0.2` | Sampling temperature. |
| `max_output_tokens` | `131072` | Output cap. |
| `thinking_level` | — | If supported by the model. |
| `request_timeout_seconds` | `300.0` | Per-request timeout. |
| `max_retries` | `3` | Per-row retries with exponential backoff. |
| `max_workers` | `32` | Parallel API calls (ThreadPoolExecutor). |

## `[align]`

LLM segment alignment.

| Field | Default | Purpose |
|---|---|---|
| `model` | `"gemini-2.5-flash"` | LLM judge. Model-name prefix selects the SDK: `gemini-*` → native google-genai (default); anything else (`gpt-*`, etc.) → OpenAI chat completions. `gpt-5.2` is a known-good OpenAI choice if you switch. |
| `version` | `None` | Pin a specific prompt version (e.g. `"v1.7"`). When None, the latest pair-specific prompt wins. |
| `max_workers` | `32` | Parallel API calls. |
| `max_retries` | `3` | Best-of-N attempts per row, kept if alignment validates. |
| `rps_limit` | `50` | Requests per second cap. |
| `request_timeout_seconds` | `120` | Per-request timeout. |
| `use_cache` | `true` | On-disk content-addressed cache under `<output_dir>/_cache/align/`. |

## `[evaluate]`

LLM per-chunk scoring (accuracy / fluency / conciseness as 0/1).

| Field | Default | Purpose |
|---|---|---|
| `model` | `"gemini-2.5-flash"` | Same dispatch rule as `[align].model`. The OpenAI path's production default is `gpt-4.1` if you switch. |
| `version` | `None` | Pin a specific evaluation prompt version. |
| `max_workers` | `32` | Parallel API calls. |
| `max_retries` | `3` | Per-row retries. |
| `rps_limit` | `50` | Requests per second cap. |
| `request_timeout_seconds` | `60` | Per-request timeout. |
| `use_cache` | `true` | On-disk cache under `<output_dir>/_cache/evaluate/`. |

## Overrides from the CLI

`kotoba-benchmark run config.toml --override translate.url=wss://... --override translate.max_concurrency=8`

Dotted keys target the same fields as the TOML sections. Values are coerced to int / float / bool / str / null.

## Prompt selection

Prompts are vendored under `kotoba_benchmark.prompts.{align,evaluate}` as TOML. Selection rules when no `version` is pinned:

1. Pick the highest-version pair-specific prompt (e.g. `en2ja-v1.7.toml`).
2. Fall back to the highest-version pair-agnostic prompt (e.g. `v1.2.toml`).

Use `kotoba-benchmark show-prompts --source-lang en --target-lang ja` to confirm which TOMLs would be picked.
