# kotoba-benchmark

Speech-to-speech (S2S) translation benchmark for Kotoba and other STS systems.

> **Alpha.** The pipeline runs end-to-end (translate → transcribe → align → score → report). Public PyPI release is still pending.

## Install

`kotoba-benchmark` itself isn't on PyPI yet, so install from source. `kotoba-sdk` and the rest of the deps are pulled from PyPI automatically. Python ≥ 3.10.

```bash
git clone https://github.com/kotoba-tech/kotoba-benchmark.git
cd kotoba-benchmark
uv venv && source .venv/bin/activate    # any venv manager works
uv pip install -e .                     # or: pip install -e .
```

Editable (`-e`) is recommended: pick up fixes with `git pull`, no re-install needed.

## Configure

Required for the default flow (`kotoba-sdk` backend + Gemini judge):

| Variable | Purpose |
|---|---|
| `KOTOBA_API_KEY` | Bearer token for Kotoba endpoints. |
| `KOTOBA_S2ST_<SRC>_<TGT>_URL` | WebSocket URL for the language pair you're benchmarking (e.g. `KOTOBA_S2ST_EN_JA_URL`). Can also be set in the TOML under `[translate].url`. |
| `GEMINI_API_KEY` | Source/target transcription **and** LLM-judged alignment + scoring. |

Optional — only needed if you override certain defaults:

| Variable | When you need it |
|---|---|
| `OPENAI_API_KEY` | Either: (a) you set `[align].model` or `[evaluate].model` to a `gpt-*` model, or (b) you set `[translate].backend = "openai-realtime"`. |
| `GOOGLE_API_KEY` | Accepted as a fallback for `GEMINI_API_KEY` if that one isn't set. |

## Run

CLI — a config is the unit of reproducibility:

```bash
kotoba-benchmark run configs/en2ja_smoke.toml
```

Python — same `Config` object, programmatic:

```python
from kotoba_benchmark import evaluate, Config

result = evaluate(Config(
    wav_dir="./my_wavs",
    source_lang="en",
    target_lang="ja",
))
print(result.scores)   # accuracy, fluency, conciseness, latency
```

The summary files are written automatically to `config.output_dir`. Call `result.write_summary("./other_dir")` to also write them somewhere else.

## Render Existing Results

To re-render summary files from a previous local run without rerunning translation,
transcription, alignment, or scoring:

```bash
kotoba-benchmark report ./out
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable script, [`docs/quickstart.md`](docs/quickstart.md) for the partner walkthrough, [`docs/config-reference.md`](docs/config-reference.md) for every TOML field, and [`docs/metrics.md`](docs/metrics.md) for what the scores mean.

## Benchmarking non-Kotoba systems

The translate stage is a pluggable backend. Two ship in the box:

- `kotoba-sdk` (default) — Kotoba S2ST endpoint.
- `openai-realtime` — OpenAI's Realtime Translation API. Set `[translate].backend = "openai-realtime"` and `OPENAI_API_KEY`. See [`configs/openai_realtime_en2ja.toml`](configs/openai_realtime_en2ja.toml).

For other STS systems (in-house servers, third parties) see [`docs/extending.md`](docs/extending.md) — a backend is a single class implementing a small `TranslateBackend` protocol.

## What it measures

Chunk-level accuracy, fluency, conciseness (LLM-judged) and translation-lag latency. See [`docs/metrics.md`](docs/metrics.md).

## License

MIT
