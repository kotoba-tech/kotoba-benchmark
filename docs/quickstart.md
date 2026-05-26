# Quickstart

Evaluate a directory of WAV files against a Kotoba S2ST endpoint in under 5 minutes.

## 1. Install

`kotoba-benchmark` isn't on PyPI yet — install from source. `kotoba-sdk` and the rest of the deps come from PyPI. Python ≥ 3.10.

```bash
git clone https://github.com/kotoba-tech/kotoba-benchmark.git
cd kotoba-benchmark
uv venv && source .venv/bin/activate
uv pip install -e .          # editable: `git pull` picks up future fixes
```

`pip install -e .` works too if you don't have `uv`.

## 2. Configure env vars

```bash
export KOTOBA_API_KEY=...                                                # bearer token
export KOTOBA_S2ST_EN_JA_URL=wss://<your-endpoint>/v1/realtime_voice
export GEMINI_API_KEY=...                                                # transcription + LLM judge
# export OPENAI_API_KEY=...                                              # only if you set align/evaluate model to gpt-*
```

Only set the `KOTOBA_S2ST_<SRC>_<TGT>_URL` for the language pair you're benchmarking. You can also specify the URL in the config file under `[translate].url`.

## 3. Prepare WAVs

Drop your audio files into one directory:

```
my_wavs/
├── clip_001.wav
├── clip_002.wav
└── ...
```

Files should be:
- Mono (the pipeline downmixes if needed)
- WAV / FLAC / OGG / MP3 (whatever soundfile can read)
- Any length — the pipeline streams each clip in realtime to the STS endpoint, so a 5-minute clip takes ~5 minutes of wall-clock to translate (multiplied across `translate.max_concurrency` clips in parallel)

## 4. Write a config

`my_run.toml`:

```toml
wav_dir = "./my_wavs"
source_lang = "en"
target_lang = "ja"
output_dir = "./out"

[translate]
backend = "kotoba-sdk"
max_concurrency = 4
label = "my_first_run"
```

## 5. Run

```bash
kotoba-benchmark run my_run.toml
```

The pipeline runs four stages:
1. **translate** — streams each WAV through the Kotoba S2ST endpoint, writes translated WAVs to `out/my_first_run__en2ja__output_s2s/`.
2. **transcribe** — Gemini extracts source + target timestamps.
3. **align** — OpenAI aligns source/target into matched chunks.
4. **score** — OpenAI rates each chunk on accuracy, fluency, conciseness; latency is computed from chunk timestamps.

When complete, open `out/my_first_run__en2ja__summary.html` in a browser — it has the metrics, embedded audio players for every output clip, and per-clip scores.

## What you get

- `out/my_first_run__en2ja__summary.json` — machine-readable scores
- `out/my_first_run__en2ja__summary.md` — Markdown report
- `out/my_first_run__en2ja__summary.html` — interactive HTML with audio
- `out/my_first_run__en2ja__output_s2s/*.wav` — translated audio
- `out/_stage_cache__<stage>__...` — per-stage checkpoints. Reruns skip completed stages.

## Re-running

The pipeline caches each stage's output. To re-render the summary without re-running:

```bash
kotoba-benchmark report ./out
```

To force a stage to rerun, delete its `_stage_cache__<stage>__*` directory.

## Programmatic use

```python
from kotoba_benchmark import Config, evaluate

result = evaluate(Config(
    wav_dir="./my_wavs",
    source_lang="en",
    target_lang="ja",
))
print(result.scores)
result.summary_paths  # {"html": ..., "md": ..., "json": ...}
```

See [`examples/`](../examples/) for more.
