# Examples

Runnable scripts that exercise `kotoba-benchmark` end-to-end. Pair each one with a config from [`../configs/`](../configs/) or pass arguments on the command line.

## Prereqs

```bash
# Install from source (kotoba-benchmark isn't on PyPI yet):
git clone https://github.com/kotoba-tech/kotoba-benchmark.git
cd kotoba-benchmark
uv venv && source .venv/bin/activate
uv pip install -e .

# Configure (only KOTOBA_* + GEMINI_API_KEY are required for the defaults):
export KOTOBA_API_KEY=...
export KOTOBA_S2ST_EN_JA_URL=wss://<your-endpoint>/v1/realtime_voice
export GEMINI_API_KEY=...   # transcription + LLM judge (default)
# export OPENAI_API_KEY=... # only if you set align/evaluate to gpt-* or use the openai-realtime backend
```

## Bundled audio

| File | Duration | Used by |
|---|---|---|
| [`audio/en/sample_10s.wav`](audio/en/sample_10s.wav) | 10 s, English, 24 kHz mono PCM16 | [`../configs/en2ja_smoke.toml`](../configs/en2ja_smoke.toml) |

This is the only clip shipped with the repo. The smoke config evaluates this one file end-to-end as the quickest "did everything wire up?" check (~30 s wall-clock for translate + ~30 s for the LLM stages).

For real evaluations, point `wav_dir` at your own directory of WAVs.

## Scripts

### `quickstart.py` — wav-dir → result in 15 lines

```bash
python examples/quickstart.py ./examples/audio/en
```

Builds a Config in code, calls `evaluate(...)`, prints the row-level scores and the summary HTML path. The minimal partner-facing path.

### `from_hf_dataset.py` — evaluate a Hugging Face dataset

```bash
python examples/from_hf_dataset.py <dataset_name_or_local_save_to_disk_path> <source_lang> <target_lang>
```

The dataset must have an `audio_<source_lang>` column. Runs the same pipeline as `quickstart.py` but skips the WAV-loading step.

### `async_concurrency.py` — bump concurrency from Python

```bash
python examples/async_concurrency.py ./my_wavs
```

Same as `quickstart.py` but uses the async API and sets `translate.max_concurrency=8`. Useful for larger eval batches once you've validated the smoke path.

## Using the CLI instead

Everything in these scripts is also available as `kotoba-benchmark run <config.toml>`. The scripts exist as docs — pick the style that fits your workflow. The smoke config:

```bash
kotoba-benchmark run configs/en2ja_smoke.toml
```

is the closest equivalent to `quickstart.py` against the bundled clip.
