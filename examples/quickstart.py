"""Minimal partner-runnable example: evaluate a directory of WAVs.

Setup:
  git clone https://github.com/kotoba-tech/kotoba-benchmark.git
  cd kotoba-benchmark && uv venv && source .venv/bin/activate && uv pip install -e .
  export KOTOBA_API_KEY=...
  export KOTOBA_S2ST_EN_JA_URL=wss://<your-endpoint>/v1/realtime_voice
  export GEMINI_API_KEY=...
  # export OPENAI_API_KEY=...   # only if you override align/evaluate to gpt-* or use openai-realtime backend

Run:
  python examples/quickstart.py /path/to/wavs
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from kotoba_benchmark import Config, evaluate


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    wav_dir = Path(sys.argv[1])

    config = Config(
        wav_dir=str(wav_dir),
        source_lang="en",
        target_lang="ja",
        output_dir="./out/quickstart",
    )
    result = evaluate(config)

    print()
    print(f"Done. Summary: {result.summary_paths.get('html', result.summary_paths.get('json'))}")
    scores = result.scores
    print(
        f"row_accuracy_mean={scores['row_accuracy_mean']} "
        f"row_fluency_mean={scores['row_fluency_mean']} "
        f"row_conciseness_mean={scores['row_conciseness_mean']} "
        f"median_chunk_latency_s={scores['median_latency_chunk']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
