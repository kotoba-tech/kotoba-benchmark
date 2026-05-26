"""Async API with custom concurrency for larger eval batches.

Usage:
  python examples/async_concurrency.py /path/to/wavs
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from kotoba_benchmark import Config, evaluate_async


logging.basicConfig(level=logging.INFO)


async def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    wav_dir = Path(sys.argv[1])
    config = Config(
        wav_dir=str(wav_dir),
        source_lang="en",
        target_lang="ja",
        output_dir="./out/concurrent",
        translate={"backend": "kotoba-sdk", "max_concurrency": 8},  # type: ignore[arg-type]
    )
    result = await evaluate_async(config)
    print(result.summary_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
