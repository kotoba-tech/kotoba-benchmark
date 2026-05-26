"""Evaluate audio from a Hugging Face dataset (or local save_to_disk path).

The dataset must have an audio column named `audio_<source_lang>` containing
HF Audio entries. The pipeline runs translate → transcribe → align → score.

Usage:
  python examples/from_hf_dataset.py kotoba-speech/some-eval-set en ja
"""

from __future__ import annotations

import logging
import sys

from kotoba_benchmark import Config, evaluate


logging.basicConfig(level=logging.INFO)


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2

    dataset, source_lang, target_lang = sys.argv[1:4]

    config = Config(
        dataset=dataset,
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=f"./out/{source_lang}2{target_lang}_hf",
    )
    result = evaluate(config)
    print(result.summary_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
