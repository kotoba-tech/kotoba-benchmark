"""kotoba-benchmark — speech-to-speech translation benchmark.

Quickstart:

    from kotoba_benchmark import evaluate, Config

    result = evaluate(Config(
        wav_dir="./my_wavs",
        source_lang="en",
        target_lang="ja",
    ))
    print(result.scores)
"""

from kotoba_benchmark.config import (
    AlignConfig,
    Config,
    EvaluateConfig,
    TranscribeConfig,
    TranslateConfig,
)
from kotoba_benchmark.pipeline import (
    Result,
    evaluate,
    evaluate_async,
    re_render_summary,
    render_summary_from_dataset,
)

__all__ = [
    "AlignConfig",
    "Config",
    "EvaluateConfig",
    "Result",
    "TranscribeConfig",
    "TranslateConfig",
    "evaluate",
    "evaluate_async",
    "re_render_summary",
    "render_summary_from_dataset",
]
