# Metrics

The pipeline scores translation quality at two levels: **per chunk** (LLM-judged) and **per row** (means over chunks). Latency is measured separately at the chunk level.

## Per-chunk scores (LLM-judged)

Source + target audio are transcribed (Gemini), aligned into matched chunks (OpenAI), then each chunk pair gets three 0/1 scores from OpenAI:

| Score | What it means |
|---|---|
| `accuracy` | The target segment correctly conveys the source segment's meaning. |
| `fluency` | The target segment reads naturally in the target language. |
| `conciseness` | The target segment doesn't add or omit meaningful information. |

A chunk's output looks like:

```json
{
  "en": "what is the weather today",
  "ja": "今日の天気はどうですか",
  "accuracy": 1,
  "fluency": 1,
  "conciseness": 1
}
```

## Per-row means

For each clip, the chunk-level scores are averaged into `row_accuracy_mean`, `row_fluency_mean`, `row_conciseness_mean`. The overall report means are the means of those row means (each clip contributes equally regardless of how many chunks it has).

## Chunk aggregation modes

The summary reports three chunk-aggregation modes — partner tooling may prefer different denominators:

| Mode | Denominator | Use case |
|---|---|---|
| `non-empty chunks (incl. unk as 0)` | All non-empty chunks; unknown values counted as 0. | Strict scoring. |
| `non-empty, non-unk chunks` | Chunks with a real 0/1 score. | Quality-of-judgment-only. |
| `non-unk chunks (empty=0)` | Real-or-empty chunks (empty counted as 0). | Penalizes the model for empty alignment chunks. |

## Latency

Two latency metrics surface in the summary:

- **`median_first_chunk_latency_translate` (seconds)** — for each input clip, time from "audio commit sent" to "first translated audio chunk received". This is the user-visible "first translation appears" latency. Median across clips.

- **`median_latency_chunk` (seconds)** — within each translated clip, for each aligned chunk that scored `accuracy=1`, the difference `target_start_time − source_start_time` (where target/source start times come from Gemini transcript timestamps mapped back to the chunk boundaries). Median across all such chunks. This is the simultaneous-translation lag metric.

Chunks with `accuracy=0` are excluded from `median_latency_chunk` — bad translations would corrupt the lag signal.

## Coverage diagnostics

Alongside scores, the summary reports:

- `rows_total`, `rows_with_output` — clips processed vs. successfully scored end-to-end.
- `translate_ok`, `translate_failed` — STS stage outcomes.
- `expected_chunks_total`, `output_chunks_total`, `empty_chunks_total`, `empty_chunk_ratio` — how many chunks the LLM filled vs. left empty (a high empty ratio indicates poor alignment).

If `unk_chunks_total > 0`, OpenAI returned malformed scores for some chunks — usually safe to ignore at small scale.
