import atexit
import json
import random
import tempfile
import time
import logging
import re
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import io
from typing import Any

import datasets as ds
import numpy as np
import soundfile as sf
from google import genai
from tqdm import tqdm


_CLIENT_LOCAL = threading.local()
_CLIENTS = weakref.WeakSet()
_CLIENTS_LOCK = threading.Lock()


def _close_all_clients() -> None:
    """Close all cached Gemini clients.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS)
        _CLIENTS.clear()

    for client in clients:
        try:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            try:
                logger.exception("Failed to close Gemini client.")
            except Exception:
                pass


atexit.register(_close_all_clients)


def _build_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_ts": {
                            "type": "string",
                            "description": "Segment start timestamp in MM:SS.mmm format (minutes only).",
                        },
                        "end_ts": {
                            "type": "string",
                            "description": "Segment end timestamp in MM:SS.mmm format (minutes only).",
                        },
                        "text": {
                            "type": "string",
                            "description": "Segment text.",
                        },
                    },
                    "required": ["start_ts", "end_ts", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    }


LANGUAGE_NAME_MAP: dict[str, str] = {
    "en": "English",
    "ja": "Japanese",
    "es": "Spanish",
    "ko": "Korean",
    "zh": "Chinese",
}


def _normalize_lang_code(language: str) -> str:
    # "EN", "en-US", "es_ES" などを想定して正規化
    return language.strip().lower().replace("_", "-")


def _prompt_language_name(language: str) -> str:
    code = _normalize_lang_code(language)
    base = code.split("-", 1)[0]  # "en-us" -> "en"
    name = LANGUAGE_NAME_MAP.get(code) or LANGUAGE_NAME_MAP.get(base)
    if name is None:
        raise ValueError(f"Unsupported language code: {language}. Supported: {sorted(LANGUAGE_NAME_MAP)}")
    return name


def _prompt_language_base(language: str) -> str:
    return _normalize_lang_code(language).split("-", 1)[0]


def _language_formatting_rules(language: str) -> str:
    """
    言語別の表記ルールをプロンプトに付与する。
    - ja: 分かち書き禁止 + 句読点/括弧の前後スペース禁止（重点）
    - ko: ハングル音節の間にスペースを入れない（単語間の通常スペースは可）
    - en/es: 標準正書法（単語内に余計なスペースを入れない）
    """
    base = _prompt_language_base(language)

    if base == "ja":
        return (
            "\n"
            "## Japanese orthography rules (critical)\n"
            "- Do NOT insert spaces between Japanese characters or words. (No wakachigaki.)\n"
            "- Do NOT add spaces before/after Japanese punctuation or brackets: 、。！？「」『』（）［］【】・…ー\n"
            "  - BAD: `えー 、 あと こっから 先 は`\n"
            "  - GOOD: `えー、あとこっから先は`\n"
            "- If you internally generate word-separated text, remove those spaces before producing the final JSON.\n"
            "- Allow a single ASCII space only when truly needed between consecutive Latin/number tokens; avoid spaces inside Japanese sequences.\n"
        )

    if base == "ko":
        return (
            "\n"
            "## Korean orthography rules\n"
            "- Use standard Korean spacing.\n"
            "- Do NOT insert spaces between Hangul syllables inside a word (e.g., BAD: `안 녕 하 세 요`, GOOD: `안녕하세요`).\n"
            "- Do NOT add unnatural spaces around Korean punctuation.\n"
        )

    if base in ("en", "es"):
        return (
            "\n"
            "## Orthography rules\n"
            "- Use standard orthography for the language.\n"
            "- Do NOT insert spaces inside words.\n"
            "- Do NOT add unnatural spaces around punctuation beyond standard writing conventions.\n"
        )

    # 基本到達しない（LANGUAGE_NAME_MAP が限定）
    return ""


def _build_prompt(language: str) -> str:
    prompt_language = _prompt_language_name(language)

    # 重要: “Only include language” は混在音声で言語が揺れるのを抑える。
    #       表記ルールは言語別に付与して、ja の空白問題を強く抑制する。
    return (
        f"Transcribe the audio into {prompt_language} with timestamped segments.\n"
        f"- Only include language: {prompt_language}\n"
        "- Do NOT include speaker labels.\n"
        "- The audio contains mixed native and non-native speech; accuracy is critical.\n"
        "- Timestamp accuracy is extremely important for downstream slicing.\n"
        "- Segment by utterance starts and meaningful semantic units; assign timestamps to match each segment.\n"
        "- Use MM:SS.mmm timestamps (minutes only; no hours).\n"
        "- Always include milliseconds with three digits (e.g., 01:16.500).\n"
        "- Return ONLY JSON that matches the provided schema.\n"
        + _language_formatting_rules(language)
    )


def _parse_json_maybe(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


_TIMESTAMP_CHAR_TRANSLATION = str.maketrans(
    {
        "：": ":",
        "﹕": ":",
        "ː": ":",
        "．": ".",
        "。": ".",
        "，": ".",
        ",": ".",
    }
)


def _preview_text(value: Any, *, limit: int = 240) -> str:
    """Return a compact one-line preview for logs."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _parse_timestamp_to_seconds(value: str) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip().translate(_TIMESTAMP_CHAR_TRANSLATION)
    text = text.strip("[](){}<>")
    if not text:
        return None
    match = re.fullmatch(
        r"(?P<minutes>\d{1,3}):(?P<seconds>\d{1,2})"
        r"(?:(?:\.(?P<fraction_dot>\d{1,6}))|(?::(?P<fraction_colon>\d{3,6})))?",
        text,
    )
    if match is None:
        return None
    try:
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        fraction_text = (
            match.group("fraction_dot") or match.group("fraction_colon") or ""
        )
    except ValueError:
        return None
    if seconds < 0 or seconds >= 60:
        return None
    fraction = float(f"0.{fraction_text}") if fraction_text else 0.0
    total = minutes * 60.0 + seconds + fraction
    return total


def _build_timestamp_segment(
    item: dict[str, Any],
    language: str,
) -> dict[str, float | str] | None:
    """Convert one Gemini segment item into the internal timestamp format."""
    start_sec = _parse_timestamp_to_seconds(item.get("start_ts", ""))
    end_sec = _parse_timestamp_to_seconds(item.get("end_ts", ""))
    if start_sec is None or end_sec is None:
        return None

    text = item.get("text", "")
    text = _postprocess_text(text, language)
    if not text:
        return None

    return {"start": start_sec, "end": end_sec, "text": text}


# --- Text postprocess (language-aware) ---

_JA_CHARS = (
    r"\u3040-\u309F"  # Hiragana
    r"\u30A0-\u30FF"  # Katakana
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs
    r"\u3000-\u303F"  # CJK Symbols and Punctuation
)

_JA_PUNCT = r"、。！？"
_JA_BR_OPEN = r"「『（［【"
_JA_BR_CLOSE = r"」』）］】"
_JA_MID = r"・…ー"


def _postprocess_text(text: str, language: str) -> str:
    """
    生成側が分かち書き風に空白を入れてくるケースの最終防衛。
    - ja: 日本語文字同士の空白、句読点/括弧の前後空白を除去
    - その他: 過剰な連続空白を1つに（必要なら）
    """
    if not isinstance(text, str):
        return ""

    s = text.strip()
    base = _prompt_language_base(language)

    # まず連続空白を軽く畳む（タブ等も）
    s = re.sub(r"[ \t]+", " ", s)

    if base != "ja":
        return s

    # 1) 日本語文字同士の間にある空白を除去（分かち書き潰し）
    s = re.sub(fr"([{_JA_CHARS}])\s+([{_JA_CHARS}])", r"\1\2", s)

    # 2) 日本語句読点の前の空白を除去
    s = re.sub(fr"\s+([{_JA_PUNCT}])", r"\1", s)

    # 3) 括弧類: 開き括弧の後、閉じ括弧の前の空白を除去
    s = re.sub(fr"([{_JA_BR_OPEN}])\s+", r"\1", s)
    s = re.sub(fr"\s+([{_JA_BR_CLOSE}])", r"\1", s)

    # 4) 中点/三点リーダ/長音などの周辺空白も除去（必要なら）
    s = re.sub(fr"\s+([{_JA_MID}])", r"\1", s)
    s = re.sub(fr"([{_JA_MID}])\s+", r"\1", s)

    # 5) もう一度、日本語文字同士の空白が残っていれば除去（繰り返し安定化）
    #    例: "本 当 に" -> 1回目で "本当 に" になり、2回目で "本当に"
    for _ in range(2):
        new_s = re.sub(fr"([{_JA_CHARS}])\s+([{_JA_CHARS}])", r"\1\2", s)
        if new_s == s:
            break
        s = new_s

    # 6) 最後に余計な連続スペースを畳む
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _decode_audio_dict(audio_dict: Any) -> tuple[np.ndarray, int]:
    """
    入力が以下のいずれでも動くようにする:
      - dict: {"array": ..., "sampling_rate": ...} または {"byte": ..., "sampling_rate": ...?}
      - AudioDecoder (torchcodec.decoders.AudioDecoder): datasets が返す音声オブジェクト
          - subscript不可なので "in" や audio_dict["..."] を使わない
          - 属性/メソッド経由で取り出す（環境差があるのでフォールバック複数）
    """
    if isinstance(audio_dict, dict):
        if "bytes" in audio_dict:
            raw = audio_dict["bytes"]
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise TypeError(f"audio_dict['bytes'] must be bytes-like, got: {type(raw)}")

            sr_hint = audio_dict.get("sampling_rate", None)
            if sr_hint is not None:
                sr_hint = int(sr_hint)

            audio, sample_rate = _decode_audio_bytes_with_soundfile(bytes(raw))

            if sr_hint is not None and sample_rate != sr_hint:
                raise ValueError(f"sampling_rate mismatch: hint={sr_hint}, decoded={sample_rate}")
            return audio, sample_rate

        raise KeyError("dict input must contain 'bytes' key.")

    # datasets の Audio 特有オブジェクトは環境差があるが、とりあえず subscript で取れるケースに合わせる
    audio = np.array(audio_dict["array"], dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    sample_rate = int(audio_dict["sampling_rate"])
    return audio, sample_rate


def _decode_audio_bytes_with_soundfile(raw: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(raw) as bio:
        audio, sample_rate = sf.read(bio, dtype="float32", always_2d=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def _describe_audio_input(audio_dict: Any) -> str:
    """Return a short stable audio label for logging."""
    if isinstance(audio_dict, dict):
        path = audio_dict.get("path")
        if isinstance(path, str) and path:
            return path
        raw = audio_dict.get("bytes")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return f"bytes:{len(raw)}"
        return f"dict_keys={sorted(audio_dict.keys())}"
    return type(audio_dict).__name__


def _normalize_timeout_seconds(request_timeout_seconds: float | None) -> float | None:
    """Normalize timeout seconds for Gemini client creation.

    Parameters
    ----------
    request_timeout_seconds : float | None
        Requested timeout in seconds. If None, invalid, or non-positive, the
        client default timeout is used.

    Returns
    -------
    float | None
        Normalized positive timeout value, or None if default should be used.
    """
    if request_timeout_seconds is None:
        return None
    try:
        timeout_value = float(request_timeout_seconds)
    except (TypeError, ValueError):
        return None
    if timeout_value <= 0:
        return None
    return timeout_value


def _extract_uploaded_file_name(uploaded: Any) -> str | None:
    """Extract uploaded file name from Gemini Files API response.

    Parameters
    ----------
    uploaded : Any
        Uploaded file object or dict returned by Files API.

    Returns
    -------
    str | None
        File name string if available, otherwise None.
    """
    if uploaded is None:
        return None

    if hasattr(uploaded, "name"):
        name = getattr(uploaded, "name")
        if isinstance(name, str) and name:
            return name

    if isinstance(uploaded, dict):
        name = uploaded.get("name")
        if isinstance(name, str) and name:
            return name

    return None


def _is_retryable_delete_error(e: Exception) -> bool:
    """Heuristically determine if a delete error is retryable.

    Parameters
    ----------
    e : Exception
        Exception raised by Files API delete call.

    Returns
    -------
    bool
        True if the error looks retryable (429/5xx/timeout), otherwise False.
    """
    msg = str(e)
    retry_markers = [
        "429",
        "Too Many Requests",
        "RESOURCE_EXHAUSTED",
        "Resource has been exhausted",
        "500",
        "502",
        "503",
        "504",
        "Deadline",
        "timeout",
        "Timed out",
        "Connection reset",
        "Connection aborted",
        "Temporary failure",
    ]
    return any(marker in msg for marker in retry_markers)


def _delete_uploaded_file(
    client: genai.Client,
    uploaded: Any,
    logger: logging.Logger,
    retries: int = 2,
    backoff_initial: float = 0.5,
    backoff_max: float = 4.0,
) -> None:
    """Best-effort delete of uploaded file with small retries.

    Parameters
    ----------
    client : genai.Client
        Gemini client instance.
    uploaded : Any
        Uploaded file object or dict returned by Files API.
    logger : logging.Logger
        Logger instance for warnings.
    retries : int, default=2
        Number of retries for retryable errors.
    backoff_initial : float, default=0.5
        Initial backoff seconds.
    backoff_max : float, default=4.0
        Maximum backoff seconds.

    Returns
    -------
    None
    """
    name = _extract_uploaded_file_name(uploaded)
    if not name:
        logger.warning("Skip deleting Gemini file: missing file name.")
        return

    delay = backoff_initial
    for attempt in range(retries + 1):
        try:
            client.files.delete(name=name)
            return
        except Exception as e:
            if attempt >= retries or not _is_retryable_delete_error(e):
                logger.warning(
                    "Failed to delete Gemini file: %s", name, exc_info=True
                )
                return

            sleep_s = min(backoff_max, delay) * (0.5 + random.random())
            time.sleep(sleep_s)
            delay = min(backoff_max, delay * 2)


def get_client(request_timeout_seconds: float | None) -> genai.Client:
    """Get or create a thread-local Gemini client.

    Parameters
    ----------
    request_timeout_seconds : float | None
        Timeout in seconds for a single HTTP request. If None or non-positive,
        the client default is used.

    Returns
    -------
    genai.Client
        Gemini client instance.
    """
    timeout_value = _normalize_timeout_seconds(request_timeout_seconds)
    client = getattr(_CLIENT_LOCAL, "client", None)
    client_timeout = getattr(_CLIENT_LOCAL, "timeout_value", None)

    if client is None or client_timeout != timeout_value:
        client = _create_genai_client(timeout_value)
        _CLIENT_LOCAL.client = client
        _CLIENT_LOCAL.timeout_value = timeout_value
        with _CLIENTS_LOCK:
            try:
                _CLIENTS.add(client)
            except TypeError:
                # Some client implementations may not support weak references.
                pass

        if os.getenv("GEMINI_DEBUG_CLIENT_REUSE") == "1":
            logger = logging.getLogger(__name__)
            thread = threading.current_thread()
            logger.info(
                "Gemini client created in thread=%s (id=%s, timeout=%s)",
                thread.name,
                thread.ident,
                timeout_value,
            )

    return client


def _create_genai_client(request_timeout_seconds: float | None) -> genai.Client:
    """Create a Gemini client with an optional HTTP timeout.

    Parameters
    ----------
    request_timeout_seconds : float | None
        Timeout in seconds for a single HTTP request. If None or non-positive,
        the client default is used.

    Returns
    -------
    genai.Client
        Gemini client instance.
    """
    timeout_value = _normalize_timeout_seconds(request_timeout_seconds)

    if timeout_value is None:
        return genai.Client()

    try:
        timeout_ms = int(timeout_value * 1000)
        return genai.Client(http_options={"timeout": timeout_ms})
    except TypeError:
        # Fallback for older client signatures.
        return genai.Client()


def _transcribe_audio_dict(
    audio_dict: dict,
    model: str,
    language: str,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str | None,
    request_timeout_seconds: float | None,
    max_retries: int,
    retry_base_seconds: float,
    debug_label: str | None = None,
) -> list[dict]:
    logger = logging.getLogger(__name__)
    audio, sample_rate = _decode_audio_dict(audio_dict)
    prompt = _build_prompt(language)
    schema = _build_schema()

    for attempt in range(1, max_retries + 2):
        client: genai.Client | None = None
        uploaded: Any | None = None
        try:
            client = get_client(request_timeout_seconds)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio, sample_rate)
                uploaded = client.files.upload(file=tmp.name)

            response = client.models.generate_content(
                model=model,
                contents=[prompt, uploaded],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    **(
                        {"thinking_config": {"thinking_level": thinking_level}}
                        if thinking_level
                        else {}
                    ),
                },
            )

            response_text = response.text or ""
            data = (
                response.parsed
                if getattr(response, "parsed", None) is not None
                else None
            )
            if data is None:
                data = _parse_json_maybe(response_text)
            if not isinstance(data, dict):
                raise ValueError(
                    "Failed to parse Gemini JSON response. "
                    f"response_preview={_preview_text(response_text)!r}"
                )

            segments_raw = data.get("segments", [])
            if not isinstance(segments_raw, list):
                raise ValueError(
                    "Gemini JSON response is missing a valid segments list. "
                    f"response_preview={_preview_text(response_text)!r}"
                )
            segments: list[dict] = []
            discarded_segments = 0
            bad_segment_examples: list[str] = []
            for item in segments_raw:
                segment = _build_timestamp_segment(item, language)
                if segment is None:
                    discarded_segments += 1
                    if len(bad_segment_examples) < 3:
                        bad_segment_examples.append(_preview_text(item, limit=160))
                    continue
                segments.append(segment)

            if not segments:
                raise ValueError(
                    "No valid timestamps parsed from Gemini response. "
                    f"segments_raw={len(segments_raw)} "
                    f"discarded={discarded_segments} "
                    f"sample_bad_segments={bad_segment_examples!r} "
                    f"response_preview={_preview_text(response_text)!r}"
                )
            return sorted(segments, key=lambda w: (w["start"], w["end"]))
        except Exception as exc:
            if attempt > max_retries:
                logger.warning(
                    "Gemini transcription failed after %d attempts for %s: %s",
                    max_retries + 1,
                    debug_label or language,
                    exc,
                    exc_info=True,
                )
                break
            logger.warning(
                "Gemini transcription attempt %d/%d failed for %s: %s",
                attempt,
                max_retries + 1,
                debug_label or language,
                exc,
            )
            delay = retry_base_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0.0, 0.25 * delay)
            time.sleep(delay)
        finally:
            if client is not None and uploaded is not None:
                _delete_uploaded_file(client, uploaded, logger)

    return []


def _transcribe_one_index(index: int, example: dict, config: dict) -> tuple[int, list[dict]]:
    logger = logging.getLogger(__name__)
    audio_dict = example[config["audio_column"]]
    debug_label = (
        f"row={index} lang={config['lang']} "
        f"audio_column={config['audio_column']} "
        f"audio={_describe_audio_input(audio_dict)}"
    )
    timestamps = _transcribe_audio_dict(
        audio_dict=audio_dict,
        model=config["model"],
        language=config["lang"],
        temperature=config["temperature"],
        max_output_tokens=config["max_output_tokens"],
        thinking_level=config["thinking_level"],
        request_timeout_seconds=config["request_timeout_seconds"],
        max_retries=config["max_retries"],
        retry_base_seconds=config["retry_base_seconds"],
        debug_label=debug_label,
    )
    if not timestamps:
        logger.warning("Gemini transcription returned no timestamps for %s", debug_label)
    return index, timestamps


def transcribe_dataset_with_gemini(
    *,
    input_dataset: ds.Dataset,
    input_audio_column: str,
    input_lang: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 8192 * 16,
    thinking_level: str | None = None,
    request_timeout_seconds: float | None = 300.0,
    max_retries: int = 3,
    retry_base_seconds: float = 1.0,
    max_workers: int = 32,
    show_progress: bool = True,
) -> ds.Dataset:
    logger = logging.getLogger(__name__)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
    logger.setLevel(logging.INFO)
    for noisy_logger in ("httpx", "httpcore", "google.api_core", "google.api_core.bidi", "google_genai.models"):
        external_logger = logging.getLogger(noisy_logger)
        if external_logger.level == logging.NOTSET:
            external_logger.setLevel(logging.WARNING)

    column_name = f"timestamps_{input_lang}"
    config = {
        "lang": input_lang,
        "audio_column": input_audio_column,
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "thinking_level": thinking_level,
        "request_timeout_seconds": request_timeout_seconds,
        "max_retries": max_retries,
        "retry_base_seconds": retry_base_seconds,
        "column_name": column_name,
    }
    if column_name in input_dataset.column_names:
        print(f"WARNING: overwriting existing column: {column_name}")
        input_dataset = input_dataset.remove_columns(column_name)

    results: list[list[dict]] = [[] for _ in range(len(input_dataset))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_transcribe_one_index, index, input_dataset[index], config): index
            for index in range(len(input_dataset))
        }
        total = len(futures)
        if total == 0:
            return input_dataset.add_column(column_name, results)

        logger.info("Gemini transcribe started: total=%d", total)
        completed = 0
        start_time = time.monotonic()
        last_log_time = start_time
        log_every = max(1, total // 20)
        min_log_interval = 30.0

        for future in tqdm(
            as_completed(futures),
            total=total,
            desc=f"transcribe ({input_lang})",
            unit="file",
            disable=not show_progress,
        ):
            index, timestamps = future.result()
            results[index] = timestamps
            completed += 1
            now = time.monotonic()
            # When the bar is shown it carries progress; otherwise fall back to
            # throttled log lines so non-TTY runs (e.g. Slurm) still report progress.
            should_log = not show_progress and (
                completed == total
                or completed % log_every == 0
                or (now - last_log_time) >= min_log_interval
            )
            if should_log:
                elapsed = now - start_time
                pct = (completed / total) * 100.0
                logger.info(
                    "Gemini transcribe progress: %d/%d (%.1f%%, elapsed %.1fs)",
                    completed, total, pct, elapsed,
                )
                last_log_time = now

    rows_with_text = 0
    total_segments = 0
    nonempty_segments = 0
    failed_indices: list[int] = []
    for index, timestamps in enumerate(results):
        total_segments += len(timestamps)
        row_nonempty_segments = sum(
            1
            for segment in timestamps
            if isinstance(segment, dict)
            and isinstance(segment.get("text"), str)
            and segment["text"].strip()
        )
        nonempty_segments += row_nonempty_segments
        if row_nonempty_segments > 0:
            rows_with_text += 1
        else:
            failed_indices.append(index)
    logger.info(
        "Gemini transcribe summary for %s: rows_with_text=%d/%d "
        "total_segments=%d nonempty_segments=%d",
        column_name,
        rows_with_text,
        len(input_dataset),
        total_segments,
        nonempty_segments,
    )
    if len(input_dataset) > 0 and rows_with_text == 0:
        logger.warning(
            "Gemini transcribe produced zero non-empty rows for %s "
            "(audio_column=%s, lang=%s); first_failed_indices=%s",
            column_name,
            input_audio_column,
            input_lang,
            failed_indices[:10],
        )

    return input_dataset.add_column(column_name, results)


