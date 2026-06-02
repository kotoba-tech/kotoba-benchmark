import hashlib
import json
import threading
import time
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from json_repair import repair_json
from tqdm import tqdm

from kotoba_benchmark._llm import LLMError, make_llm_client
from kotoba_benchmark._vendored._align_prompt_builder import PromptBuilder


def save_json(data: dict, path: Path):
    """Save dictionary to JSON file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    tmp_path.replace(path)


def normalize_text(text: str) -> str:
    """Normalize text for strict-but-robust alignment validation."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        c
        for c in text
        if c not in ",.!?;:!()'-、。 「」『』【】〔〕［］〈〉《》〔〕｛｝" + " "
    )
    return text.casefold()


def extract_alignment_text(value) -> str:
    """Extract plain concatenated text from one alignment input."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError(
            "Alignment input must be a string or a list of {start,end,text} items."
        )

    parts: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(
                f"Alignment input item {idx} must be a dict, got {type(item).__name__}"
            )
        text = item.get("text")
        if text is None and "word" in item:
            text = item.get("word")
        if not isinstance(text, str):
            raise ValueError(
                "Alignment input items must include text or word as a string."
            )
        parts.append(text)
    return "".join(parts)


def serialize_alignment_input(value) -> str:
    """Serialize one alignment input into a stable cache-key string."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError(
            "Alignment input must be a string or a list of {start,end,text} items."
        )

    normalized: list[dict[str, float | str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(
                f"Alignment input item {idx} must be a dict, got {type(item).__name__}"
            )
        text = item.get("text")
        if text is None and "word" in item:
            text = item.get("word")
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None or not isinstance(text, str):
            raise ValueError(
                "Alignment input items must include start, end, and text/word."
            )
        normalized.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def preview_alignment_input(value, *, limit: int = 50) -> str:
    """Build a short preview string for logs."""
    text = extract_alignment_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def classify_openai_error_reason(error: Exception) -> str:
    """Classify an OpenAI API exception into a compact reason code.

    Parameters
    ----------
    error : Exception
        Exception raised by OpenAI SDK.

    Returns
    -------
    str
        Machine-readable reason code for alignment failure logs.
    """
    error_name = error.__class__.__name__.lower()
    message = str(error).lower()
    if "timeout" in error_name or "timed out" in message or "timeout" in message:
        return "openai_timeout"
    if "rate" in message and "limit" in message:
        return "openai_rate_limit"
    if "connection" in message:
        return "openai_connection_error"
    if "authentication" in message or "api key" in message:
        return "openai_auth_error"
    return f"openai_error_{error.__class__.__name__}"


class RateLimiter:
    """Rate limiter for realtime API calls."""

    def __init__(self, rps_limit: int):
        """
        Initialize rate limiter.

        Parameters
        ----------
        rps_limit : int
            Maximum number of requests per second.
            Non-positive values disable rate limiting.
        """
        self.rps_limit = rps_limit
        self.request_timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """
        Acquire one request slot.

        Returns
        -------
        None
            Returns when the caller can safely issue one API request.
        """
        if self.rps_limit <= 0:
            return

        while True:
            with self._lock:
                current_time = time.monotonic()
                while (
                    self.request_timestamps
                    and current_time - self.request_timestamps[0] > 1.0
                ):
                    self.request_timestamps.popleft()

                if len(self.request_timestamps) < self.rps_limit:
                    self.request_timestamps.append(current_time)
                    return

                earliest_timestamp = self.request_timestamps[0]
                sleep_time = 1.0 - (current_time - earliest_timestamp)

            time.sleep(max(sleep_time, 0.0))


class TextAlignmentManager:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        cache_dir: Path | str = None,
        use_cache: bool = True,
        max_workers: int = 500,
        max_retries: int = 3,
        model_name: str = "gpt-5",
        rps_limit: int = 50,
        request_timeout_sec: int = 120,
        # model_name: str = "gpt-4.1-mini",
        # model_name: str = "gpt-4.1",
        prompt_path: Path | str = "src/align_segments/prompt/v1.0.toml",
        show_progress: bool = True,
    ):
        self.prompt_path = Path(prompt_path)

        self.llm = make_llm_client(model_name)
        self.max_workers = max_workers
        self.max_worker = max_workers  # Backward-compatibility
        self.rps_limit = rps_limit
        self.request_timeout_sec = request_timeout_sec
        self.show_progress = show_progress

        self.source_lang = source_lang
        self.target_lang = target_lang

        self.model_name = model_name
        self.max_retries = max_retries
        self.use_cache = use_cache

        self.prompt_builder = PromptBuilder(
            source_lang=source_lang,
            target_lang=target_lang,
            prompt_path=prompt_path,
        )

        self.version = self.prompt_path.stem
        if cache_dir is None:
            cache_dir = Path(f".cache/align_segments/{self.version}/{self.model_name}")
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.failure_reason_by_key: dict[str, str] = {}
        self._failure_reason_lock = threading.Lock()

    def _set_failure_reason(
        self,
        source_input,
        target_input,
        reason: str,
    ) -> None:
        """Store failure reason for one text pair.

        Parameters
        ----------
        source_input
            Source alignment input.
        target_input
            Target alignment input.
        reason : str
            Reason code.

        Returns
        -------
        None
            This function updates the in-memory failure-reason map.
        """
        cache_key = self._build_cache_key(source_input, target_input)
        with self._failure_reason_lock:
            self.failure_reason_by_key[cache_key] = reason

    def _clear_failure_reason(
        self,
        source_input,
        target_input,
    ) -> None:
        """Remove stale failure reason after a successful request.

        Parameters
        ----------
        source_input
            Source alignment input.
        target_input
            Target alignment input.

        Returns
        -------
        None
            This function updates the in-memory failure-reason map.
        """
        cache_key = self._build_cache_key(source_input, target_input)
        with self._failure_reason_lock:
            self.failure_reason_by_key.pop(cache_key, None)

    def parse_output(self, result_text: str) -> dict:
        """Clean up common formatting issues in API responses."""
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = repair_json(result_text.strip())
        return json.loads(result_text)

    def _build_cache_key(self, source_input, target_input) -> str:
        """
        Build deterministic cache key from prompt and input texts.

        Parameters
        ----------
        source_input
            Source alignment input.
        target_input
            Target alignment input.

        Returns
        -------
        str
            Cache key for this text pair.
        """
        serialized_source = serialize_alignment_input(source_input)
        serialized_target = serialize_alignment_input(target_input)
        combined_text = (
            f"model:{self.model_name}\n"
            f"{self.prompt_builder.prompt_template}\n"
            f"{self.source_lang}:{serialized_source}|{self.target_lang}:{serialized_target}"
        )
        return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()[:16]

    def _get_cache_path(self, source_input, target_input) -> Path:
        """
        Get cache file path for a text pair.

        Parameters
        ----------
        source_input
            Source alignment input.
        target_input
            Target alignment input.

        Returns
        -------
        pathlib.Path
            Cache file path.
        """
        key = self._build_cache_key(source_input, target_input)
        return self.cache_dir / f"{key}.json"

    def _load_cached_alignment(
        self,
        cache_path: Path,
    ) -> tuple[list[str], list[str]] | None:
        """
        Load cached alignment if possible.

        Parameters
        ----------
        cache_path : pathlib.Path
            Cache file path.

        Returns
        -------
        tuple[list[str], list[str]] | None
            Cached source/target segments, or ``None`` when unavailable/invalid.
        """
        if not self.use_cache:
            return None
        if not cache_path.exists():
            return None

        try:
            with cache_path.open(encoding="utf-8") as f:
                cached_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading cache `{cache_path}`: {e}")
            return None

        source_segments = cached_data.get("source_segments")
        target_segments = cached_data.get("target_segments")
        if not isinstance(source_segments, list) or not isinstance(target_segments, list):
            return None
        if not all(isinstance(seg, str) for seg in source_segments):
            return None
        if not all(isinstance(seg, str) for seg in target_segments):
            return None

        return source_segments, target_segments

    def _save_alignment_cache(
        self,
        cache_path: Path,
        source_input=None,
        target_input=None,
        source_segments: list[str] | None = None,
        target_segments: list[str] | None = None,
        *,
        source_text=None,
        target_text=None,
    ) -> None:
        """
        Save aligned segments into cache.

        Parameters
        ----------
        cache_path : pathlib.Path
            Cache file path.
        source_input
            Source alignment input.
        target_input
            Target alignment input.
        source_segments : list[str]
            Segmented source text.
        target_segments : list[str]
            Segmented target text.

        Returns
        -------
        None
            This function only writes cache as a side effect.
        """
        if source_input is None:
            source_input = source_text
        if target_input is None:
            target_input = target_text
        if source_input is None or target_input is None:
            raise TypeError("source_input/target_input or source_text/target_text is required")
        if source_segments is None or target_segments is None:
            raise TypeError("source_segments and target_segments are required")

        source_text = extract_alignment_text(source_input)
        target_text = extract_alignment_text(target_input)
        if not self.use_cache:
            return
        cache_data = {
            "source_segments": source_segments,
            "target_segments": target_segments,
            "source_text": source_text,
            "target_text": target_text,
            "source_input": source_input,
            "target_input": target_input,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }
        save_json(cache_data, cache_path)

    def _extract_alignment_result_list(
        self,
        result_json,
        source_input,
        target_input,
    ) -> list[dict] | None:
        """Extract the top-level alignment list from one model response."""
        alignment_result: list[dict] | None = None
        if isinstance(result_json, dict):
            if "alignments" in result_json:
                alignment_result = result_json["alignments"]
            else:
                for key, value in result_json.items():
                    if not isinstance(key, str):
                        continue
                    normalized = key.strip().strip('"').strip("'")
                    if normalized == "alignments":
                        alignment_result = value
                        break
        elif isinstance(result_json, list):
            alignment_result = result_json

        if not isinstance(alignment_result, list):
            self._set_failure_reason(source_input, target_input, "missing_alignments")
            raise KeyError("alignments")
        return alignment_result

    def _parse_text_segments_from_result(
        self,
        alignment_result: list[dict],
        source_text: str,
        target_text: str,
        source_input,
        target_input,
    ) -> tuple[list[str], list[str]] | None:
        """Parse legacy raw-text alignment items."""
        source_segments: list[str] = []
        target_segments: list[str] = []
        for item in alignment_result:
            if not isinstance(item, dict):
                self._set_failure_reason(
                    source_input,
                    target_input,
                    "alignment_item_not_dict",
                )
                return None
            source_value = item.get("source_segment", "")
            target_value = item.get("target_segment", "")
            if source_value is None:
                source_value = ""
            if target_value is None:
                target_value = ""
            if not isinstance(source_value, str) or not isinstance(target_value, str):
                self._set_failure_reason(source_input, target_input, "segment_not_string")
                return None
            source_segments.append(source_value)
            target_segments.append(target_value)

        if normalize_text("".join(source_segments)) != normalize_text(source_text):
            self._set_failure_reason(source_input, target_input, "source_text_mismatch")
            return None
        if normalize_text("".join(target_segments)) != normalize_text(target_text):
            self._set_failure_reason(source_input, target_input, "target_text_mismatch")
            return None

        if not any(seg.strip() for seg in source_segments):
            self._set_failure_reason(
                source_input,
                target_input,
                "all_source_segments_empty",
            )
            return None
        if not any(seg.strip() for seg in target_segments):
            self._set_failure_reason(
                source_input,
                target_input,
                "all_target_segments_empty",
            )
            return None

        return source_segments, target_segments

    def call_openai_api_for_text_alignments(
        self,
        source_input,
        target_input,
    ) -> tuple[list[str], list[str]] | None:
        """Align source and target texts using OpenAI API."""
        source_text = extract_alignment_text(source_input)
        target_text = extract_alignment_text(target_input)
        prompt = self.prompt_builder.build_prompt(
            source_input=source_input,
            target_input=target_input,
        )
        messages = prompt.messages

        try:
            message_content = self.llm.chat_complete(
                messages,
                model=self.model_name,
                temperature=1.0,
                timeout=self.request_timeout_sec,
                expect_json=True,
            )
        except LLMError as e:
            self._set_failure_reason(source_input, target_input, e.reason)
            print(f"Error calling LLM API ({e.reason}): {e}")
            return None

        try:
            result_json = self.parse_output(message_content)
        except Exception as e:
            self._set_failure_reason(
                source_input,
                target_input,
                f"invalid_json_{e.__class__.__name__}",
            )
            raise
        alignment_result = self._extract_alignment_result_list(
            result_json,
            source_input,
            target_input,
        )
        parsed = self._parse_text_segments_from_result(
            alignment_result=alignment_result,
            source_text=source_text,
            target_text=target_text,
            source_input=source_input,
            target_input=target_input,
        )
        if parsed is None:
            return None
        source_segments, target_segments = parsed

        self._clear_failure_reason(source_input, target_input)
        return source_segments, target_segments

    def _align_with_retries(
        self,
        source_input,
        target_input,
    ) -> tuple[list[str], list[str]] | None:
        """
        Request alignments with retry and best-effort fallback.

        Parameters
        ----------
        source_input
            Source alignment input.
        target_input
            Target alignment input.

        Returns
        -------
        tuple[list[str], list[str]] | None
            Alignment result when successful; otherwise ``None``.
        """
        result: tuple[list[str], list[str]] | None = None
        best_result: tuple[list[str], list[str]] | None = None
        best_diff: int | None = None
        best_count: int | None = None

        for _ in range(self.max_retries):
            try:
                current = self.call_openai_api_for_text_alignments(
                    source_input,
                    target_input,
                )
                if current is None:
                    continue

                source_segments, target_segments = current
                source_non_empty = sum(1 for seg in source_segments if seg.strip())
                target_non_empty = sum(1 for seg in target_segments if seg.strip())
                diff = abs(source_non_empty - target_non_empty)
                current_count = max(source_non_empty, target_non_empty)

                if diff == 0:
                    result = current
                    break

                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_result = current
                    best_count = current_count
                elif diff == best_diff and (
                    best_count is None or current_count > best_count
                ):
                    best_result = current
                    best_count = current_count
            except Exception as e:
                print(f"Error aligning texts: {e}. Retrying...")

        if result is None and best_result is not None:
            result = best_result

        return result

    def _resolve_max_workers(self, num_requests: int) -> int:
        """
        Resolve worker count for realtime batch execution.

        Parameters
        ----------
        num_requests : int
            Number of pending requests.

        Returns
        -------
        int
            Worker count to use for `ThreadPoolExecutor`.
        """
        if num_requests <= 0:
            return 1
        if self.max_workers <= 0:
            return num_requests
        return min(self.max_workers, num_requests)

    def get_aligned_segments(
        self,
        source_input,
        target_input,
    ) -> tuple[list[str], list[str]]:
        cache_path = self._get_cache_path(source_input, target_input)
        cached = self._load_cached_alignment(cache_path)
        if cached is not None:
            return cached

        result = self._align_with_retries(source_input, target_input)
        if result is None:
            print(
                "Failed to align texts after multiple attempts: "
                f"{preview_alignment_input(source_input)}..."
            )
            return [], []

        source_segments, target_segments = result
        self._save_alignment_cache(
            cache_path=cache_path,
            source_input=source_input,
            target_input=target_input,
            source_segments=source_segments,
            target_segments=target_segments,
        )
        return source_segments, target_segments

    def get_aligned_segments_in_parallel(
        self,
        source_inputs: list,
        target_inputs: list,
    ) -> tuple[list[list[str] | None], list[list[str] | None]]:
        if len(source_inputs) != len(target_inputs):
            raise ValueError(
                f"source_inputs and target_inputs must have the same length: "
                f"{len(source_inputs)} != {len(target_inputs)}"
            )

        source_segments_list: list[list[str] | None] = [None] * len(source_inputs)
        target_segments_list: list[list[str] | None] = [None] * len(source_inputs)

        pending_requests: dict[str, tuple[object, object, Path]] = {}
        key_to_indices: dict[str, list[int]] = {}

        for idx, (source_input, target_input) in enumerate(zip(source_inputs, target_inputs)):
            cache_path = self._get_cache_path(source_input, target_input)
            if self.use_cache:
                cached = self._load_cached_alignment(cache_path)
                if cached is not None:
                    source_segments_list[idx] = list(cached[0])
                    target_segments_list[idx] = list(cached[1])
                    continue

                key = cache_path.stem
                key_to_indices.setdefault(key, []).append(idx)
                if key not in pending_requests:
                    pending_requests[key] = (source_input, target_input, cache_path)
                continue

            key = str(idx)
            key_to_indices[key] = [idx]
            pending_requests[key] = (source_input, target_input, cache_path)

        if not pending_requests:
            return source_segments_list, target_segments_list

        rate_limiter = RateLimiter(rps_limit=self.rps_limit)

        def worker(
            key: str,
            source_input,
            target_input,
            cache_path: Path,
        ) -> tuple[str, tuple[list[str], list[str]] | None]:
            """
            Process one unique alignment request in realtime mode.

            Parameters
            ----------
            key : str
                Cache key for this request.
            source_input
                Source alignment input.
            target_input
                Target alignment input.
            cache_path : pathlib.Path
                Cache file path.

            Returns
            -------
            tuple[str, tuple[list[str], list[str]] | None]
                Cache key and alignment result.
            """
            rate_limiter.acquire()
            result = self._align_with_retries(source_input, target_input)
            if result is None:
                return key, None

            source_segments, target_segments = result
            self._save_alignment_cache(
                cache_path=cache_path,
                source_input=source_input,
                target_input=target_input,
                source_segments=source_segments,
                target_segments=target_segments,
            )
            return key, result

        max_workers = self._resolve_max_workers(len(pending_requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, key, source_input, target_input, cache_path): key
                for key, (source_input, target_input, cache_path) in pending_requests.items()
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="align",
                disable=not self.show_progress,
            ):
                key = futures[future]
                try:
                    _, result = future.result()
                except Exception as e:
                    print(f"Error processing text pair: {e}")
                    result = None

                if result is None:
                    source_segments, target_segments = [], []
                else:
                    source_segments, target_segments = result

                for idx in key_to_indices[key]:
                    source_segments_list[idx] = list(source_segments)
                    target_segments_list[idx] = list(target_segments)

        return source_segments_list, target_segments_list
