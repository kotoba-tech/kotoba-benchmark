import hashlib
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from filelock import FileLock
from json_repair import repair_json
from tqdm import tqdm

from kotoba_benchmark._llm import LLMError, make_llm_client
from kotoba_benchmark._vendored._eval_prompt_builder import PromptBuilder


def save_json(data: dict, path: Path):
    """Save dictionary to JSON file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(str(lock_path), timeout=10):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        tmp_path.replace(path)


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


class TranslationEvaluationManager:
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
        request_timeout_sec: int = 60,
        # model_name: str = "gpt-5-mini",
        version: str = "v1.1",
        prompt_path: Path | str = None,
        show_progress: bool = True,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.show_progress = show_progress

        if prompt_path is None:
            prompt_path = f"src/translation_evaluation/prompt/{source_lang}2{target_lang}-{version}.toml"

        self.llm = make_llm_client(model_name)
        self.model_name = model_name
        self.max_workers = max_workers
        self.rps_limit = rps_limit
        self.request_timeout_sec = request_timeout_sec
        self.use_cache = use_cache

        if isinstance(prompt_path, str):
            prompt_path = prompt_path.format(source_lang=source_lang, target_lang=target_lang)
        self.prompt_path = Path(prompt_path)

        self.max_retries = max_retries
        self.version = self.prompt_path.stem
        if cache_dir is None:
            cache_dir = Path(
                f".cache/translation_evaluation/{self.version}/{self.model_name}"
            )

        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_builder = PromptBuilder(prompt_path=self.prompt_path)
        self.source_lang = self.prompt_builder.source_lang
        self.target_lang = self.prompt_builder.target_lang

    def parse_output(self, result_text: str) -> dict | list:
        """Clean up common formatting issues in API responses."""
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = repair_json(result_text.strip())
        return json.loads(result_text)

    def _clone_outputs(self, outputs: list[dict]) -> list[dict]:
        """
        Clone output list to avoid shared mutable references.

        Parameters
        ----------
        outputs : list[dict]
            Evaluation outputs.

        Returns
        -------
        list[dict]
            Shallow-copied outputs.
        """
        return [dict(output) for output in outputs]

    def _build_cache_key(
        self,
        source_segments: list[str],
        target_segments: list[str],
    ) -> str:
        """
        Build deterministic cache key from prompt and input segments.

        Parameters
        ----------
        source_segments : list[str]
            Source text segments.
        target_segments : list[str]
            Target text segments.

        Returns
        -------
        str
            Cache key for this segment pair.
        """
        payload = {
            "model_name": self.model_name,
            "prompt_template": self.prompt_builder.prompt_template,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "source_segments": source_segments,
            "target_segments": target_segments,
        }
        key_material = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]

    def _get_cache_path(
        self,
        source_segments: list[str],
        target_segments: list[str],
    ) -> Path:
        """
        Get cache file path for segment pair.

        Parameters
        ----------
        source_segments : list[str]
            Source text segments.
        target_segments : list[str]
            Target text segments.

        Returns
        -------
        pathlib.Path
            Cache file path.
        """
        key = self._build_cache_key(source_segments, target_segments)
        return self.cache_dir / f"{key}.json"

    def _load_cached_output(self, cache_path: Path) -> list[dict] | None:
        """
        Load cached evaluation output if available.

        Parameters
        ----------
        cache_path : pathlib.Path
            Cache file path.

        Returns
        -------
        list[dict] | None
            Cached outputs when available and valid; otherwise ``None``.
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

        if not isinstance(cached_data, list):
            return None
        if not all(isinstance(item, dict) for item in cached_data):
            return None
        return cached_data

    def _save_output_cache(self, cache_path: Path, outputs: list[dict]) -> None:
        """
        Save evaluation outputs to cache.

        Parameters
        ----------
        cache_path : pathlib.Path
            Cache file path.
        outputs : list[dict]
            Evaluation outputs.

        Returns
        -------
        None
            This function only writes cache as a side effect.
        """
        if not self.use_cache:
            return
        save_json(outputs, cache_path)

    def call_openai_api_for_evaluation(
        self,
        source_segments: list[str],
        target_segments: list[str],
    ) -> list[dict] | None:
        """Evaluate translation pairs using OpenAI API with JSON schema."""

        messages = self.prompt_builder.build_prompt(
            source_texts=source_segments,
            target_texts=target_segments,
        ).messages

        try:
            message_content = self.llm.chat_complete(
                messages,
                model=self.model_name,
                temperature=1.0,
                timeout=self.request_timeout_sec,
                expect_json=True,
            )
        except LLMError as e:
            print(f"Error calling LLM API for evaluation ({e.reason}): {e}")
            return None

        result_json = self.parse_output(message_content)
        outputs: list[dict] | None = None
        if isinstance(result_json, dict):
            if "output" in result_json:
                outputs = result_json["output"]
            else:
                for key, value in result_json.items():
                    if not isinstance(key, str):
                        continue
                    normalized = key.strip().strip('"').strip("'")
                    if normalized == "output":
                        outputs = value
                        break
        elif isinstance(result_json, list):
            outputs = result_json

        if not isinstance(outputs, list):
            raise KeyError("output")
        if len(outputs) != len(source_segments):
            raise ValueError(
                f"Unexpected output length: {len(outputs)} != {len(source_segments)}"
            )

        for output in outputs:
            if not isinstance(output, dict):
                raise TypeError("Each output should be a dictionary")
            if self.source_lang not in output:
                raise KeyError(f"Output should contain '{self.source_lang}'")
            if self.target_lang not in output:
                raise KeyError(f"Output should contain '{self.target_lang}'")
            if "fluency" not in output:
                raise KeyError("Output should contain 'fluency'")
            if "accuracy" not in output:
                raise KeyError("Output should contain 'accuracy'")
            if "conciseness" not in output:
                raise KeyError("Output should contain 'conciseness'")

        return self._clone_outputs(outputs)

    def _evaluate_with_retries(
        self,
        source_segments: list[str],
        target_segments: list[str],
    ) -> list[dict] | None:
        """
        Evaluate translation segments with retry.

        Parameters
        ----------
        source_segments : list[str]
            Source text segments.
        target_segments : list[str]
            Target text segments.

        Returns
        -------
        list[dict] | None
            Evaluation outputs on success; otherwise ``None``.
        """
        for _ in range(self.max_retries):
            try:
                outputs = self.call_openai_api_for_evaluation(
                    source_segments,
                    target_segments,
                )
                if outputs is not None:
                    return outputs
            except Exception as e:
                print(f"Error evaluating texts: {e}. Retrying...")
        return None

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

    def evaluate_translations(
        self,
        source_segments: list[str],
        target_segments: list[str],
    ) -> list[dict] | None:
        if len(source_segments) != len(target_segments):
            raise ValueError(
                f"source_segments and target_segments must have the same length: "
                f"{len(source_segments)} != {len(target_segments)}"
            )

        cache_path = self._get_cache_path(source_segments, target_segments)
        cached_data = self._load_cached_output(cache_path)
        if cached_data is not None:
            return self._clone_outputs(cached_data)

        outputs = self._evaluate_with_retries(source_segments, target_segments)
        if outputs is None:
            print(
                f"Failed to evaluate texts after multiple attempts: {source_segments[:2]}..."
            )
            return None

        self._save_output_cache(cache_path, outputs)
        return self._clone_outputs(outputs)

    def evaluate_translations_in_parallel(
        self,
        source_segments_list: list[list[str]],
        target_segments_list: list[list[str]],
    ) -> list[list[dict] | None]:
        """Process dataset with parallel evaluation."""
        if len(source_segments_list) != len(target_segments_list):
            raise ValueError(
                f"source_segments_list and target_segments_list must have the same length: "
                f"{len(source_segments_list)} != {len(target_segments_list)}"
            )

        all_outputs: list[list[dict] | None] = [None] * len(source_segments_list)
        pending_requests: dict[str, tuple[list[str], list[str], Path]] = {}
        key_to_indices: dict[str, list[int]] = {}

        for idx, (source_segments, target_segments) in enumerate(
            zip(source_segments_list, target_segments_list)
        ):
            if len(source_segments) != len(target_segments):
                all_outputs[idx] = None
                continue

            cache_path = self._get_cache_path(source_segments, target_segments)
            if self.use_cache:
                cached_data = self._load_cached_output(cache_path)
                if cached_data is not None:
                    all_outputs[idx] = self._clone_outputs(cached_data)
                    continue

                key = cache_path.stem
                key_to_indices.setdefault(key, []).append(idx)
                if key not in pending_requests:
                    pending_requests[key] = (
                        source_segments,
                        target_segments,
                        cache_path,
                    )
                continue

            key = str(idx)
            key_to_indices[key] = [idx]
            pending_requests[key] = (source_segments, target_segments, cache_path)

        if not pending_requests:
            return all_outputs

        rate_limiter = RateLimiter(rps_limit=self.rps_limit)

        def worker(
            key: str,
            source_segments: list[str],
            target_segments: list[str],
            cache_path: Path,
        ) -> tuple[str, list[dict] | None]:
            """
            Process one unique evaluation request in realtime mode.

            Parameters
            ----------
            key : str
                Cache key for this request.
            source_segments : list[str]
                Source text segments.
            target_segments : list[str]
                Target text segments.
            cache_path : pathlib.Path
                Cache file path.

            Returns
            -------
            tuple[str, list[dict] | None]
                Cache key and evaluation outputs.
            """
            rate_limiter.acquire()
            outputs = self._evaluate_with_retries(source_segments, target_segments)
            if outputs is None:
                return key, None
            self._save_output_cache(cache_path, outputs)
            return key, outputs

        max_workers = self._resolve_max_workers(len(pending_requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    worker,
                    key,
                    source_segments,
                    target_segments,
                    cache_path,
                ): key
                for key, (
                    source_segments,
                    target_segments,
                    cache_path,
                ) in pending_requests.items()
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="score",
                disable=not self.show_progress,
            ):
                key = futures[future]
                try:
                    _, outputs = future.result()
                except Exception as e:
                    print(f"Error processing text pair: {e}")
                    outputs = None

                for idx in key_to_indices[key]:
                    if outputs is None:
                        all_outputs[idx] = None
                    else:
                        all_outputs[idx] = self._clone_outputs(outputs)

        return all_outputs
