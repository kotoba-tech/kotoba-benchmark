"""LLM-client abstraction routed by model-name prefix.

`gpt-*` / anything else → OpenAI Chat Completions via the `openai` SDK.
`gemini-*`              → Gemini `generate_content` via the `google-genai` SDK.

Both implementations return the response *text* (the model's output content)
and raise a uniform `LLMError` so the align/eval managers can run their
existing JSON repair + retry logic without caring which provider answered.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import openai

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Backend-agnostic error from an LLM call."""

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


class LLMClient(Protocol):
    def chat_complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        timeout: float | int,
        expect_json: bool,
    ) -> str: ...


def _classify_openai_error(e: Exception) -> str:
    name = e.__class__.__name__.lower()
    msg = str(e).lower()
    if "timeout" in name or "timed out" in msg or "timeout" in msg:
        return "openai_timeout"
    if "rate" in msg and "limit" in msg:
        return "openai_rate_limit"
    if "connection" in msg:
        return "openai_connection_error"
    if "authentication" in msg or "api key" in msg:
        return "openai_auth_error"
    return f"openai_error_{e.__class__.__name__}"


def _classify_gemini_error(e: Exception) -> str:
    name = e.__class__.__name__.lower()
    msg = str(e).lower()
    if "timeout" in name or "deadline" in msg or "504" in msg:
        return "gemini_timeout"
    if "429" in msg or "resource_exhausted" in msg or ("rate" in msg and "limit" in msg):
        return "gemini_rate_limit"
    if "permission" in msg or "401" in msg or "403" in msg or "api key" in msg:
        return "gemini_auth_error"
    return f"gemini_error_{e.__class__.__name__}"


class OpenAILLMClient:
    """Thin wrapper around `openai.OpenAI().chat.completions.create`."""

    def __init__(self) -> None:
        self._client = openai.OpenAI()

    def chat_complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 1.0,
        timeout: float | int = 120,
        expect_json: bool = True,  # noqa: ARG002 — OpenAI behavior parity: no response_format
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
        except openai.OpenAIError as e:
            raise LLMError(str(e), reason=_classify_openai_error(e)) from e
        if not response.choices:
            raise LLMError("empty choices", reason="empty_choices")
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("empty content", reason="empty_message_content")
        return content


class GeminiLLMClient:
    """Native Gemini client via `google-genai`. Uses `response_mime_type=application/json`
    when JSON is expected — gives stricter structured output than the OpenAI-compat shim."""

    def __init__(self) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set to use a gemini-* model"
            )
        self._client = genai.Client(api_key=api_key)

    def chat_complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 1.0,
        timeout: float | int = 120,
        expect_json: bool = True,
    ) -> str:
        from google.genai import errors as genai_errors
        from google.genai import types as genai_types

        system_parts: list[str] = []
        contents: list[genai_types.Content] = []
        for m in messages:
            role = m.get("role")
            text = m.get("content", "")
            if not isinstance(text, str):
                continue
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                contents.append(
                    genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
                )
            elif role == "assistant":
                contents.append(
                    genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
                )

        config_kwargs: dict = {"temperature": temperature}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if expect_json:
            config_kwargs["response_mime_type"] = "application/json"

        http_options = genai_types.HttpOptions(timeout=int(timeout) * 1000)

        try:
            client = (
                self._client.with_options(http_options=http_options)
                if hasattr(self._client, "with_options")
                else self._client
            )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as e:  # type: ignore[attr-defined]
            raise LLMError(str(e), reason=_classify_gemini_error(e)) from e
        except Exception as e:  # noqa: BLE001 — network / transport errors from google-genai vary
            raise LLMError(str(e), reason=_classify_gemini_error(e)) from e

        text = getattr(response, "text", None)
        if text:
            return text
        # Fallback: some responses put text under candidates[0].content.parts
        try:
            return response.candidates[0].content.parts[0].text  # type: ignore[union-attr]
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError("empty content", reason="empty_message_content") from e


def make_llm_client(model_name: str) -> LLMClient:
    """Route to the native SDK based on model-name prefix."""
    if model_name.lower().startswith("gemini-"):
        logger.debug("LLM dispatch: %s → GeminiLLMClient", model_name)
        return GeminiLLMClient()
    logger.debug("LLM dispatch: %s → OpenAILLMClient", model_name)
    return OpenAILLMClient()
