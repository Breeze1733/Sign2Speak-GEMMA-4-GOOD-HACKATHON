"""Thin wrapper around the LiteRT-LM Python engine.

Pinned to `litert-lm-api==0.19.0` so the remote backend matches the Android
app's `com.google.ai.edge.litertlm:litertlm-android:0.19.0`.

Mirrors Gloss2TextEngine.kt:
  * one Engine, loaded once and reused
  * a fresh Conversation per request (no cross-sentence context leakage)
  * sampler top_k=40 / top_p=0.95 / temperature=0.3
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import litert_lm

from prompt import (
    SAMPLER_TEMPERATURE,
    SAMPLER_TOP_K,
    SAMPLER_TOP_P,
    build_system_message,
    build_user_message,
)

log = logging.getLogger("s2s.backend")

_MAX_TEXT_FIELDS = ("text", "content", "message")


def _coerce_text(value) -> str | None:
    if isinstance(value, str):
        return value
    # Contents / list of Content(text=...)
    contents = getattr(value, "contents", None)
    if contents:
        parts = []
        for c in contents:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
            else:
                parts.append(str(c))
        return "".join(parts)
    return None


def extract_text(response) -> str:
    """Pull the generated string out of whatever the binding returns.

    The LiteRT-LM Python API returns a Message, which is dict-like; shapes have
    changed across versions, so try the documented fields before falling back
    to ``str(response)``.
    """
    for field in _MAX_TEXT_FIELDS:
        candidate = None
        if isinstance(response, dict):
            candidate = response.get(field)
        else:
            candidate = getattr(response, field, None)
        text = _coerce_text(candidate)
        if text and text.strip():
            return text

    text = _coerce_text(getattr(response, "contents", None))
    if text and text.strip():
        return text
    return str(response)


def clean_sentence(text: str) -> str:
    for marker in ("<end_of_turn>", "<eos>"):
        text = text.replace(marker, "")
    return text.strip()


@dataclass
class GenerationResult:
    text: str
    status: str
    inference_ms: float


class GlossTranslator:
    """Serial, thread-safe gloss -> sentence generator."""

    def __init__(
        self,
        model_path: str,
        *,
        thread_count: int | None = None,
        max_output_tokens: int = 64,
        max_num_tokens: int | None = 2048,
        cache_dir: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.thread_count = thread_count
        self.max_output_tokens = max_output_tokens
        self._lock = threading.Lock()
        self._engine = None

        started = time.perf_counter()
        backend = litert_lm.Backend.CPU(thread_count=thread_count)
        kwargs = {"backend": backend, "cache_dir": cache_dir}
        if max_num_tokens:
            kwargs["max_num_tokens"] = max_num_tokens
        self._engine = litert_lm.Engine(model_path, **kwargs)
        self.load_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "engine loaded: path=%s threads=%s load_ms=%.1f",
            model_path,
            thread_count,
            self.load_ms,
        )
        self._sampler_kwargs = dict(
            top_k=SAMPLER_TOP_K,
            top_p=SAMPLER_TOP_P,
            temperature=SAMPLER_TEMPERATURE,
        )

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def generate(self, glosses: list[str], style: str = "friendly") -> GenerationResult:
        if not glosses:
            return GenerationResult("", "empty_output", 0.0)

        started = time.perf_counter()
        conversation = None
        try:
            conversation = self._engine.create_conversation(
                system_message=build_system_message(style),
                sampler_config=litert_lm.SamplerConfig(**self._sampler_kwargs),
                max_output_tokens=self.max_output_tokens,
            )
            response = conversation.send_message(build_user_message(glosses))
            sentence = clean_sentence(extract_text(response))
            elapsed = (time.perf_counter() - started) * 1000.0
            if not sentence:
                return GenerationResult("", "empty_output", elapsed)
            return GenerationResult(sentence, "generated", elapsed)
        except Exception as exc:  # noqa: BLE001 - report instead of crashing worker
            elapsed = (time.perf_counter() - started) * 1000.0
            log.exception("generation failed: %s", exc)
            return GenerationResult("", f"generation_error: {exc}", elapsed)
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.close()
                except Exception:  # noqa: BLE001
                    pass
                self._engine = None
