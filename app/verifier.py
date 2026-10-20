"""Numeric-faithfulness check: every number in a draft answer must come from a tool result.

Deterministic on purpose (no LLM): the verifier is the safety net against fabricated numbers.
A number is *supported* when, rounded to the number of decimals it is written with, it equals a
numeric value found anywhere in the tool results, or when it was already present in the user's
question (e.g. "top 5") or is the fixed 80% label of the p10-p90 interval.
"""

from __future__ import annotations

import re
from typing import Any

ALWAYS_ALLOWED = (80.0,)  # "80% interval" (p10-p90)

_IDENTIFIERS = re.compile(r"\b[A-Za-z]+-\d{2,4}\b|\b[A-Za-z]+\d+\b")  # Cerrado-2027, L042, m12, p90
_LIST_MARKERS = re.compile(r"(?m)^\s*\d+[.)]\s")  # "1. ", "2) "
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> list[str]:
    """Numbers (as written) in ``text``, ignoring identifiers such as L042 or Cerrado-2027 and list markers."""
    cleaned = _LIST_MARKERS.sub(" ", text)
    cleaned = _IDENTIFIERS.sub(" ", cleaned)
    cleaned = _DECIMAL_COMMA.sub(".", cleaned)
    return _NUMBER.findall(cleaned)


def collect_numbers(obj: Any) -> list[float]:
    """All numeric leaves of a nested structure (bools excluded)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in collect_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in collect_numbers(v)]
    return []


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def is_supported(number: str, allowed: list[float]) -> bool:
    x, d = float(number), _decimals(number)
    tolerance = 0.5 * 10**-d + 1e-9
    return any(abs(v - x) <= tolerance for v in allowed)


def unsupported_numbers(text: str, tool_results: list[Any], question: str = "") -> list[str]:
    """Numbers in ``text`` that cannot be traced to ``tool_results`` or ``question`` (order kept, no repeats)."""
    allowed = collect_numbers(tool_results) + [float(n) for n in extract_numbers(question)] + list(ALWAYS_ALLOWED)
    seen: dict[str, None] = {}
    for number in extract_numbers(text):
        if not is_supported(number, allowed):
            seen[number] = None
    return list(seen)


def redact(text: str, numbers: list[str]) -> str:
    """Replace the given (unverified) numbers in ``text`` with a visible placeholder."""
    for number in sorted(set(numbers), key=len, reverse=True):
        text = re.sub(rf"(?<![\w.-]){re.escape(number)}(?![\w]|\.\d)", "[unverified]", text)
    return text
