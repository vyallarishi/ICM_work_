from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUMBER_RE = re.compile(rf"(?<![\w/]){NUMBER_PATTERN}(?![\w/])")
FRACTION_RE = re.compile(rf"^\s*({NUMBER_PATTERN})\s*/\s*({NUMBER_PATTERN})\s*$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_LINE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*(.+)",
    flags=re.IGNORECASE,
)


def normalize_number(text: str | int | float | Decimal | None) -> Decimal | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    raw = raw.replace("\u2212", "-").replace(",", "").rstrip(".")
    fraction = FRACTION_RE.fullmatch(raw)
    if fraction:
        numerator = normalize_number(fraction.group(1))
        denominator = normalize_number(fraction.group(2))
        if numerator is None or denominator in (None, Decimal("0")):
            return None
        return numerator / denominator

    try:
        parsed = Decimal(raw)
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return None
    return parsed


def extract_gold_answer(example: dict) -> tuple[str | None, Decimal | None]:
    for key in ("final_answer", "answer_number", "target", "label", "ref_answer", "answer"):
        if key in example and example[key] is not None:
            value = str(example[key])
            parsed = normalize_number(value)
            if parsed is not None:
                return value, parsed

    answer = str(example.get("answer", ""))
    if "####" in answer:
        candidate = answer.rsplit("####", 1)[-1]
        parsed = extract_last_number(candidate)
        if parsed is not None:
            return candidate.strip(), parsed

    parsed = extract_last_number(answer)
    return (answer if parsed is not None else None), parsed


def extract_prediction_answer(text: str) -> tuple[str | None, Decimal | None]:
    marked = extract_marked_answer_segment(text)
    if marked is not None:
        raw, parsed = extract_number_from_answer_segment(marked)
        return raw, parsed

    parsed = extract_last_number(text)
    if parsed is None:
        return None, None
    return str(parsed), parsed


def extract_marked_answer_segment(text: str) -> str | None:
    normalized = str(text)
    if "####" in normalized:
        return first_nonempty_line(normalized.rsplit("####", 1)[-1])

    matches = ANSWER_LINE_RE.findall(normalized)
    if matches:
        return first_nonempty_line(matches[-1])
    return None


def first_nonempty_line(text: str) -> str | None:
    for line in str(text).splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def extract_number_from_answer_segment(segment: str) -> tuple[str | None, Decimal | None]:
    raw = clean_answer_segment(segment)
    parsed = normalize_number(raw)
    if parsed is not None:
        return raw, parsed

    if contains_letters(raw):
        return None, None

    parsed = extract_last_number(raw)
    if parsed is None:
        return None, None
    return str(parsed), parsed


def clean_answer_segment(segment: str) -> str:
    raw = str(segment).strip()
    boxed = BOXED_RE.search(raw)
    if boxed:
        raw = boxed.group(1).strip()
    return raw.strip().strip("$").strip().rstrip(".。")


def contains_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def extract_last_number(text: str) -> Decimal | None:
    matches = NUMBER_RE.findall(str(text).replace("\u2212", "-"))
    for raw in reversed(matches):
        parsed = normalize_number(raw)
        if parsed is not None:
            return parsed
    return None


def numbers_match(predicted: Decimal | None, gold: Decimal | None, tolerance: Decimal) -> bool:
    if predicted is None or gold is None:
        return False
    return abs(predicted - gold) <= tolerance
