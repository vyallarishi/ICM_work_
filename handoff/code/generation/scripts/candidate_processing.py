from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")
DIGIT_RUN_RE = re.compile(r"\d{40,}")
SOURCE_PRIORITY = {"main": 0, "retry": 1, "wrong_chain_ablation": 2}
GOLD_RECORD_FIELDS = {
    "gold_raw",
    "gold_number",
    "original_answer",
    "original_answer_number",
    "source_records",
}
GOLD_CANDIDATE_FIELDS = {"correct"}


@dataclass(frozen=True)
class QualityConfig:
    min_chars: int = 80
    max_chars: int = 6000
    require_parseable: bool = True


def prepare_candidates(
    candidates: list[dict[str, Any]], config: QualityConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated = [annotate_quality(candidate, config) for candidate in candidates]
    usable = sorted(
        dedupe_candidates([candidate for candidate in annotated if candidate["quality_ok"]]),
        key=candidate_sort_key,
    )
    return annotated, usable


def annotate_quality(candidate: dict[str, Any], config: QualityConfig) -> dict[str, Any]:
    text = str(candidate.get("text") or "")
    reasons = []
    if config.require_parseable and candidate.get("pred_number") is None:
        reasons.append("unparseable_answer")
    if len(text.strip()) < config.min_chars and "####" not in text:
        reasons.append("too_short")
    if len(text) > config.max_chars:
        reasons.append("too_long")
    if is_degenerate_text(text):
        reasons.append("degenerate_text")
    if "####" not in text and candidate.get("pred_number") is None:
        reasons.append("missing_answer_marker")
    return {
        **candidate,
        "quality_ok": not reasons,
        "quality_reasons": reasons,
        "normalized_text": normalize_text(text),
    }


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_texts: set[str] = set()
    seen_answer_text_pairs: set[tuple[str | None, str]] = set()
    deduped = []
    for candidate in candidates:
        normalized = str(candidate["normalized_text"])
        answer_text_pair = (candidate.get("pred_number"), normalized[:240])
        if normalized in seen_texts or answer_text_pair in seen_answer_text_pairs:
            continue
        seen_texts.add(normalized)
        seen_answer_text_pairs.add(answer_text_pair)
        deduped.append(candidate)
    return deduped


def select_random(
    candidates: list[dict[str, Any]], size: int, seed: int, record_id: str
) -> list[dict[str, Any]]:
    ordered = list(candidates)
    random.Random(stable_seed(seed, record_id, "random")).shuffle(ordered)
    return ordered[:size]


def select_diverse(
    candidates: list[dict[str, Any]], size: int, seed: int, record_id: str
) -> list[dict[str, Any]]:
    """Select answer- and regime-diverse chains without consulting gold correctness."""
    if len(candidates) <= size:
        return list(candidates)

    rng = random.Random(stable_seed(seed, record_id, "diverse"))
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # First cover distinct model-predicted answers. These are observable model outputs,
    # not comparisons with the reference answer.
    answer_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in shuffled:
        answer_groups.setdefault(answer_key(candidate), []).append(candidate)
    groups = sorted(answer_groups.values(), key=lambda group: (-len(group), candidate_id(group[0])))
    for group in groups:
        candidate = best_text_diversity_candidate(group, selected)
        add_candidate(selected, selected_ids, candidate)
        if len(selected) == size:
            return selected

    # If fewer answer clusters than slots exist, prefer regimes not represented yet.
    represented_regimes = {str(candidate.get("regime_name") or "") for candidate in selected}
    for candidate in shuffled:
        regime = str(candidate.get("regime_name") or "")
        if regime not in represented_regimes and add_candidate(selected, selected_ids, candidate):
            represented_regimes.add(regime)
            if len(selected) == size:
                return selected

    while len(selected) < size:
        remaining = [
            candidate for candidate in shuffled if candidate_id(candidate) not in selected_ids
        ]
        if not remaining:
            break
        add_candidate(selected, selected_ids, best_text_diversity_candidate(remaining, selected))
    return selected


def select_oracle_balanced(
    candidates: list[dict[str, Any]], size: int, min_correct: int, max_correct: int
) -> list[dict[str, Any]]:
    correct = [candidate for candidate in candidates if candidate.get("correct") is True]
    incorrect = [candidate for candidate in candidates if candidate.get("correct") is not True]
    correct_target = min(max_correct, len(correct))
    if correct_target < min_correct and len(correct) >= min_correct:
        correct_target = min_correct
    selected = correct[:correct_target] + incorrect[: size - correct_target]
    selected_ids = {candidate_id(candidate) for candidate in selected}
    if len(selected) < size:
        selected.extend(
            candidate
            for candidate in candidates
            if candidate_id(candidate) not in selected_ids
        )
    return selected[:size]


def sanitize_record(
    record: dict[str, Any], candidates: list[dict[str, Any]], include_gold: bool = False
) -> dict[str, Any]:
    excluded = {"candidates", "prompt", "skipped", "pooled_candidate_count"}
    if not include_gold:
        excluded |= GOLD_RECORD_FIELDS
    output = {key: value for key, value in record.items() if key not in excluded}
    output["candidates"] = [sanitize_candidate(candidate, include_gold) for candidate in candidates]
    return output


def sanitize_candidate(candidate: dict[str, Any], include_gold: bool = False) -> dict[str, Any]:
    excluded = {"normalized_text"}
    if not include_gold:
        excluded |= GOLD_CANDIDATE_FIELDS
    return {key: value for key, value in candidate.items() if key not in excluded}


def candidate_statistics(
    record: dict[str, Any], annotated: list[dict[str, Any]], usable: list[dict[str, Any]]
) -> dict[str, Any]:
    correct = sum(candidate.get("correct") is True for candidate in usable)
    incorrect = sum(candidate.get("correct") is not True for candidate in usable)
    quality_failures: dict[str, int] = {}
    for candidate in annotated:
        for reason in candidate.get("quality_reasons", []):
            quality_failures[reason] = quality_failures.get(reason, 0) + 1
    quality_pass_count = sum(bool(candidate.get("quality_ok")) for candidate in annotated)
    regime_stats = grouped_candidate_statistics(annotated, usable, "regime_name")
    source_stats = grouped_candidate_statistics(annotated, usable, "candidate_source")
    return {
        "id": str(record.get("id")),
        "original_id": record.get("original_id"),
        "language": record.get("language"),
        "raw_candidate_count": len(annotated),
        "quality_pass_candidate_count": quality_pass_count,
        "usable_candidate_count": len(usable),
        "duplicates_removed": quality_pass_count - len(usable),
        "correct_usable_count": correct,
        "incorrect_usable_count": incorrect,
        "usable_accuracy": correct / len(usable) if usable else None,
        "distinct_predicted_answers": len({answer_key(candidate) for candidate in usable}),
        "quality_failure_counts": quality_failures,
        "regime_stats": regime_stats,
        "source_stats": source_stats,
        "gold_number": record.get("gold_number"),
    }


def grouped_candidate_statistics(
    annotated: list[dict[str, Any]], usable: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    groups = {
        str(candidate.get(field) or "unknown") for candidate in [*annotated, *usable]
    }
    result = {}
    for group in sorted(groups):
        raw_group = [
            candidate
            for candidate in annotated
            if str(candidate.get(field) or "unknown") == group
        ]
        usable_group = [
            candidate
            for candidate in usable
            if str(candidate.get(field) or "unknown") == group
        ]
        correct = sum(candidate.get("correct") is True for candidate in usable_group)
        result[group] = {
            "raw_candidates": len(raw_group),
            "usable_candidates": len(usable_group),
            "correct_usable_candidates": correct,
            "incorrect_usable_candidates": len(usable_group) - correct,
        }
    return result


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    source = str(candidate.get("candidate_source") or "main")
    return (
        SOURCE_PRIORITY.get(source, 99),
        int(str(candidate.get("regime_name") or "").endswith("high_temp")),
        len(str(candidate.get("text") or "")),
        candidate_id(candidate),
    )


def best_text_diversity_candidate(
    candidates: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    if not selected:
        return min(candidates, key=candidate_sort_key)
    selected_tokens = [word_set(candidate) for candidate in selected]

    def score(candidate: dict[str, Any]) -> tuple[float, tuple[int, int, int, str]]:
        tokens = word_set(candidate)
        max_overlap = max(jaccard(tokens, other) for other in selected_tokens)
        return max_overlap, candidate_sort_key(candidate)

    return min(candidates, key=score)


def add_candidate(
    selected: list[dict[str, Any]], selected_ids: set[str], candidate: dict[str, Any]
) -> bool:
    identity = candidate_id(candidate)
    if identity in selected_ids:
        return False
    selected.append(candidate)
    selected_ids.add(identity)
    return True


def answer_key(candidate: dict[str, Any]) -> str:
    value = candidate.get("pred_number")
    return str(value) if value is not None else "<unparseable>"


def candidate_id(candidate: dict[str, Any]) -> str:
    identity = candidate.get("global_sample_id")
    if identity is not None:
        return str(identity)
    return json.dumps(
        [candidate.get("regime_name"), candidate.get("seed"), candidate.get("sample_index")]
    )


def stable_seed(seed: int, record_id: str, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).strip().lower())


def repetition_ratio(text: str) -> float:
    words = [word for word in re.split(r"\W+", text.lower()) if word]
    if len(words) < 20:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))


def is_degenerate_text(text: str) -> bool:
    """Detect echo loops, babble, and numeric spam without penalizing terse answers.

    Word-level type-token ratio misfires on annotated GSM8K solutions: numbers,
    nouns, and particles legitimately repeat, especially in short non-English
    traces. Degeneracy is structural instead: very long digit runs (decimal
    spam), the same long line emitted three or more times (prompt-echo loops),
    or heavy word-trigram duplication (babble).
    """
    if DIGIT_RUN_RE.search(text):
        return True
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    if lines and max(Counter(lines).values()) >= 3:
        return True
    words = [word for word in re.split(r"\W+", text.lower()) if word]
    if len(words) >= 30:
        trigrams = list(zip(words, words[1:], words[2:]))
        if len(set(trigrams)) / len(trigrams) < 0.5:
            return True
    return False


def word_set(candidate: dict[str, Any]) -> set[str]:
    return set(normalize_text(str(candidate.get("text") or "")).split())


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
