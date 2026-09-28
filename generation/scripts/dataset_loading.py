from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, get_dataset_config_names, load_dataset, load_from_disk


QUESTION_COLUMNS = ("question", "problem", "query", "input", "prompt")
LANGUAGE_COLUMNS = ("language", "lang", "locale", "config", "subset")
DEFAULT_LANGUAGES_FILE = Path(__file__).with_name("langs.txt")


@dataclass(frozen=True)
class LanguageDataset:
    language: str
    dataset: Dataset


def discover_language_datasets(
    dataset_name: str,
    split: str,
    languages: Iterable[str] | None = None,
    languages_file: Path | None = DEFAULT_LANGUAGES_FILE,
) -> list[LanguageDataset]:
    language_codes = list(languages or load_language_codes(languages_file))
    if language_codes:
        return [
            LanguageDataset(language=language, dataset=load_language_dataset(dataset_name, split, language))
            for language in language_codes
        ]

    configs = _safe_config_names(dataset_name)
    if configs and configs != ["default"]:
        return [
            LanguageDataset(language=config, dataset=load_dataset(dataset_name, config, split=split))
            for config in configs
        ]

    dataset = load_dataset(dataset_name, split=split)
    language_col = first_existing_column(dataset.column_names, LANGUAGE_COLUMNS)
    if language_col is None:
        return [LanguageDataset(language="default", dataset=dataset)]

    indices_by_language: dict[str, list[int]] = {}
    for idx, value in enumerate(dataset[language_col]):
        indices_by_language.setdefault(str(value), []).append(idx)

    return [
        LanguageDataset(language=language, dataset=dataset.select(indices))
        for language, indices in sorted(indices_by_language.items())
    ]


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def get_question(example: dict, columns: Iterable[str]) -> str:
    column = first_existing_column(columns, QUESTION_COLUMNS)
    if column is None:
        raise KeyError(f"No question column found. Tried: {', '.join(QUESTION_COLUMNS)}")
    return str(example[column])


def load_language_dataset(dataset_name: str, split: str, language: str) -> Dataset:
    local_path = Path(dataset_name).expanduser() / split / language
    if local_path.exists():
        return load_from_disk(str(local_path))

    data_file = f"data/{split}/{language}.jsonl"
    try:
        return load_dataset(dataset_name, data_files={split: data_file}, split=split)
    except Exception:
        pass

    try:
        return load_dataset(dataset_name, language, split=split)
    except Exception:
        pass

    dataset = load_dataset(dataset_name, split=split)
    language_col = first_existing_column(dataset.column_names, LANGUAGE_COLUMNS)
    if language_col is None:
        raise ValueError(f"Could not load language '{language}' from {dataset_name}.")

    indices = [idx for idx, value in enumerate(dataset[language_col]) if str(value) == language]
    if not indices:
        raise ValueError(f"Could not find language '{language}' in column '{language_col}'.")
    return dataset.select(indices)


def load_language_codes(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if "=" in text:
        text = text.split("=", 1)[1].strip()

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [
            line.split("#", 1)[0].strip().strip("\"'")
            for line in text.splitlines()
            if line.split("#", 1)[0].strip()
        ]

    if isinstance(parsed, dict):
        return [str(key) for key in parsed]
    if isinstance(parsed, (list, tuple, set)):
        return [str(value) for value in parsed]
    raise ValueError(f"Unsupported languages file format in {path}")


def load_icm_seed_json(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    seeds: dict[str, dict] = {}
    for row in rows:
        key = str(row["consistency_id"])
        seeds.setdefault(
            key,
            {
                "question": row["question"],
                "ref_cot": row.get("ref_cot"),
                "ref_answer": row.get("ref_answer"),
                "choices": [],
            },
        )
        seeds[key]["choices"].append(row)
    return seeds


def _safe_config_names(dataset_name: str) -> list[str]:
    try:
        return list(get_dataset_config_names(dataset_name))
    except Exception:
        return []
