from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROMPT_TEMPLATES = {
    "zero_shot_cot": (
        "Please solve the following math problem step by step. Show your reasoning process. "
        "The last line must contain only '####' followed by a numerical answer. Do not include "
        "units, variables, or expressions after '####'.\n\nQuestion: {question}\n\nAnswer:"
    ),
    "few_shot_cot": (
        "{few_shot_examples}\n\n"
        "Now solve this problem step by step. The last line must contain only '####' followed "
        "by a numerical answer.\n\nQuestion: {question}\n\nAnswer:"
    ),
    "hhh_zero_shot_cot": (
        "{assistant_preamble}\n\n"
        "Human: Please solve the following math problem step by step. The last line must contain "
        "only '####' followed by a numerical answer.\n\nQuestion: {question}\n\nAssistant:"
    ),
    "wrong_cot": (
        "Write a plausible but incorrect step-by-step solution to the following math problem. "
        "The reasoning should look coherent, but it must contain a subtle arithmetic or setup "
        "mistake that leads to an incorrect final numeric answer. The last line must contain "
        "only '####' followed by that incorrect numerical answer.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
}

PROMPTS_DIR = Path(__file__).with_name("prompts")
PROMPT_FIELD = "prompt"
LANGUAGE_FIELD = "language_code"
NATIVE_FEW_SHOT_COUNT = 6
SUPPORTED_FEW_SHOT_COUNTS = (2, 6)
DEFAULT_FEW_SHOT_ORIGINAL_IDS = (0, 5, 7, 10, 13, 20)
EXAMPLE_BLOCK_PATTERN = re.compile(r"<example>.*?</example>\s*", re.DOTALL)


def render_prompt(
    question: str,
    template_name: str,
    language: str | None = None,
    english_prompt_only: bool = False,
    few_shot_examples: str = "",
    assistant_preamble: str = "",
    demonstrations: list[dict[str, str]] | None = None,
) -> str:
    template = get_prompt_template(template_name, language, english_prompt_only)
    values = {
        "question": question.strip(),
        "query_question": question.strip(),
        "few_shot_examples": few_shot_examples.strip(),
        "assistant_preamble": assistant_preamble.strip(),
    }
    if demonstrations is not None:
        if len(demonstrations) not in SUPPORTED_FEW_SHOT_COUNTS:
            raise ValueError(
                f"Native few-shot prompts require one of {SUPPORTED_FEW_SHOT_COUNTS} demonstrations; "
                f"received {len(demonstrations)}."
            )
        template = truncate_example_blocks(template, len(demonstrations))
        for index, demonstration in enumerate(demonstrations, start=1):
            values[f"demo_{index}_question"] = demonstration["question"].strip()
            values[f"demo_{index}_answer"] = demonstration["answer"].strip()
    try:
        return template.format(**values).strip()
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt field in template '{template_name}': {exc}") from exc


def derive_stop_sequences(
    template_name: str,
    language: str | None = None,
    english_prompt_only: bool = False,
) -> list[str]:
    """Derive per-language stop strings from the rendered template itself.

    Base models keep sampling after their answer, typically by echoing the
    prompt structure (the native question label or an <example> block). The
    question label is whatever text immediately precedes the {question} slot
    in this language's template, so stopping on it is language-agnostic.
    """
    template = get_prompt_template(template_name, language, english_prompt_only)
    stops = ["<example>"]
    slot = max(("{query_question}", "{question}"), key=template.rfind)
    head, _, _ = template.rpartition(slot)
    label = head.rstrip().rsplit("\n", 1)[-1].strip()
    if label and "{" not in label:
        stops.append("\n" + label)
    if "\nQuestion:" not in stops:
        stops.append("\nQuestion:")
    return stops


def truncate_example_blocks(template: str, keep: int) -> str:
    """Keep only the first `keep` <example> blocks in a native few-shot template.

    The localized few_shot_cot templates hardcode six <example> blocks with
    demo_1..demo_6 slots. Dropping the trailing blocks yields a shorter prompt
    without touching the localized instructions. Returns the template unchanged
    when it already has at most `keep` blocks.
    """
    blocks = EXAMPLE_BLOCK_PATTERN.findall(template)
    if len(blocks) <= keep:
        return template
    for block in blocks[keep:]:
        template = template.replace(block, "", 1)
    return template


def get_prompt_template(
    template_name: str,
    language: str | None = None,
    english_prompt_only: bool = False,
) -> str:
    templates = load_prompt_templates(template_name)
    prompt_language = "en" if english_prompt_only else language

    if prompt_language and prompt_language in templates:
        return templates[prompt_language]
    if "en" in templates:
        return templates["en"]
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name]

    known = ", ".join(available_prompt_names())
    raise ValueError(f"Unknown prompt template '{template_name}'. Known templates: {known}")


def get_prompt_language(
    template_name: str,
    language: str | None = None,
    english_prompt_only: bool = False,
) -> str:
    templates = load_prompt_templates(template_name)
    if english_prompt_only:
        return "en"
    if language and language in templates:
        return language
    return "en"


@lru_cache(maxsize=None)
def load_prompt_templates(template_name: str) -> dict[str, str]:
    templates = dict(PROMPT_TEMPLATES)
    path = find_prompt_file(template_name)
    if path is None:
        return templates

    file_templates = load_prompt_file(path)
    if not file_templates:
        raise ValueError(f"No prompts found in {path}")
    return file_templates


def find_prompt_file(template_name: str) -> Path | None:
    normalized = template_name.replace("-", "_")
    candidates = [
        PROMPTS_DIR / f"{template_name}.jsonl",
        PROMPTS_DIR / f"{normalized}.jsonl",
        PROMPTS_DIR / f"{normalized.replace('_', '')}.jsonl",
        PROMPTS_DIR / f"{template_name}.json",
        PROMPTS_DIR / f"{normalized}.json",
        PROMPTS_DIR / f"{normalized.replace('_', '')}.json",
    ]

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def load_prompt_file(path: Path) -> dict[str, str]:
    if path.suffix == ".jsonl":
        return load_jsonl_prompt_file(path)
    if path.suffix == ".json":
        return load_json_prompt_file(path)
    raise ValueError(f"Unsupported prompt file format: {path}")


def load_jsonl_prompt_file(path: Path) -> dict[str, str]:
    templates: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            language, prompt = parse_prompt_record(row, path, line_number)
            templates[language] = prompt
    return templates


def load_json_prompt_file(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        templates = {}
        for index, row in enumerate(data, start=1):
            language, prompt = parse_prompt_record(row, path, index)
            templates[language] = prompt
        return templates
    if isinstance(data, dict):
        if PROMPT_FIELD in data and LANGUAGE_FIELD in data:
            language, prompt = parse_prompt_record(data, path, 1)
            return {language: prompt}
        return {str(language): str(prompt) for language, prompt in data.items() if isinstance(prompt, str)}
    raise ValueError(f"Unsupported JSON prompt file shape: {path}")


def parse_prompt_record(row: Any, path: Path, line_number: int) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise ValueError(f"Prompt record in {path}:{line_number} must be a JSON object.")
    language = row.get(LANGUAGE_FIELD) or row.get("language") or row.get("lang")
    prompt = row.get(PROMPT_FIELD) or row.get("template")
    if not language or not prompt:
        raise ValueError(
            f"Prompt record in {path}:{line_number} must include language_code and prompt."
        )
    return str(language), str(prompt)


def available_prompt_names() -> list[str]:
    names = set(PROMPT_TEMPLATES)
    if PROMPTS_DIR.exists():
        names.update(path.stem for path in PROMPTS_DIR.glob("*.jsonl"))
        names.update(path.stem for path in PROMPTS_DIR.glob("*.json"))
    return sorted(names)
