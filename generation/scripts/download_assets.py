from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console

from dataset_loading import DEFAULT_LANGUAGES_FILE, load_language_codes, load_language_dataset


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model and GSM8K-106 into self-contained offline directories."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--dataset", default="open-multilang/gsm8k-106")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = args.languages or load_language_codes(DEFAULT_LANGUAGES_FILE)
    if not languages:
        raise ValueError("No languages were provided or found in scripts/langs.txt.")

    args.model_dir.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Downloading model[/bold] {args.model} -> {args.model_dir}")
    snapshot_download(repo_id=args.model, local_dir=args.model_dir)

    downloaded = []
    for split in args.splits:
        for language in languages:
            output_path = args.dataset_dir / split / language
            state_path = output_path / "state.json"
            if state_path.exists() and not args.force:
                console.print(f"[cyan]Skipping[/cyan] {split}/{language}: already saved")
                downloaded.append({"split": split, "language": language, "path": str(output_path)})
                continue

            console.print(f"[bold]Downloading dataset[/bold] {split}/{language}")
            dataset = load_language_dataset(args.dataset, split, language)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(output_path), max_shard_size="1GB")
            downloaded.append(
                {
                    "split": split,
                    "language": language,
                    "rows": len(dataset),
                    "path": str(output_path),
                }
            )

    manifest = {
        "model_id": args.model,
        "model_path": str(args.model_dir),
        "dataset_id": args.dataset,
        "dataset_path": str(args.dataset_dir),
        "splits": args.splits,
        "languages": languages,
        "artifacts": downloaded,
    }
    (args.dataset_dir / "offline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print("[green]Offline assets are ready.[/green]")


if __name__ == "__main__":
    main()
