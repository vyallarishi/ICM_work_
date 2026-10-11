"""Rebuild data/ exactly as it was on the original cluster, from public sources, and verify every byte.

Git holds no data. This script recreates it in two phases (compute nodes often have no internet):

  python setup/build_data.py download   # login node (internet): fetch raw files into data/_downloads/
  python setup/build_data.py build      # anywhere, offline (~1 h; setup/build_data.sbatch runs it as a job) (~1 h; resumable)

Sources (pinned):
  * GSM8K-106: huggingface.co/datasets/open-multilang/gsm8k-106 @ DATASET_REVISION (unchanged since
    2026-04-09; the original copy was made from this revision). 67 languages x {train, test}.
  * MGSM human translations (Shi et al. 2022): github.com/google-research/url-nlp, mgsm/mgsm_<lang>.tsv
    @ MGSM_COMMIT, checked against setup/fingerprints/mgsm_md5sums.txt.

Verification (build fails on any mismatch):
  * every dataset table: sha256 of its rows (setup/fingerprints/dataset_rows_sha256.json)
  * every derived file: sha256 of the file (setup/fingerprints/derived_files.sha256)
The derived files are written by the same logic as on the original cluster (all67_build.py,
core24_build.py, build_mgsm_eval.py); prompt_of / completion_of are verbatim copies of the
project's q09 build_sft_sets.py.

    python setup/build_data.py fingerprint <data dir>   # (original cluster only) write the fingerprints
"""
import csv
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
DATA = BUNDLE / "data"
DL = DATA / "_downloads"
FP = BUNDLE / "setup" / "fingerprints"
DATASET_ID = "open-multilang/gsm8k-106"
DATASET_REVISION = "900d55906eb205eedd1120435439716666593646"
MGSM_COMMIT = "3622039cf51f7eeffa58b957332a8e8c337981d7"   # last change to mgsm/, 2024-05-28
MGSM_URL = "https://raw.githubusercontent.com/google-research/url-nlp/{commit}/mgsm/mgsm_{lang}.tsv"
MGSM_LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MGSM_CODE = {"zh": "zh-cn"}                 # MGSM zh is simplified Chinese = pipeline code zh-cn
N_TRAIN, N_TEST = 7473, 1319
FEW_SHOT_TEST_IDS = {0, 5}                  # the generation prompt's 2 examples: excluded from every test set
BROKEN_TEST_ITEMS = {"test_km_382"}         # Khmer translation of 13,183 characters (median 247): exceeds context
ANSWER_LINE = re.compile(r"^\s*####")


# ---- verbatim from ICM_work_/icm/experiments/q09_does_the_dataset_beat_translation/build_sft_sets.py
def prompt_of(question):
    return f"Question: {question.strip()}\nAnswer:"


def completion_of(text):
    """Solution text up to and including the first '#### N' line; None if there is none."""
    out = []
    for line in text.strip().split("\n"):
        out.append(line.rstrip())
        if ANSWER_LINE.match(line):
            return " " + "\n".join(out).strip()
    return None
# ----


def languages():
    sel = json.load(open(DATA / "language_selection.json"))
    core = {"en": "ref"}
    for t, v in sel["core"]["tiers"].items():
        for l in v["langs"]:
            core[l["code"]] = t
    ext = {l["code"]: t for t, v in sel["extension"]["per_tier"].items() for l in v}
    assert len(core) == 25 and len(ext) == 66
    return core, ext


def table_digest(ds):
    h = hashlib.sha256()
    for row in ds:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str).encode())
        h.update(b"\n")
    return h.hexdigest()


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download():
    from huggingface_hub import hf_hub_download
    core, ext = languages()
    langs = sorted(set(core) | set(ext))
    for split in ("train", "test"):
        for l in langs:
            hf_hub_download(DATASET_ID, f"data/{split}/{l}.jsonl", repo_type="dataset",
                            revision=DATASET_REVISION, local_dir=DL / "gsm8k-106")
        print(f"gsm8k-106 {split}: {len(langs)} languages")
    (DL / "mgsm").mkdir(parents=True, exist_ok=True)
    for l in MGSM_LANGS:
        urllib.request.urlretrieve(MGSM_URL.format(commit=MGSM_COMMIT, lang=l), DL / "mgsm" / f"mgsm_{l}.tsv")
    print(f"MGSM: {len(MGSM_LANGS)} languages -> {DL}")
    print("download done; now run:  python setup/build_data.py build")


def build():
    from datasets import load_dataset, load_from_disk
    core, ext = languages()
    langs = sorted(set(core) | set(ext))
    fails = []

    # 1. GSM8K-106 in the on-disk layout the generation code reads (<split>/<lang>, save_to_disk)
    want = json.load(open(FP / "dataset_rows_sha256.json"))
    out = DATA / "gsm8k-106"
    for split in ("train", "test"):
        for l in langs:
            if (out / split / l / "state.json").exists() and table_digest(load_from_disk(str(out / split / l))) == want[f"{split}/{l}"]:
                continue                                   # already built and verified (resumable)
            ds = load_dataset("json", data_files={split: str(DL / "gsm8k-106" / "data" / split / f"{l}.jsonl")},
                              split=split)
            ds.save_to_disk(str(out / split / l), max_shard_size="1GB")
            got = table_digest(load_from_disk(str(out / split / l)))
            if got != want[f"{split}/{l}"]:
                fails.append(f"dataset {split}/{l}")
    (out / "offline_manifest.json").write_text(json.dumps(
        {"dataset_id": DATASET_ID, "revision": DATASET_REVISION, "splits": ["train", "test"], "languages": langs}, indent=1))
    print(f"gsm8k-106: {2 * len(langs)} tables written and checked")

    def rows(split, l):
        return load_from_disk(str(out / split / l))

    # 2. all-67 training set, translated arm (as all67_build.py: en first, then the 66 sorted)
    (DATA / "all67_ablation").mkdir(parents=True, exist_ok=True)
    with open(DATA / "all67_ablation" / "sft_translated.jsonl", "w", encoding="utf-8") as h:
        for l in ["en"] + sorted(ext):
            by_id = {int(r["id"].rsplit("_", 1)[1]): r for r in rows("train", l)}
            assert set(by_id) == set(range(N_TRAIN)), l
            for i in range(N_TRAIN):
                comp = completion_of(by_id[i]["answer"])
                if comp is not None:
                    h.write(json.dumps({"id": by_id[i]["id"], "lang": l, "prompt": prompt_of(by_id[i]["question"]),
                                        "completion": comp}, ensure_ascii=False) + "\n")

    # 3. test sets (as core24_build.py, then the two documented removals)
    (DATA / "eval_sets").mkdir(parents=True, exist_ok=True)
    heldout = {l: t for l, t in ext.items() if l not in core}
    for name, group in (("test_core25_languages", core), ("test_42_extension_languages", heldout)):
        with open(DATA / "eval_sets" / f"{name}.jsonl", "w", encoding="utf-8") as h:
            for l in sorted(group):
                d = sorted(rows("test", l), key=lambda r: int(r["original_id"]))
                assert len(d) == N_TEST, l
                for r in d:
                    if int(r["original_id"]) in FEW_SHOT_TEST_IDS or r["id"] in BROKEN_TEST_ITEMS:
                        continue
                    h.write(json.dumps({"id": r["id"], "lang": l, "tier": group[l], "prompt": prompt_of(r["question"]),
                                        "gold": r["answer_number"]}, ensure_ascii=False) + "\n")

    # 4. MGSM human translations (as build_mgsm_eval.py; tiers = the pre-registered ones)
    md5 = dict(line.split()[::-1] for line in open(FP / "mgsm_md5sums.txt") if line.strip())
    with open(DATA / "eval_sets" / "test_mgsm_human_translations.jsonl", "w", encoding="utf-8") as h:
        for l in MGSM_LANGS:
            p = DL / "mgsm" / f"mgsm_{l}.tsv"
            if hashlib.md5(p.read_bytes()).hexdigest() != md5[f"mgsm_{l}.tsv"]:
                fails.append(f"MGSM source mgsm_{l}.tsv")
            code = MGSM_CODE.get(l, l)
            with open(p, encoding="utf-8", newline="") as f:
                src = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
            assert len(src) == 250, l
            for i, r in enumerate(src):
                if i in FEW_SHOT_TEST_IDS:
                    continue
                g = r[1].strip().replace(",", "")
                gold = int(g) if g.lstrip("-").isdigit() else float(g)
                h.write(json.dumps({"id": f"mgsm_{code}_{i}", "lang": code, "tier": core[code],
                                    "prompt": prompt_of(r[0]), "gold": gold}, ensure_ascii=False) + "\n")

    # 5. verify every derived file byte for byte
    for line in open(FP / "derived_files.sha256"):
        sha, rel = line.split()
        if file_sha(BUNDLE / rel) != sha:
            fails.append(f"derived file {rel}")
    if fails:
        sys.exit("BUILD FAILED, mismatches:\n  " + "\n  ".join(fails))
    print("DATA BUILT AND VERIFIED: every table and file matches the original cluster")


def fingerprint(data_dir):
    """Original cluster only: record what build() must reproduce."""
    from datasets import load_from_disk
    core, ext = languages()
    src = Path(data_dir)
    FP.mkdir(parents=True, exist_ok=True)
    d = {f"{s}/{l}": table_digest(load_from_disk(str(src / "gsm8k-106" / s / l)))
         for s in ("train", "test") for l in sorted(set(core) | set(ext))}
    json.dump(d, open(FP / "dataset_rows_sha256.json", "w"), indent=0, sort_keys=True)
    with open(FP / "derived_files.sha256", "w") as h:
        for rel in ("data/all67_ablation/sft_translated.jsonl", "data/eval_sets/test_core25_languages.jsonl",
                    "data/eval_sets/test_42_extension_languages.jsonl", "data/eval_sets/test_mgsm_human_translations.jsonl"):
            h.write(f"{file_sha(src.parent / rel)}  {rel}\n")
    print(f"fingerprints: {len(d)} tables, 4 derived files -> {FP}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        download()
    elif cmd == "build":
        build()
    elif cmd == "fingerprint":
        fingerprint(sys.argv[2])
    else:
        sys.exit(__doc__)
