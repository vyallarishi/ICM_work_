"""Verify every icm/experiments module against its ~/phase0 original.

Each differing line must be one of the four permitted edits (see README,
"What changed"); anything else is reported as a FAIL. Also checks that
generation/ is byte-identical to git HEAD of ~/multilingual_icm.

    python tools/check_provenance.py
"""
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = Path.home() / "phase0"
GEN_REPO = Path.home() / "multilingual_icm"
PHASE0_LIT = "/home/rvyalla/phase0"

prov = json.loads((ROOT / "icm" / "provenance.json").read_text())["python"]
local = {p.stem for p in PHASE0.glob("*.py")}


def is_import_rewrite(old, new):
    """old names a phase0 module; new is the same statement with a relative module."""
    o, n = old.strip(), new.strip()
    m = re.match(r"from\s+(\w+)\s+import\s+(.*)$", o)
    if m and m.group(1) in local:
        m2 = re.match(r"from\s+\.+[\w.]*\s+import\s+(.*)$", n)
        return bool(m2) and m2.group(1) == m.group(2)
    m = re.match(r"import\s+(\w+)(?:\s+as\s+(\w+))?\s*(#.*)?$", o)
    if m and m.group(1) in local:
        alias = m.group(2) or m.group(1)
        m2 = re.match(rf"from\s+\.+[\w.]*\s+import\s+(\w+)\s+as\s+{alias}\s*(#.*)?$", n)
        return bool(m2) and (m2.group(2) or "") == (m.group(3) or "")
    m = re.match(r"import\s+(.*)$", o)  # `import a, stage8 as g, b` split onto one line
    if m and ";" in n:
        names = [x.strip() for x in m.group(1).split(",")]
        loc = [x for x in names if x.split()[0] in local]
        rest = ", ".join(x for x in names if x not in loc)
        if len(loc) == 1:
            alias = loc[0].split()[-1]
            return bool(re.match(rf"from\s+\.+[\w.]*\s+import\s+\w+\s+as\s+{alias};\s+import\s+{re.escape(rest)}$", n))
    return False


def classify(old, new):
    if old is None:
        return "added RUN_DIR import" if re.fullmatch(r"from \.+paths import RUN_DIR", new.strip()) else None
    if new is None:
        return None
    if is_import_rewrite(old, new):
        return "import"
    if re.fullmatch(r"\s*OUT_DIR = Path\(__file__\)\.parent\s*", old) and \
            re.fullmatch(r"\s*from \.+paths import RUN_DIR as OUT_DIR  # was: Path\(__file__\)\.parent", new):
        return "OUT_DIR"
    if "sys.path" in old and PHASE0_LIT in old and "[ICM_work_] removed" in new:
        return "sys.path removed"
    if "Path(__file__).with_name(" in old and \
            re.sub(r'Path\(__file__\)\.with_name\(("[^"]*")\)', r"RUN_DIR / \1", old) == new:
        return "with_name -> RUN_DIR"
    if PHASE0_LIT in old:
        back = new.replace('f"{RUN_DIR}/', 'f"' + PHASE0_LIT + "/")
        if old == back or re.sub(r'f"' + re.escape(PHASE0_LIT), '"' + PHASE0_LIT, back) == old:
            return "phase0 path -> RUN_DIR"
    return None


fails, counts = [], {}
for new_rel, orig in sorted(prov.items()):
    a = (PHASE0 / orig).read_text(encoding="utf-8").split("\n")
    b = (ROOT / new_rel).read_text(encoding="utf-8").split("\n")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        olds, news = a[i1:i2], b[j1:j2]
        if tag == "replace" and len(olds) != len(news):
            ins = [k for k, l in enumerate(news) if re.fullmatch(r"from \.+paths import RUN_DIR", l.strip())]
            if len(ins) == 1 and len(news) == len(olds) + 1:
                counts["added RUN_DIR import"] = counts.get("added RUN_DIR import", 0) + 1
                news = news[: ins[0]] + news[ins[0] + 1:]
            else:
                fails.append((new_rel, i1 + 1, "\n".join(olds), "\n".join(news)))
                continue
        pairs = list(zip(olds, news)) if tag == "replace" else \
            [(None, n) for n in news] if tag == "insert" else [(o, None) for o in olds]
        for o, n in pairs:
            kind = classify(o, n)
            if kind is None:
                fails.append((new_rel, i1 + 1, o, n))
            else:
                counts[kind] = counts.get(kind, 0) + 1

print(f"experiment modules checked: {len(prov)}")
for k, v in sorted(counts.items()):
    print(f"  {v:4d} x {k}")

# generation side: tracked files of ~/multilingual_icm HEAD, plus uv.lock
tracked = subprocess.run(["git", "-C", GEN_REPO, "ls-files"], capture_output=True, text=True,
                         check=True).stdout.split()
gen_files = [f for f in tracked if not f.startswith(("organized_rishi/", "probe_core/"))]
gen_bad = 0
for f in gen_files:
    head = subprocess.run(["git", "-C", GEN_REPO, "show", f"HEAD:{f}"], capture_output=True,
                          check=True).stdout
    if (ROOT / "generation" / f).read_bytes() != head:
        gen_bad += 1
        print(f"  GENERATION DIFF: {f}")
if (ROOT / "generation" / "uv.lock").read_bytes() != (GEN_REPO / "uv.lock").read_bytes():
    gen_bad += 1
    print("  GENERATION DIFF: uv.lock")
print(f"generation files checked: {len(gen_files) + 1}, differing: {gen_bad}")

if fails:
    print(f"\nFAIL: {len(fails)} unexplained line changes")
    for rel, ln, o, n in fails[:40]:
        print(f"  {rel}:{ln}\n    - {o!r}\n    + {n!r}")
    sys.exit(1)
if gen_bad:
    sys.exit(1)
print("\nPASS: every change is one of the permitted edit types")
