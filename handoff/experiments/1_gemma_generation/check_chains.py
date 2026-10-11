"""Quality check of generated chains, per language: problems, chains per problem, share of EMPTY
chains (the failure the min_tokens rule fixes), share with a '#### N' answer line, and accuracy.
usage: python check_chains.py <directory searched recursively for generated/*.jsonl>
Fails (exit 1) if any language has more than 5% empty chains."""
import glob, json, sys
from collections import defaultdict
root = sys.argv[1]
files = sorted(glob.glob(f"{root}/**/generated/*.jsonl", recursive=True))
if not files:
    sys.exit(f"no generated/*.jsonl under {root}")
st = defaultdict(lambda: [0, 0, 0, 0, 0])   # problems, chains, empty, answer line, correct
for f in files:
    lang = f.rsplit("/", 1)[1][:-6]
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        s = st[lang]; s[0] += 1
        for c in r.get("candidates", []):
            s[1] += 1; s[2] += not c["text"].strip(); s[3] += "####" in c["text"]; s[4] += bool(c.get("correct"))
bad = []
print(f"{'lang':7}{'problems':>9}{'chains/pr':>10}{'empty':>7}{'ans-line':>9}{'acc':>6}")
for l, (p, n, e, a, c) in sorted(st.items()):
    print(f"{l:7}{p:>9}{n / max(1, p):>10.1f}{e / max(1, n):>7.3f}{a / max(1, n):>9.3f}{c / max(1, n):>6.3f}")
    if n and e / n > 0.05:
        bad.append(l)
if bad:
    sys.exit(f"FAIL: more than 5% empty chains in {bad}")
print(f"OK: {len(st)} languages, {sum(v[0] for v in st.values())} problem rows")
