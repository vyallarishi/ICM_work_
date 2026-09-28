"""Static import check for the icm/ package (no code is executed).

For every relative import: the target module must exist, and every imported
name must be bound at the top level of that module. Also fails if any module
still imports a ~/phase0 name (stageNN_*, rung*, ...) absolutely -- that would
silently pick up the original instead of the copy -- or uses RUN_DIR/OUT_DIR
before the import that binds it.

    python tools/check_imports.py
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "icm"
old_names = {Path(v).stem
             for v in json.loads((PKG / "provenance.json").read_text())["python"].values()}


def top_level_names(tree):
    names = set()
    for node in tree.body:
        for n in ast.walk(node) if isinstance(node, (ast.If, ast.Try)) else [node]:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
    return names


def resolve(mod_path, level, module):
    base = mod_path.parent
    for _ in range(level - 1):
        base = base.parent
    return base.joinpath(*(module.split(".") if module else []))


errors, n_imports = [], 0
modules = sorted(PKG.rglob("*.py"))
for path in modules:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in old_names:
                    errors.append(f"{rel}:{node.lineno} absolute import of phase0 module {a.name}")
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module and node.module.split(".")[0] in old_names:
                errors.append(f"{rel}:{node.lineno} absolute import of phase0 module {node.module}")
            continue
        n_imports += 1
        target = resolve(path, node.level, node.module)
        if target.with_suffix(".py").is_file():
            names = top_level_names(ast.parse(target.with_suffix(".py").read_text(encoding="utf-8")))
            for a in node.names:
                if a.name != "*" and a.name not in names:
                    errors.append(f"{rel}:{node.lineno} {a.name!r} not defined in "
                                  f"{target.with_suffix('.py').relative_to(ROOT)}")
        elif (target / "__init__.py").is_file():
            for a in node.names:  # `from ..core import pools as gate`
                if not (target / f"{a.name}.py").is_file():
                    errors.append(f"{rel}:{node.lineno} no module {a.name} in {target.relative_to(ROOT)}")
        else:
            errors.append(f"{rel}:{node.lineno} unresolved relative import (level={node.level}, "
                          f"module={node.module})")

# RUN_DIR / OUT_DIR come from an import added during the move: it must precede every use
for path in modules:
    lines = path.read_text(encoding="utf-8").split("\n")
    for name in ("RUN_DIR", "OUT_DIR"):
        bind = [i for i, l in enumerate(lines) if "import RUN_DIR" in l and
                (name == "RUN_DIR" and l.strip().endswith("RUN_DIR") or f"as {name}" in l)]
        uses = [i for i, l in enumerate(lines) if name in l and i not in bind
                and not l.lstrip().startswith("#")]
        if bind and uses and min(uses) < min(bind):
            errors.append(f"{path.relative_to(ROOT)}:{min(uses)+1} uses {name} before its import "
                          f"at line {min(bind)+1}")

print(f"modules: {len(modules)}, relative imports checked: {n_imports}")
if errors:
    print(f"FAIL: {len(errors)}")
    for e in errors:
        print("  " + e)
    sys.exit(1)
print("PASS: every relative import resolves and every imported name exists")
