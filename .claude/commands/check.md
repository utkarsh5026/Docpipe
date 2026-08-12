---
description: Run the full CI gate locally — 3.8 parse, zero-dependency import, tests, no-OpenCV tests, docs coverage
argument-hint: "[fast]  (fast = skip the slow full pytest run)"
allowed-tools: Bash, Read, Edit, Grep
---

Run everything `.github/workflows/ci.yml` and `docs.yml` run, in the same order, so a push
cannot fail on something reproducible here. Stop at the first failing gate, fix it, then
re-run from that gate onward.

If `$ARGUMENTS` contains `fast`, skip gate 4 (the full suite) and run only
`pytest -q tests/test_ir.py tests/test_preprocess.py tests/test_extract.py` — but say
clearly in the summary that the full suite was not run.

## Gate 1 — the Python 3.8 promise

```bash
python -c "import ast; ast.parse(open('docpipe.py', encoding='utf-8').read(), feature_version=(3,8)); print('3.8 OK')"
```

Failure means someone used `X | Y`, `match`, or a walrus. Rewrite with `Optional`/`Union`
from `typing` — do not raise the floor in `pyproject.toml` to make this pass.

## Gate 2 — the core imports with no optional dependencies

```bash
python - <<'PY'
import builtins
blocked = {"numpy", "cv2", "PIL", "fitz", "pymupdf", "pydantic"}
real = builtins.__import__
def guard(name, *a, **k):
    if name.split(".")[0] in blocked:
        raise ImportError("blocked for this check: " + name)
    return real(name, *a, **k)
builtins.__import__ = guard
import docpipe
assert docpipe.parse_amount("1,23,456.78") is not None
assert docpipe.fuse_confidence({"validation": 0.9}) > 0.5
assert docpipe.Text.parse_amount("1,23,456.78") is not None
assert docpipe.Confidence.fuse_confidence({"validation": 0.9}) > 0.5
print("core imports and runs with no optional dependencies")
PY
```

Failure means a heavy import escaped to module scope. Move it behind `require(...)` or
`have(...)` at the point of use — never a top-level `import numpy`.

## Gate 2b — every namespaced staticmethod keeps its module-level alias

The aliases are load-bearing: staticmethods call one another through the bare names,
resolved from module globals at call time, so a missing alias fails only at runtime.

```bash
python - <<'PY'
import docpipe, sys
names = ["Caps", "Util", "Image", "Quality", "Ingest", "Ops",
         "Policies", "Pricing", "Text", "Confidence", "Validators"]
bad = []
for cls_name in names:
    cls = getattr(docpipe, cls_name)
    for attr, raw in vars(cls).items():
        if attr.startswith("_") or not isinstance(raw, staticmethod):
            continue
        if getattr(docpipe, attr, None) is not getattr(cls, attr):
            bad.append("%s.%s" % (cls_name, attr))
if bad:
    sys.exit("missing or mismatched module-level alias: %s" % ", ".join(bad))
total = sum(len([a for a, r in vars(getattr(docpipe, n)).items()
                 if isinstance(r, staticmethod)]) for n in names)
print("%d staticmethods across %d namespaces, all aliased" % (total, len(names)))
PY
```

## Gate 3 — install state

```bash
pip install -e ".[dev]" 2>&1 | tail -3
```

Skip if already installed and unchanged.

## Gate 4 — the suite

```bash
pytest -q
```

~660 tests, roughly 75 seconds.

## Gate 5 — the pure-NumPy fallbacks

```bash
DOCPIPE_DISABLE_OPENCV=1 pytest -q tests/test_image_ops.py tests/test_quality.py tests/test_preprocess.py
```

Failure means a new image op has no NumPy fallback, or the fallback disagrees with OpenCV
beyond tolerance. Both are real defects — OpenCV is optional by promise.

## Gate 6 — the reference covers every public name

```bash
python docs/build_reference.py --out site/index.html
python - <<'PY'
import ast, io, sys
page = io.open("site/index.html", encoding="utf-8").read()
tree = ast.parse(io.open("docpipe.py", encoding="utf-8").read())
exported = []
for stmt in tree.body:
    if isinstance(stmt, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in stmt.targets):
        exported = [e.value for e in stmt.value.elts if isinstance(e, ast.Constant)]
missing = [n for n in exported if "id='%s'" % n not in page and not n.startswith("__")]
if missing:
    sys.exit("missing from the reference: %s" % ", ".join(missing))
print("all %d exported names are documented" % len(exported))
PY
```

Failure usually means a new public name landed in `__all__` without a docstring, or was
defined somewhere the generator doesn't scan. Fix the source, not the check.

`site/` is build output — do not commit it.

## Report

One line per gate: `PASS` / `FAIL` / `SKIPPED`, and for any failure the specific file and
line plus what you changed. If you fixed something, re-run that gate and every gate after it.
