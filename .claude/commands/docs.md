---
description: Rebuild the API reference and verify docstring/annotation coverage across docpipe.py
argument-hint: "[open]  (open = also print the output path to view locally)"
allowed-tools: Bash, Read, Edit, Grep
---

Rebuild and verify the generated API reference.

## Build

```bash
python docs/build_reference.py --out site/index.html
```

The generator `ast`-parses `docpipe.py` and never imports it, so it needs no dependencies —
not even docpipe's optional ones. `--fragment` omits the html/head/body wrapper for embedding.
`site/` is build output and must not be committed.

## Verify coverage

Every name in `__all__` must have an entry on the page:

```bash
python - <<'PY'
import ast, io, sys
page = io.open("site/index.html", encoding="utf-8").read()
tree = ast.parse(io.open("docpipe.py", encoding="utf-8").read())
exported = []
for stmt in tree.body:
    if isinstance(stmt, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in stmt.targets):
        exported = [e.value for e in stmt.value.elts if isinstance(e, ast.Constant)]
missing = [n for n in exported if "id='%s'" % n not in page and not n.startswith("__")]
print("exported: %d   missing: %s" % (len(exported), missing or "none"))
sys.exit(1 if missing else 0)
PY
```

Every function, method and class must have a docstring:

```bash
python - <<'PY'
import ast, io
tree = ast.parse(io.open("docpipe.py", encoding="utf-8").read())
bad = []
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if not ast.get_docstring(node):
            bad.append("%s:%d %s" % ("docpipe.py", node.lineno, node.name))
print("undocumented: %s" % (bad or "none"))
PY
```

## Fix what the checks find

Undocumented or missing entries are fixed **in the source**, never by loosening the check.
The generator renders the conventions the file already uses:

- `` :func:`x` `` / `` :class:`X` `` roles become cross-links — use them when a docstring
  mentions another public name
- `:param name:` blocks become definition lists
- `#:` comments above a constant or dataclass field document it
- prose under a `# === N. Title ===` banner becomes that section's introduction

A docstring should say what the code does not already show: why a threshold is where it is,
what a `to_dict` omits and why, what returning `None` means, which failures are deliberately
swallowed.

## Also confirm annotations

Signatures are fully annotated, and PEP 563 is in force so forward references need no quotes.
Optional-dependency types resolve to the aliases declared after the imports (`ImageArray`,
`GrayImage`, `FloatArray`, `Module`, `Source`, `JSONDict`, `SchemaLike`) — use the alias that
carries the real contract rather than a bare `Any`.

If `$ARGUMENTS` contains `open`, print the absolute path to `site/index.html` at the end.

Report: names exported, anything missing, anything undocumented, and what you fixed.
