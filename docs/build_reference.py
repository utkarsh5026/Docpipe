#!/usr/bin/env python3
"""Build the docpipe API reference as a single self-contained HTML page.

Reads ``docpipe.py`` with :mod:`ast` -- it is never imported, so the reference
builds without any of the optional dependencies installed -- and renders every
class, function, and documented constant, grouped by the numbered sections the
source already organises itself into.

Standard library only, deliberately: docpipe itself carries no dependencies and
its documentation should not be the thing that reintroduces a toolchain.

::

    python docs/build_reference.py              # -> site/index.html
    python docs/build_reference.py -o out.html
"""

from __future__ import annotations

import argparse
import ast
import html
import io
import os
import re
import sys
import textwrap
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(ROOT, "docpipe.py")

#: Blob URL for "source" links.  Overridable so a fork's pages link to itself.
REPO = os.environ.get("DOCPIPE_REPO", "utkarsh5026/Docpipe")
REF = os.environ.get("DOCPIPE_REF", "main")


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class Entry(object):
    """One documented object: a class, a function, or a constant."""

    def __init__(self, name, kind, signature, doc, lineno, public=False,
                 decorators=None, bases=None):
        self.name = name
        self.kind = kind                      # class | function | data
        self.signature = signature
        self.doc = doc or ""
        self.lineno = lineno
        self.public = public
        self.decorators = decorators or []
        self.bases = bases or []
        self.members: List[Entry] = []        # methods / documented attributes
        self.is_field = False                 # a dataclass constructor field
        self.is_dataclass = False

    @property
    def anchor(self) -> str:
        return self.name.replace(".", "-")


class Section(object):
    """One numbered ``# N. Title`` banner in the source, and what follows it."""

    def __init__(self, number, title, lineno, intro=""):
        self.number = number
        self.title = title
        self.lineno = lineno
        self.intro = intro
        self.entries: List[Entry] = []

    @property
    def anchor(self) -> str:
        return "section-%s" % self.number


# -----------------------------------------------------------------------------
# Source extraction
# -----------------------------------------------------------------------------

BANNER_RE = re.compile(r"^# (\d+)\.\s+(.*?)\s*$")


def read_sections(lines: List[str]) -> List[Section]:
    """Find the ``# N. Title`` banners and the comment prose beneath each."""
    sections = []
    for i, line in enumerate(lines):
        m = BANNER_RE.match(line.rstrip("\n"))
        if not m:
            continue
        # A real banner is fenced by ``# ====`` rules above and below.
        above = lines[i - 1].strip() if i else ""
        below = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not (above.startswith("# ===") and below.startswith("# ===")):
            continue
        intro = []
        j = i + 2
        while j < len(lines) and lines[j].lstrip().startswith("#"):
            intro.append(re.sub(r"^#\s?", "", lines[j].strip()))
            j += 1
        sections.append(Section(m.group(1), m.group(2), i + 1,
                                "\n".join(intro).strip()))
    return sections


def signature_of(lines: List[str], node: ast.AST) -> str:
    """The exact source text of a def's signature, minus the trailing colon."""
    start = node.lineno - 1
    end = node.body[0].lineno - 1
    # Walk back over comment/blank lines between the signature and the body.
    while end > start and not lines[end - 1].rstrip().endswith(":"):
        end -= 1
    text = "".join(lines[start:end]).rstrip()
    text = textwrap.dedent(text).rstrip()
    if text.endswith(":"):
        text = text[:-1]
    return re.sub(r"^(async\s+)?def\s+", "", text, count=1)


def class_signature(lines: List[str], node: ast.ClassDef) -> str:
    """``ClassName(Base, ...)`` as written in the source."""
    bases = [ast.unparse(b) for b in node.bases]
    bases = [b for b in bases if b != "object"]
    return node.name + ("(%s)" % ", ".join(bases) if bases else "")


def attribute_docs(lines: List[str], node: ast.ClassDef) -> List[Entry]:
    """Class attributes carrying a Sphinx ``#:`` doc comment, or a dataclass field."""
    out = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            declared = ast.unparse(stmt.annotation)
            default = " = " + ast.unparse(stmt.value) if stmt.value else ""
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            declared = ""
            default = " = " + ast.unparse(stmt.value)
        else:
            continue
        if name.startswith("__"):
            continue
        # A trailing `#:` comment on the same line documents the attribute.
        line = lines[stmt.lineno - 1]
        doc = ""
        m = re.search(r"#:\s?(.*)$", line)
        if m:
            doc = m.group(1).strip()
        else:
            # ...or one or more `#:` lines immediately above it.
            above = []
            r = stmt.lineno - 2
            while r >= 0 and lines[r].lstrip().startswith("#:"):
                above.insert(0, re.sub(r"^#:\s?", "", lines[r].strip()))
                r -= 1
            doc = " ".join(above).strip()
        sig = name + (": " + declared if declared else "") + default
        entry = Entry(name, "attribute", sig, doc, stmt.lineno)
        # An annotated assignment in a dataclass body is a constructor field,
        # which is worth listing even when it carries no `#:` comment.
        entry.is_field = isinstance(stmt, ast.AnnAssign)
        out.append(entry)
    return out


def module_constants(lines: List[str], tree: ast.Module,
                     exported: set) -> List[Tuple[int, Entry]]:
    """Module-level constants that are exported or carry a ``#:`` comment."""
    out = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
        else:
            continue
        if name.startswith("__") or name == "logger":
            continue
        doc = ""
        above = []
        r = stmt.lineno - 2
        while r >= 0 and lines[r].lstrip().startswith("#:"):
            above.insert(0, re.sub(r"^#:\s?", "", lines[r].strip()))
            r -= 1
        if above:
            doc = " ".join(above).strip()
        if not doc and name not in exported:
            continue
        value = ast.unparse(stmt.value) if getattr(stmt, "value", None) else ""
        if len(value) > 90:
            value = value[:88] + "..."
        out.append((stmt.lineno, Entry(name, "data", "%s = %s" % (name, value),
                                       doc, stmt.lineno, name in exported)))
    return out


def collect(path: str):
    """Parse the source into sections full of entries."""
    with io.open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    module_doc = ast.get_docstring(tree) or ""

    exported = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
                getattr(t, "id", "") == "__all__" for t in stmt.targets):
            for elt in stmt.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    exported.add(elt.value)

    sections = read_sections(lines)
    if not sections:
        raise SystemExit("no numbered section banners found in %s" % path)

    placed: List[Tuple[int, Entry]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            entry = Entry(stmt.name, "class", class_signature(lines, stmt),
                          ast.get_docstring(stmt), stmt.lineno,
                          stmt.name in exported,
                          [ast.unparse(d) for d in stmt.decorator_list],
                          [ast.unparse(b) for b in stmt.bases])
            entry.is_dataclass = any("dataclass" in d for d in entry.decorators)
            entry.members.extend(attribute_docs(lines, stmt))
            for sub in stmt.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if sub.name.startswith("__") and sub.name not in (
                            "__init__", "__call__", "__getitem__", "__contains__",
                            "__len__", "__iter__", "__add__"):
                        continue
                    entry.members.append(Entry(
                        "%s.%s" % (stmt.name, sub.name), "method",
                        signature_of(lines, sub), ast.get_docstring(sub),
                        sub.lineno,
                        decorators=[ast.unparse(d) for d in sub.decorator_list]))
            placed.append((stmt.lineno, entry))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            placed.append((stmt.lineno, Entry(
                stmt.name, "function", signature_of(lines, stmt),
                ast.get_docstring(stmt), stmt.lineno, stmt.name in exported,
                [ast.unparse(d) for d in stmt.decorator_list])))
    placed.extend(module_constants(lines, tree, exported))
    placed.sort(key=lambda p: p[0])

    for lineno, entry in placed:
        target = None
        for section in sections:
            if section.lineno <= lineno:
                target = section
            else:
                break
        if target is not None:
            target.entries.append(entry)
    return module_doc, sections, exported


# -----------------------------------------------------------------------------
# reStructuredText-lite -> HTML
# -----------------------------------------------------------------------------

ROLE_RE = re.compile(r":(class|func|meth|data|attr|mod|exc):`~?([^`]+)`")
LITERAL_RE = re.compile(r"``(.+?)``")
KNOWN: set = set()


def _inline(text: str) -> str:
    """Escape, then apply inline reST roles and literals."""
    out = html.escape(text)

    def role(m):
        target = m.group(2)
        short = target.split(".")[-1]
        anchor = target.replace(".", "-")
        if target in KNOWN:
            return '<a class="ref" href="#%s"><code>%s</code></a>' % (anchor, short)
        if short in KNOWN:
            return '<a class="ref" href="#%s"><code>%s</code></a>' % (short, short)
        return "<code>%s</code>" % short

    out = ROLE_RE.sub(role, out)
    # Emphasis, but never across a literal and never over a ``**kwargs``.
    out = re.sub(r"(?<![\w*`])\*([^\s*][^*`]*?)\*(?![\w*])",
                 lambda m: "<em>%s</em>" % m.group(1), out)
    out = LITERAL_RE.sub(lambda m: "<code>%s</code>" % m.group(1), out)
    out = re.sub(r"(?<!\w)--(?!\w)", "&#8212;", out)
    return out


def _is_rule(line: str, char: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {char, " "} and char in stripped


def render_doc(text: str, heading_level: int = 3) -> str:
    """Render a docstring's reST-lite markup as HTML."""
    if not text:
        return ""
    lines = textwrap.dedent(text).strip("\n").split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Simple table: a rule of '=' delimits it top and bottom.
        if _is_rule(line, "=") and i + 1 < n:
            border = line.rstrip("\n")
            spans = [(m.start(), m.end()) for m in re.finditer(r"=+", border)]
            rows = []
            i += 1
            while i < n and not _is_rule(lines[i], "="):
                if lines[i].strip():
                    raw = lines[i].rstrip("\n")
                    cells = []
                    for k, (a, b) in enumerate(spans):
                        cells.append(raw[a:b if k < len(spans) - 1 else None].strip())
                    # A cell wider than its column (which real docstrings do
                    # contain) would be sliced mid-word by the fixed spans, so
                    # fall back to splitting on the run of spaces instead.
                    overflows = any(
                        len(raw) > b and raw[b - 1:b + 1].strip() and k < len(spans) - 1
                        for k, (a, b) in enumerate(spans))
                    if overflows:
                        cells = re.split(r"\s{2,}", raw.strip(), len(spans) - 1)
                        cells += [""] * (len(spans) - len(cells))
                    rows.append(cells)
                i += 1
            i += 1
            if rows:
                out.append("<table class='doc-table'><tbody>")
                for row in rows:
                    out.append("<tr>" + "".join(
                        "<td>%s</td>" % _inline(c) for c in row) + "</tr>")
                out.append("</tbody></table>")
            continue

        # Section heading: text underlined by ---- or ==== or ~~~~.
        if i + 1 < n and any(_is_rule(lines[i + 1], c) for c in "-=~^"):
            if len(lines[i + 1].strip()) >= max(3, len(stripped) - 2):
                out.append("<h%d class='doc-h'>%s</h%d>"
                           % (heading_level, _inline(stripped), heading_level))
                i += 2
                continue

        # Literal block introduced by a trailing '::'.
        if stripped.endswith("::"):
            lead = stripped[:-2].rstrip()
            if lead:
                out.append("<p>%s:</p>" % _inline(lead))
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            block = []
            base = None
            while i < n:
                if not lines[i].strip():
                    block.append("")
                    i += 1
                    continue
                indent = len(lines[i]) - len(lines[i].lstrip())
                if base is None:
                    base = indent
                if indent < base:
                    break
                block.append(lines[i][base:].rstrip())
                i += 1
            while block and not block[-1]:
                block.pop()
            out.append("<pre class='code'><code>%s</code></pre>"
                       % html.escape("\n".join(block)))
            continue

        # Field list: :param x: ..., :returns: ..., :raises X: ...
        if re.match(r"^:(param|returns?|raises|rtype|type)\b", stripped):
            items = []
            while i < n:
                m = re.match(r"^:(param|returns?|raises|rtype|type)\s*([^:]*):\s*(.*)$",
                             lines[i].strip())
                if not m:
                    break
                kind, name, rest = m.group(1), m.group(2).strip(), m.group(3)
                body = [rest]
                i += 1
                while i < n and lines[i].strip() and not lines[i].strip().startswith(":") \
                        and (len(lines[i]) - len(lines[i].lstrip())) > 0:
                    body.append(lines[i].strip())
                    i += 1
                label = name if kind == "param" else kind
                items.append((label, " ".join(b for b in body if b)))
            out.append("<dl class='fields'>")
            for label, body in items:
                out.append("<dt><code>%s</code></dt><dd>%s</dd>"
                           % (html.escape(label), _inline(body)))
            out.append("</dl>")
            continue

        # Bullet list.
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            tag = "ol" if ordered else "ul"
            items = []
            while i < n:
                s = lines[i].strip()
                m = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", s)
                if not m:
                    if s and items and (len(lines[i]) - len(lines[i].lstrip())) >= 2:
                        items[-1] += " " + s
                        i += 1
                        continue
                    break
                items.append(m.group(1))
                i += 1
            out.append("<%s>%s</%s>" % (
                tag, "".join("<li>%s</li>" % _inline(x) for x in items), tag))
            continue

        # Definition list: an unindented term followed by an indented body.
        # (Used for the `method` options in estimate_skew, for example.)
        if i + 1 < n and lines[i + 1].strip() and \
                (len(lines[i + 1]) - len(lines[i + 1].lstrip())) > \
                (len(line) - len(line.lstrip())) and not stripped.endswith((".", ":", ",")):
            term = stripped
            body = []
            i += 1
            indent = len(lines[i]) - len(lines[i].lstrip())
            while i < n and (not lines[i].strip() or
                             (len(lines[i]) - len(lines[i].lstrip())) >= indent):
                if not lines[i].strip():
                    if i + 1 < n and lines[i + 1].strip() and \
                            (len(lines[i + 1]) - len(lines[i + 1].lstrip())) < indent:
                        break
                    body.append("")
                else:
                    body.append(lines[i].strip())
                i += 1
            out.append("<dl class='defs'><dt><code>%s</code></dt><dd>%s</dd></dl>"
                       % (_inline(term), _inline(" ".join(b for b in body if b))))
            continue

        # Plain paragraph.
        para = [stripped]
        i += 1
        while i < n and lines[i].strip() and \
                not re.match(r"^[-*:]\s|^\d+\.\s", lines[i].strip()) and \
                not lines[i].strip().endswith("::") and \
                not _is_rule(lines[i], "=") and \
                not (i + 1 < n and any(_is_rule(lines[i + 1], c) for c in "-=~^")):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>%s</p>" % _inline(" ".join(para)))

    return "\n".join(out)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def highlight_signature(sig: str) -> str:
    """Very light syntax colouring for a signature: name, params, annotations."""
    escaped = html.escape(sig)
    # Annotations and defaults after ':' or '=' up to a comma at depth 0.
    escaped = re.sub(r"(-&gt;\s*)([^:]+)$",
                     lambda m: m.group(1) + "<span class='ann'>%s</span>" % m.group(2),
                     escaped)
    escaped = re.sub(r"(:\s)([A-Za-z_][\w\.\[\], &;\"']*)",
                     lambda m: m.group(1) + "<span class='ann'>%s</span>" % m.group(2),
                     escaped)
    escaped = re.sub(r"^([A-Za-z_]\w*)", r"<span class='fn'>\1</span>", escaped)
    return escaped


def source_link(lineno: int) -> str:
    return "https://github.com/%s/blob/%s/docpipe.py#L%d" % (REPO, REF, lineno)


def render_entry(entry: Entry, level: int = 2) -> str:
    kind_label = {"class": "class", "function": "func", "method": "method",
                  "data": "const", "attribute": "attr"}[entry.kind]
    if entry.kind == "method" and any("property" in d for d in entry.decorators):
        kind_label = "property"
    elif entry.kind == "method" and any("classmethod" in d for d in entry.decorators):
        kind_label = "classmethod"
    elif entry.kind == "method" and any("staticmethod" in d for d in entry.decorators):
        kind_label = "staticmethod"
    badge = "<span class='badge %s'>%s</span>" % (entry.kind, kind_label)
    if entry.is_dataclass:
        badge += "<span class='badge'>dataclass</span>"
    public = "<span class='badge public'>public API</span>" if entry.public else ""
    deco = ""
    if entry.decorators:
        shown = [d for d in entry.decorators
                 if not d.startswith(("dataclass", "staticmethod", "classmethod",
                                      "property", "functools.wraps"))]
        if shown:
            deco = "<div class='decorators'>%s</div>" % "".join(
                "<span>@%s</span>" % html.escape(d) for d in shown)
    out = [
        "<article class='entry' id='%s' data-name='%s'>" % (entry.anchor,
                                                            entry.name.lower()),
        "<div class='entry-head'>",
        "<h%d>%s</h%d>" % (level, html.escape(entry.name), level),
        badge, public,
        "<a class='src' href='%s' title='view source'>source</a>" % source_link(entry.lineno),
        "</div>",
        deco,
    ]
    # For a class with no bases the signature is just its name, which the
    # heading already says; the Fields list below carries the real information.
    if entry.signature != entry.name:
        out.append("<pre class='sig'><code>%s</code></pre>"
                   % highlight_signature(entry.signature))
    out.append("<div class='doc'>%s</div>"
               % render_doc(entry.doc, heading_level=level + 2))
    fields = [m for m in entry.members if m.is_field] if entry.is_dataclass else []
    attrs = [m for m in entry.members
             if m.kind == "attribute" and m.doc and not m.is_field]
    methods = [m for m in entry.members if m.kind == "method"]
    if fields:
        out.append("<h4 class='sub'>Fields</h4><dl class='attrs'>")
        for f in fields:
            out.append("<dt><code>%s</code></dt><dd>%s</dd>"
                       % (highlight_signature(f.signature), _inline(f.doc)))
        out.append("</dl>")
    if attrs:
        out.append("<h4 class='sub'>Attributes</h4><dl class='attrs'>")
        for a in attrs:
            out.append("<dt><code>%s</code></dt><dd>%s</dd>"
                       % (highlight_signature(a.signature), _inline(a.doc)))
        out.append("</dl>")
    if methods:
        out.append("<h4 class='sub'>Methods</h4><div class='methods'>")
        for m in methods:
            out.append(render_entry(m, level=level + 1))
        out.append("</div>")
    out.append("</article>")
    return "\n".join(out)


CSS = """
:root {
  --bg: #ffffff; --fg: #1b1f24; --muted: #5b6673; --rule: #e3e8ef;
  --panel: #f7f9fc; --code-bg: #f2f5f9; --accent: #0b62d0; --accent-soft: #e7f0fd;
  --sig-bg: #f7f9fc; --badge: #eef2f7; --ann: #7a3fa8; --fn: #0b62d0;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1319; --fg: #e6edf3; --muted: #9aa7b4; --rule: #232b36;
    --panel: #151b23; --code-bg: #161d26; --accent: #62a8ff; --accent-soft: #12233c;
    --sig-bg: #131a22; --badge: #1d2530; --ann: #d2a8ff; --fn: #79b8ff;
    --shadow: none;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1319; --fg: #e6edf3; --muted: #9aa7b4; --rule: #232b36;
  --panel: #151b23; --code-bg: #161d26; --accent: #62a8ff; --accent-soft: #12233c;
  --sig-bg: #131a22; --badge: #1d2530; --ann: #d2a8ff; --fn: #79b8ff;
  --shadow: none;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
code, pre, .mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo,
  Consolas, "Liberation Mono", monospace; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

/* ---- sidebar ---- */
.sidebar {
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  border-right: 1px solid var(--rule); background: var(--panel); padding: 20px 0 60px;
}
@media (max-width: 900px) {
  .sidebar { position: static; height: auto; border-right: 0;
             border-bottom: 1px solid var(--rule); }
}
.brand { padding: 0 20px 14px; }
.brand h1 { margin: 0; font-size: 20px; letter-spacing: -.01em; }
.brand .ver { color: var(--muted); font-size: 12px; }
.searchbox { padding: 0 20px 14px; }
.searchbox input {
  width: 100%; padding: 8px 10px; border-radius: 8px; font-size: 14px;
  border: 1px solid var(--rule); background: var(--bg); color: var(--fg);
}
.searchbox input:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
.nav { padding: 0 8px; }
.nav details { margin: 0 0 2px; }
.nav summary {
  cursor: pointer; padding: 6px 12px; border-radius: 6px; font-size: 13px;
  font-weight: 600; color: var(--fg); list-style: none;
}
.nav summary::-webkit-details-marker { display: none; }
.nav summary:hover { background: var(--accent-soft); }
.nav summary .num { color: var(--muted); font-weight: 400; margin-right: 6px; }
.nav ul { list-style: none; margin: 2px 0 8px; padding: 0 0 0 12px; }
.nav li a {
  display: block; padding: 3px 10px; font-size: 12.5px; color: var(--muted);
  border-radius: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.nav li a:hover { background: var(--accent-soft); color: var(--accent); text-decoration: none; }
.nav li a.hit { color: var(--fg); }

/* ---- content ---- */
main { padding: 32px 40px 120px; max-width: 980px; }
@media (max-width: 640px) { main { padding: 24px 18px 80px; } }
.section { margin: 0 0 56px; scroll-margin-top: 16px; }
.section > h2 {
  font-size: 22px; margin: 0 0 6px; padding-bottom: 8px;
  border-bottom: 1px solid var(--rule); letter-spacing: -.01em;
}
.section > h2 .num { color: var(--muted); font-weight: 400; }
.section-intro { color: var(--muted); font-size: 14px; margin: 12px 0 26px; }
.section-intro p { margin: .5em 0; }

.entry { margin: 0 0 30px; padding: 0 0 4px; scroll-margin-top: 16px; }
.entry-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.entry-head h2, .entry-head h3, .entry-head h4 {
  margin: 0; font-size: 17px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: -.01em;
}
.entry-head h3 { font-size: 15px; }
.badge {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 20px; background: var(--badge); color: var(--muted);
  font-weight: 600;
}
.badge.public { background: var(--accent-soft); color: var(--accent); }
.src { margin-left: auto; font-size: 12px; color: var(--muted); }
.decorators { margin: 6px 0 0; font-size: 12px; color: var(--muted);
              font-family: ui-monospace, Menlo, monospace; }
pre.sig {
  margin: 8px 0 12px; padding: 10px 12px; background: var(--sig-bg);
  border: 1px solid var(--rule); border-radius: 8px; overflow-x: auto;
  font-size: 13px; line-height: 1.5;
}
pre.sig .fn { color: var(--fn); font-weight: 600; }
pre.sig .ann { color: var(--ann); }
.doc p { margin: .6em 0; }
.doc code, .attrs code, .fields code {
  background: var(--code-bg); padding: .12em .38em; border-radius: 4px; font-size: .9em;
}
pre.code {
  background: var(--code-bg); border: 1px solid var(--rule); border-radius: 8px;
  padding: 12px 14px; overflow-x: auto; font-size: 13px; line-height: 1.5;
}
pre.code code { background: none; padding: 0; }
.doc-h { font-size: 14px; margin: 1.2em 0 .4em; text-transform: uppercase;
         letter-spacing: .04em; color: var(--muted); }
dl.fields, dl.defs, dl.attrs { margin: .6em 0; }
dl.fields dt, dl.defs dt, dl.attrs dt { font-weight: 600; margin-top: .5em; }
dl.fields dd, dl.defs dd, dl.attrs dd { margin: .15em 0 .15em 1.4em; color: var(--fg); }
dl.attrs dt code { background: none; padding: 0; color: var(--fg); }
table.doc-table { border-collapse: collapse; margin: .8em 0; font-size: 13.5px;
                  display: block; overflow-x: auto; }
table.doc-table td { border-top: 1px solid var(--rule); padding: 6px 14px 6px 0;
                     vertical-align: top; }
h4.sub { font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
         color: var(--muted); margin: 18px 0 8px; }
.methods { border-left: 2px solid var(--rule); padding-left: 18px; }
.methods .entry { margin-bottom: 18px; }
.hidden { display: none !important; }
.toolbar { display: flex; gap: 10px; align-items: center; margin: 0 0 26px;
           flex-wrap: wrap; }
.toolbar button {
  font: inherit; font-size: 13px; padding: 5px 11px; border-radius: 7px; cursor: pointer;
  border: 1px solid var(--rule); background: var(--bg); color: var(--fg);
}
.toolbar button:hover { border-color: var(--accent); color: var(--accent); }
.toolbar .count { color: var(--muted); font-size: 13px; margin-left: auto; }
.overview { margin: 0 0 48px; }
.overview h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -.02em; }
.lede { color: var(--muted); font-size: 15.5px; margin: 0 0 20px; }
.noresults { color: var(--muted); padding: 20px 0; }
"""

JS = """
(function () {
  var input = document.getElementById('q');
  var entries = Array.prototype.slice.call(
      document.querySelectorAll('main .section > .entry'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('main .section'));
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.nav li a'));
  var count = document.getElementById('count');
  var privToggle = document.getElementById('toggle-private');
  var showPrivate = false;

  function isPrivate(el) {
    var n = el.getAttribute('data-name') || '';
    return n.charAt(0) === '_';
  }

  function apply() {
    var q = (input.value || '').trim().toLowerCase();
    var shown = 0;
    entries.forEach(function (el) {
      var name = el.getAttribute('data-name') || '';
      var hay = name + ' ' + (el.textContent || '').toLowerCase();
      var matches = !q || hay.indexOf(q) !== -1;
      var allowed = showPrivate || !isPrivate(el);
      var visible = matches && allowed;
      el.classList.toggle('hidden', !visible);
      if (visible) shown++;
    });
    sections.forEach(function (s) {
      var any = s.querySelector('.entry:not(.hidden)');
      s.classList.toggle('hidden', !any);
    });
    navLinks.forEach(function (a) {
      var target = a.getAttribute('href').slice(1);
      var el = document.getElementById(target);
      a.classList.toggle('hidden', !!(el && el.classList.contains('hidden')));
    });
    count.textContent = shown + (shown === 1 ? ' entry' : ' entries');
    var none = document.getElementById('noresults');
    none.classList.toggle('hidden', shown !== 0);
  }

  input.addEventListener('input', apply);
  privToggle.addEventListener('click', function () {
    showPrivate = !showPrivate;
    privToggle.textContent = showPrivate ? 'Hide private helpers'
                                         : 'Show private helpers';
    apply();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    } else if (e.key === 'Escape' && document.activeElement === input) {
      input.value = '';
      apply();
      input.blur();
    }
  });
  apply();
})();
"""


def build(source: str, out_path: str, fragment: bool = False) -> str:
    module_doc, sections, exported = collect(source)

    global KNOWN
    KNOWN = set()
    for section in sections:
        for entry in section.entries:
            KNOWN.add(entry.name)
            for m in entry.members:
                KNOWN.add(m.name)

    version = "unknown"
    m = re.search(r'__version__ = "([^"]+)"', io.open(source, encoding="utf-8").read())
    if m:
        version = m.group(1)

    # The first paragraph of the module docstring is the lede.
    doc_lines = module_doc.split("\n")
    lede = ""
    for k, line in enumerate(doc_lines):
        if line.startswith("``docpipe``"):
            block = []
            while k < len(doc_lines) and doc_lines[k].strip():
                block.append(doc_lines[k].strip())
                k += 1
            lede = " ".join(block)
            break
    # Everything after the lede paragraph becomes the overview body.
    overview_body = module_doc
    if lede:
        overview_body = module_doc.split(lede.split(".")[0], 1)[-1]
        idx = module_doc.find("Design in one paragraph")
        overview_body = module_doc[idx:] if idx > 0 else module_doc

    nav = []
    for section in sections:
        items = [e for e in section.entries]
        if not items:
            continue
        nav.append("<details open><summary><span class='num'>%s</span>%s</summary><ul>"
                   % (section.number, _inline(section.title)))
        for e in items:
            nav.append("<li><a href='#%s'%s>%s</a></li>"
                       % (e.anchor, " class='hit'" if e.public else "",
                          html.escape(e.name)))
        nav.append("</ul></details>")

    body = []
    total = 0
    for section in sections:
        if not section.entries:
            continue
        body.append("<section class='section' id='%s'>" % section.anchor)
        body.append("<h2><span class='num'>%s.</span> %s</h2>"
                    % (section.number, _inline(section.title)))
        if section.intro:
            body.append("<div class='section-intro'>%s</div>"
                        % render_doc(section.intro, heading_level=4))
        for entry in section.entries:
            body.append(render_entry(entry))
            total += 1
        body.append("</section>")

    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>docpipe %(version)s &middot; API reference</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="API reference for docpipe, a single-file document intelligence substrate.">
<style>%(css)s</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <h1>docpipe</h1>
      <div class="ver">v%(version)s &middot; API reference</div>
    </div>
    <div class="searchbox">
      <input id="q" type="search" placeholder="Filter &nbsp;/" autocomplete="off"
             spellcheck="false" aria-label="Filter entries">
    </div>
    <nav class="nav">%(nav)s</nav>
  </aside>
  <main>
    <div class="overview">
      <h1>docpipe</h1>
      <p class="lede">%(lede)s</p>
      <div class="doc">%(overview)s</div>
    </div>
    <div class="toolbar">
      <button id="toggle-private" type="button">Show private helpers</button>
      <a class="src" href="https://github.com/%(repo)s">github.com/%(repo)s</a>
      <span class="count" id="count">%(total)d entries</span>
    </div>
    <div id="noresults" class="noresults hidden">No entry matches that filter.</div>
    %(body)s
  </main>
</div>
<script>%(js)s</script>
</body>
</html>
""" % {
        "version": html.escape(version),
        "css": CSS,
        "js": JS,
        "nav": "\n".join(nav),
        "lede": _inline(lede),
        "overview": render_doc(overview_body, heading_level=3),
        "body": "\n".join(body),
        "total": total,
        "repo": html.escape(REPO),
    }

    if fragment:
        page = re.sub(r"^.*?<body>\n", "", page, count=1, flags=re.S)
        page = page.replace("</body>\n</html>\n", "")
        page = ('<title>docpipe %s &middot; API reference</title>\n<style>%s</style>\n'
                % (html.escape(version), CSS)) + page

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-o", "--out", default=os.path.join(ROOT, "site", "index.html"),
                        help="output HTML path (default: site/index.html)")
    parser.add_argument("--source", default=SOURCE, help="path to docpipe.py")
    parser.add_argument("--fragment", action="store_true",
                        help="omit the <html>/<head>/<body> wrapper")
    args = parser.parse_args(argv)
    path = build(args.source, args.out, fragment=args.fragment)
    size = os.path.getsize(path) / 1024.0
    print("wrote %s (%.0f KB)" % (path, size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
