# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`docpipe` is the domain-independent 70% of a scanned-document extraction pipeline:
ingest → quality measurement → adaptive preprocessing → pluggable OCR/VLM reading →
schema-driven extraction with provenance → calibrated confidence → evaluation harness.

`docpipe.py` is the entire library. `README.md` is the user-facing doc, and its *Scope
boundary* and *Honest limitations* sections are the authority when a design question comes
up — the module docstring at the top of `docpipe.py` carries the same reasoning in short form.

## Layout

| Path | What |
|---|---|
| `docpipe.py` | The whole library — ~9,400 lines, 16 numbered sections |
| `tests/` | pytest suite, one file per section group |
| `docs/build_reference.py` | Generates the API reference by `ast`-parsing `docpipe.py` |
| `examples/hospital_bill.py` | End-to-end consumer example |
| `.github/workflows/` | `ci.yml` (tests + compat gates), `docs.yml` (Pages) |

`docpipe.py` is organised into 16 banner-delimited sections (`# === N. Title ===`), matching
the layer numbering in the module docstring. New code goes in the section it belongs to —
find the banner first, don't append to the end of the file.

Within a section, groups of free functions that share a subject live on a **namespace
class** as staticmethods — `Caps`, `Util`, `Image`, `Quality`, `Ingest`, `Ops`, `Policies`,
`Pricing`, `Text`, `Confidence`, `Validators`. See *The namespace classes* below before
adding a function to one.

## Non-negotiable constraints

These are load-bearing promises, each enforced by CI. Breaking one is a defect, not a
tradeoff.

**1. It stays one file.** Never split `docpipe.py` into a package. The premise is that a
project can vendor a single file with no build step. `pyproject.toml` declares
`py-modules = ["docpipe"]`.

**2. Python 3.8 floor.** No `X | Y` annotations, no `match`, no walrus cleverness. Use
`Optional[...]`, `Union[...]`, `Dict`, `List` from `typing`. CI runs
`ast.parse(..., feature_version=(3, 8))` over the file.

`from __future__ import annotations` is in force, so annotations are never evaluated at
runtime — forward references need no quotes, and annotating with an optional dependency's
type costs nothing.

**3. The core imports with zero optional dependencies.** `dependencies = []` is deliberate.
Nothing heavy may be imported at module import time. Every optional import goes through
`Caps.require(name, purpose)` (raises `MissingDependency` with a pip hint) or
`Caps.have(name)` at the point of use. CI imports `docpipe` with `numpy`, `cv2`, `PIL`, `fitz` and `pydantic` all
blocked and asserts core functions still run.

A missing dependency must fail where the user called it, not three frames deep in a backend.

**4. OpenCV is optional.** Image ops need a pure-NumPy fallback. `DOCPIPE_DISABLE_OPENCV=1`
(or `Caps.set_opencv_enabled(False)` / `with Caps.without_opencv():`) forces the fallback
path, and CI
re-runs the image tests that way. The `both_image_backends` fixture in `tests/conftest.py` is
parametrized over both paths — use it for anything touching pixels.

**5. Every public name is in `__all__` (section 16) and appears in the reference.** The docs
workflow fails if an `__all__` entry has no entry on the generated page. Adding a public
function means: implement it as a staticmethod on the right namespace class → add its
module-level alias next to the others → docstring it. `__all__` gains an entry only when
you add a *new namespace or top-level object*, not for each function.

**6. Everything has a docstring.** 516/516 functions and 82/82 classes, and it stays that
way. `docs/build_reference.py` reads the source with `ast` and never imports it, so it
depends on the source's conventions:

- reST roles (`` :func:`x` ``, `` :class:`X` ``) become cross-links
- `:param name:` blocks become definition lists
- `#:` comments above a constant or dataclass field document it
- prose under a section banner becomes that section's introduction

Docstrings say what is *not* obvious from the code — why a threshold exists, what returning
`None` means, which failures are deliberately swallowed so one bad page can't cost you the
document.

**7. No test touches a network or a paid API.** `EchoClient` is the deterministic LLM double
and ships in the library because consumers need it too. `TruthBackend` (conftest) makes a
controlled number of mistakes so accuracy assertions are exact, not plausible.

## Scope boundary

The library owns everything that does not depend on what a document *means*. Domain schemas,
business rules and downstream integrations belong to consuming projects — **no `HospitalBill`
in core, ever**.

The test for any proposed addition: *could a team working on a completely unrelated document
type use this?* If no, it belongs in that project. See README.md's *Scope boundary*.

Escape hatches are mandatory: every layer usable standalone, and anyone can register a custom
op, backend or router without editing this file. If a consumer must fork to get their job
done, that's a bug in the library.

## Design invariants worth knowing before you edit

- **Preprocessing is channel equalisation, not cosmetics.** Ops correct *measured*
  degradation. A policy that applies a fixed sequence to every page is wrong — it destroys
  signal on clean pages. Note that `default_policy` deliberately omits binarisation:
  it helps classical OCR and hurts VLMs, which use greyscale gradient.
- **Confidence fuses independent signals** (page quality, backend confidence, cross-read
  agreement, logprob, validation, format match). Never surface a model's self-reported
  confidence as *the* confidence — it measures fluency, not correctness.
- **Provenance is recoverable at read time and unrecoverable afterwards.** Thread `BBox`
  through everything. Never fabricate coordinates for a VLM read — mark spans
  `approximate_bbox` and recover geometry by matching against a second read.
- **`raster()` is lazy by contract**, not as an optimisation. A 300-page bundle at 400 DPI
  eagerly rendered is tens of gigabytes.
- **`page.history` records every op applied.** That's what makes an eval result attributable
  to a change. Ops must append to it.
- **`PRICING` ships empty on purpose.** Token counts are always tracked; money reads `0.00`
  and warns until the consumer calls `Pricing.set_pricing`. A wrong price is worse than a
  missing one.

## The namespace classes

Eleven classes group free functions by subject so the export surface is 113 names instead of
221. Each is a pure namespace — no state, no `__init__`, nothing to instantiate.

```python
class Image(object):
    """Raster primitives, each with an OpenCV path and a NumPy fallback."""

    @staticmethod
    def to_gray(img: ImageArray) -> GrayImage:
        """..."""
        arr = ensure_uint8(img)      # the flat alias, not Image.ensure_uint8
        ...


# Module-level aliases.  These are load-bearing, not merely back-compatible: ...
ensure_uint8 = Image.ensure_uint8
to_gray      = Image.to_gray
```

Three things about this that will bite you if you don't know them:

- **The aliases are load-bearing.** Staticmethods call one another through the *bare* names,
  resolved from module globals at call time. Deleting an alias as "back-compat cruft" breaks
  the class from the inside. Every alias block carries a comment saying so.
- **Defaults and decorators run when the class body executes.** A module constant named in a
  default argument (`min_chars: int = MIN_NATIVE_CHARS`) or in a decorator must be defined
  *above* the class. That is why a few constants sit just before their namespace.
- **`@staticmethod` goes outermost**, above `@register_op`.

Only add a class when a group of functions genuinely shares a subject. Stateful classes
(`Page`, `BaseBackend`, `EvalSuite`), the layer entry points (`read`, `preprocess`,
`extract`, `process`) and op machinery (`Op`, `compose`, `register_op`) stay top-level.

## Extension patterns

**A preprocessing op** — a raster function decorated into an `Op` factory, as a staticmethod
on `Ops`:

```python
class Ops(object):
    ...
    @staticmethod
    @register_op("my_op", geometric=False, needs_raster=True)
    def my_op(img: ImageArray, page: Page, strength: float = 1.0) -> Optional[ImageArray]:
        """One line on what distortion this corrects, then why the threshold is what it is."""
        ...
        return out   # or None, meaning "nothing to correct" — the page is left untouched


my_op = Ops.my_op    # with the other aliases below the class
```

`geometric=True` means the op moves pixels, so existing span coordinates must be remapped
(see `deskew` for the affine mapping pattern). `__all__` does not change — `Ops` is already
exported. Consider whether `default_policy` should reach for it.

Registering an op from *outside* this file needs no class at all: `register_op` populates
`OP_FACTORIES` from anywhere.

**A reading backend** — subclass `BaseBackend`, implement `_read` (not `read`; the base
handles timing, span sourcing and script detection) and `is_available`, then register a
*factory*, never an instance:

```python
registry.register("myocr", lambda: MyOCRBackend())
```

Lazy construction matters: instantiating PaddleOCR or Surya loads hundreds of megabytes of
weights, and a document that never routes there must not pay for it.

## Commands

```bash
pip install -e ".[dev]"

pytest -q                                    # ~660 tests, ~75s
DOCPIPE_DISABLE_OPENCV=1 pytest -q tests/test_image_ops.py tests/test_quality.py tests/test_preprocess.py
python docs/build_reference.py --out site/index.html

docpipe caps                                 # what is importable right now
docpipe quality scan.pdf --show-policy       # why is this extracting badly?
docpipe eval datasets/bills --schema s.json --baseline reports/prev.json   # non-zero on regression
```

`/check` runs the whole CI gate locally; `/docs` rebuilds and verifies the reference.

## Releasing

The PyPI distribution is **`docpipe-core`** — the plain name `docpipe` belongs to an
unrelated project. The import name is unaffected: one module, `docpipe.py`, so
`pip install docpipe-core` still gives `import docpipe`. Don't "fix" the mismatch.

The version is stated twice — `pyproject.toml` and `__version__` in `docpipe.py` — because a
vendored copy has no `pyproject.toml` to read it from. CI fails if the two drift, and
`publish.yml` fails if the tag disagrees with either.

```bash
# bump both, then:
git tag v0.2.0 && git push origin v0.2.0
```

The tag triggers `publish.yml`: it re-runs the full `ci.yml` gate on that commit, checks
tag ↔ version agreement, builds, `twine check --strict`s, installs the wheel into a clean
venv *outside the checkout* (inside it, `sys.path` picks up `./docpipe.py` and the check
proves nothing), and uploads via PyPI Trusted Publishing — no API token in secrets.

`MANIFEST.in` puts `tests/` **including `conftest.py`** in the sdist. Every fixture lives in
that file; shipping the tests without it gives downstream packagers a suite that cannot
collect.

## Style

- Code comments and docstrings use British spelling (`normalise`, `rasterise`) — the prose,
  not the identifiers. Public API names are US-spelled (`normalize_text`, `binarize`).
- `%`-formatting throughout. There is not one f-string in the repository — not in
  `docpipe.py`, the tests, the docs generator or the example. Match that.
- Commit subjects are imperative and specific; the body explains *why*, not what the diff
  already shows.
