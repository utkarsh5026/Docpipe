---
description: Add a reading backend (OCR engine or vision model) to docpipe.py
argument-hint: "<engine name, e.g. 'Surya' or 'Azure Document Intelligence'>"
allowed-tools: Bash, Read, Edit, Grep, Glob
---

Add a reading backend for: **$ARGUMENTS**

Read `TesseractOCR` (local engine, real bounding boxes) and `AnthropicVisionBackend` (remote,
token-costed, no geometry) in `docpipe.py` section 8 first. One of those two shapes is
almost certainly yours.

## 1. Declare the dependency correctly

The engine's SDK is **optional**. Add it to `[project.optional-dependencies]` in
`pyproject.toml` as its own extra, and to `_PIP_NAMES` in section 1 if the import name
differs from the distribution name. Never import it at module scope — only through
`require("modname", "purpose")` inside the method that needs it.

## 2. Implement in section 8

```python
class MyEngineBackend(BaseBackend):
    """One line on what this engine is good at, and what it costs."""

    name = "myengine"
    needs_raster = True
    preferred_dpi = DEFAULT_DPI

    def is_available(self) -> bool:
        """Whether the SDK (and any model weights or binary) is importable right now."""
        return have("myengine")

    def _read(self, page: Page) -> ReadResult:
        """Produce spans.  Timing, span sourcing and script detection are the base's job."""
```

Requirements:

- **Implement `_read`, not `read`.** `BaseBackend.read` handles timing, tagging spans with
  the backend name, and filling in `script`. Overriding it duplicates that work and loses it.
- **Convert coordinates with `self._bbox(page, x0, y0, x1, y1)`** — engines return pixels,
  the IR stores canonical page points. Do not hand-roll the conversion.
- **Never fabricate bounding boxes.** If the engine returns text with no geometry (any VLM),
  mark spans `approximate_bbox` rather than inventing coordinates. Invented boxes point a
  reviewer at the wrong part of the page, which is worse than admitting you have none.
- **Set `confidence` to what the engine actually reports**, and `None` when it reports
  nothing. A made-up 0.99 corrupts confidence fusion downstream.
- **Override `estimate_cost`** for anything paid, returning token counts. Do **not** add
  prices to `PRICING` — it ships empty on purpose, and consumers register their own via
  `Pricing.set_pricing`.
- **Swallow per-page failures deliberately or not at all.** If you catch, say in the
  docstring why one bad page must not cost the whole document, and record it on the page.
- Python 3.8, `%`-formatting, full docstrings on every method.

## 3. Register a factory, never an instance

```python
registry.register("myengine", lambda: MyEngineBackend())
```

Instantiating a heavyweight engine loads hundreds of megabytes of weights. A document that
never routes there must not pay for it — that is the entire reason the registry stores
callables.

## 4. Routing

Decide whether `default_router` should ever choose this backend, and under what measured
condition (`page.kind`, `page.quality.verdict`, layout). Native-text pages must keep going to
`pymupdf` — free, exact, and a model reading a picture of a text layer can hallucinate.
If the backend is opt-in only, say so in its docstring.

## 5. Wire up and test

- Add the class name to `__all__` in section 8's group.
- Add the extra to the `all` extra in `pyproject.toml` **only if** it installs cleanly with
  no GPU and no system package.
- Test in `tests/test_backends.py`: availability reporting when the SDK is absent, span
  geometry against the conftest ground-truth boxes, cost estimation, and graceful failure.
  No network calls — mock the client the way the existing vision backend tests do.

```bash
pytest -q tests/test_backends.py
python -c "import docpipe as dp; print(dp.registry.names()); print(dp.registry.available())"
```

Report: what the engine is good at, when the router picks it, what it costs, and what the
tests assert.
