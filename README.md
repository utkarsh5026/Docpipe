# docpipe

**A single-file document intelligence substrate.**

`docpipe` is the domain-independent 70% of every scanned-document extraction
pipeline: ingest, page-quality measurement, adaptive preprocessing, pluggable
OCR/VLM reading backends, schema-driven extraction with provenance, calibrated
confidence fusion, and an evaluation harness.

Everything lives in **one file** on purpose. Copy `docpipe.py` into a project,
or `pip install docpipe-core` — both work, and neither requires a build step.

```bash
pip install "docpipe-core[core,schema]"   # or just: cp docpipe.py yourproject/
```

The distribution is named `docpipe-core` because `docpipe` on PyPI is an
unrelated project. The import name is unchanged: `import docpipe`.

---

## Why

Several projects independently solve the same first 70% of the pipeline: DPI
selection, native-text-layer detection, deskewing, blur handling, OCR backend
choice, page classification, retry and cost accounting, per-field confidence,
provenance. Each of those is a small decision; collectively they are weeks of
work and the source of most extraction defects. Duplicating them is expensive.
Duplicating them *inconsistently* is worse, because two projects then give
different answers for the same page and nobody can explain why.

The point is not saved headcount. It is that accuracy improvements start
compounding across projects instead of being rediscovered one project at a
time. When someone finds that a particular binarisation lifts recall by four
points on faded thermal-printer bills, that ships to every consumer in a
version bump instead of living in one person's notebook.

## The idea in one paragraph

A scanned page is *a fact that survived a noisy channel*: printed,
photocopied, faxed, scanned at whatever DPI the clerk's flatbed defaulted to,
JPEG'd, and wrapped in a PDF. Extraction is channel inversion. Three things
follow, and they shape the whole library:

- **Preprocessing is equalisation, not cosmetics.** The right correction for a
  page is a function of that page's *measured* degradation. A fixed pipeline
  over-processes clean pages (destroying signal that was never damaged) and
  under-processes bad ones.
- **Confidence is a property of the channel, not of the model's mood.** Asking
  a language model how confident it is measures how confident the text
  *sounds*. A real estimate fuses independent evidence and is then calibrated
  against labelled data.
- **Provenance is recoverable at read time and unrecoverable afterwards.** Every
  fact came from a rectangle on a page. Drop that link and you can never show a
  reviewer or an auditor why you believe something.

---

## Quickstart

```python
import docpipe as dp

doc = dp.Ingest.ingest("claim_47812.pdf")            # PDF, image, TIFF, or .eml
doc = dp.preprocess(doc, policy=dp.Policies.default_policy)
doc = dp.read(doc, router=dp.default_router)

print(doc.text())
print(doc.pages[0].quality)
# PageQuality(verdict=degraded, blur=0.41, skew=2.30 deg, contrast=0.52, dpi~210, score=0.61)
```

With a schema:

```python
from pydantic import BaseModel
from datetime import date
from decimal import Decimal

class LineItem(BaseModel):
    description: str
    amount: Decimal

class HospitalBill(BaseModel):
    hospital_name: str
    patient_name: str
    admission_date: date
    discharge_date: date | None = None
    line_items: list[LineItem] = []
    total: Decimal

result = dp.extract(
    doc,
    schema=HospitalBill,
    context="Indian private hospital bill. Amounts in INR. "
            "Dates may be DD-MM-YYYY or DD/MM/YY.",
    client=dp.AnthropicClient(model="claude-sonnet-5"),
    validators=[
        dp.Validators.line_items_sum_to_total(),
        dp.Validators.date_order("admission_date", "discharge_date"),
    ],
)

result.fields["total"].value        # Decimal("118250.00")
result.fields["total"].confidence   # 0.94  -- fused, calibratable
result.fields["total"].evidence     # [BBox(page=3, 402.0, 688.0, 471.0, 699.0)]
result.fields["total"].signals      # the components behind that number
result.low_confidence(0.8)          # what a human should look at
```

That is the entire domain-side integration surface. The project owns
`HospitalBill`, the two validators, and the context string. Everything else is
library.

---

## Architecture

Six layers. Each is independently useful; use one and ignore the rest.

| Layer | What it owns |
|---|---|
| **5 — Evaluation** | `EvalSuite`, metrics, pipeline A/B, regression gates, calibration |
| **4 — Extraction** | schema in, typed result with per-field evidence and confidence out |
| **3 — Reading** | pluggable, cost-aware OCR / VLM backends and routing |
| **2 — Preprocessing** | composable `Page -> Page` ops, driven by measured quality |
| **1 — IR** | `Document`, `Page`, `TextSpan`, `BBox`, `PageQuality` |
| **0 — Ingest** | PDF, TIFF, image, email attachment; format detection and repair |

Cross-cutting: caching · cost accounting · retries · tracing · provenance.

### Layer 1 — the intermediate representation *is* the product

Shared libraries fail when they expose *procedures* — a `process_document()`
that does eleven things and returns a dict. The moment a consumer needs the
eighth thing done differently they copy the source and the library is dead.

They succeed when they expose a *data type* everyone agrees on, plus small
independent functions over it. Because every backend consumes and produces
`TextSpan`, backends are swappable. Because ops are `Page -> Page`, they
compose freely. Because two pipelines share input and output types, you can A/B
them.

Two details that matter more than they look:

- **`page.history`** records every op applied, with parameters and timings.
  That is what makes an experiment reproducible and lets the eval harness
  attribute an accuracy change to a specific op.
- **`page.raster()` is lazy by contract.** A 300-page legal bundle rendered
  eagerly at 400 DPI is tens of gigabytes of RAM. Laziness has to live in the
  type; it cannot be retrofitted by a caller.

### Layer 2 — preprocessing is adaptive, and does *not* binarise by default

```python
def default_policy(page):
    ops = []
    if page.quality.illumination < 0.55:  ops.append(dp.Ops.normalize_illumination())
    if page.quality.noise > 0.35:         ops.append(dp.Ops.denoise("light"))
    if page.quality.effective_dpi < 300:  ops.append(dp.Ops.ensure_dpi(300))
    if abs(page.quality.skew_deg) > 0.4:  ops.append(dp.Ops.deskew())
    if page.quality.blur < 0.35:          ops.append(dp.Ops.unsharp(amount=1.2))
    if page.quality.contrast < 0.30:      ops.append(dp.Ops.autocontrast())
    return ops
```

Note what is deliberately absent: unconditional binarisation. It reliably helps
classical OCR and reliably *hurts* vision-language models, which use greyscale
gradient to disambiguate faint strokes that a threshold has already destroyed.
That trade-off is encoded once here — `ocr_policy` binarises, `vlm_policy` does
not — instead of three times in three codebases.

**Ops available:** `to_grayscale` `invert_if_dark` `autocontrast` `gamma`
`clahe` `normalize_illumination` `remove_shadow` `rescale` `ensure_dpi`
`resize_max_side` `deskew` `rotate` `auto_orient` `crop_to_content`
`remove_border` `pad` `perspective_correct` `denoise` `despeckle` `unsharp`
`morphology` `binarize` `remove_lines` `remove_stamps` — plus
`split_multi_bill_page` for two bills photocopied onto one A4.

**Binarisation methods:** `otsu` `adaptive` `sauvola` (default) `niblack`
`wolf` `nick` `bradley`. All are integral-image based, so cost is independent
of window size.

Ops are data — `dp.op_from_dict(op.to_dict())` round-trips, so a policy can be
recorded in an eval report and reproduced exactly.

### Layer 3 — backends and routing

Shipped: `PyMuPDFTextLayer` · `TesseractOCR` · `PaddleOCRBackend` ·
`RapidOCRBackend` · `EasyOCRBackend` · `DocTRBackend` · `SuryaBackend` ·
`AnthropicVisionBackend` · `OpenAIVisionBackend` · `GeminiVisionBackend`.

Wrappers: `CachingBackend` (keyed on raster content), `RetryingBackend`
(accounts for attempts the provider already billed), `EnsembleBackend` (two
reads, for cross-read agreement).

```python
def default_router(page):
    if page.kind is dp.PageKind.DIGITAL_NATIVE:
        return "pymupdf"        # free and exact -- never pay a model for this
    if page.quality.verdict is dp.Verdict.UNREADABLE or page.layout.has_handwriting:
        return "vlm:claude"     # degraded, handwritten, multi-script
    return "paddle"             # clean print, especially ruled tables
```

The cost delta between "send every page to a vision model" and "send only the
pages that need one" is large. The accuracy delta on native-text pages runs the
*other* way: a text layer read directly is exact, while a model reading a
picture of it can hallucinate. Both point the same way.

Registering your own takes no change to this file:

```python
dp.registry.register("acme-ocr", MyBackend)
```

### Layer 4 — confidence, done properly

A model's self-reported confidence is not a probability of correctness. It
correlates with fluency. A clean, confidently-formatted hallucination scores
high; a correct reading of a smudged digit scores low. Shipping that number to
a claims reviewer as "confidence" is worse than shipping no number, because it
will be trusted.

`docpipe` fuses signals chosen to be as uncorrelated as possible, in log-odds
space rather than as a weighted average:

| Signal | What it measures |
|---|---|
| `cross_read_agree` | two independent engines produced the same string for the same region |
| `validation` | did the value survive arithmetic and cross-field checks |
| `text_support` | was the value found in the text some engine actually read |
| `backend_conf` | the OCR engine's own confidence over that region |
| `page_quality` | measured degradation, localised to the evidence rectangle |
| `format_match` | does the value match the expected shape for its type |

Log-odds matters: averaging lets three confident signals bury a failed
validator, whereas in log-odds a single strong disagreement pulls the result
down hard. Signals set to `None` are *absent*, not zero — encoding "the engine
declined to guess" as 0.0 would punish every backend that is honest.

**Confidence without a calibration set is decoration**, which is why Layer 5
exists and closes the loop:

```python
report = suite.run({"candidate": pipeline})
calibrator = report.fit_calibrator(kind="platt")   # or "isotonic"
result = dp.extract(doc, Schema, client=client, calibrator=calibrator)
```

### Layer 5 — the evaluation harness

If only one component of this library were built, it should be this one.
Without it, "did that change help?" is answered by eyeballing a handful of
documents. That is not a measurement.

```python
suite = dp.EvalSuite.from_dir("datasets/hospital_bills_v3")   # docs + <stem>.json

report = suite.run({"baseline": pipeline_a, "candidate": pipeline_b})
print(report.summary())

report.by_field()                 # which fields degraded, not just the average
report.by_page_quality()          # did we only improve on the clean pages?
report.confidence_curve()         # is a 0.9 right 90% of the time?
report.worst_cases()              # where to look first
report.compare("baseline", "candidate")
report.regression_vs("reports/release-2026-08.json", tolerance=0.01)
```

`by_page_quality()` is the one to watch. A change that lifts the average by
improving pages that were already fine, while leaving the degraded ones
untouched, has not solved the problem it was meant to solve.

---

## CLI

```bash
docpipe caps                                   # what is installed and usable
docpipe info      scan.pdf                     # pages, kinds, metadata
docpipe quality   scan.pdf --show-policy       # why is this extracting badly?
docpipe preprocess scan.pdf --policy ocr --out ./debug
docpipe read      scan.pdf --backend tesseract
docpipe extract   bill.pdf --schema schema.json --client anthropic
docpipe eval      datasets/bills --schema schema.json --report run.json \
                  --baseline reports/release-2026-08.json
```

`docpipe eval --baseline` exits non-zero on a regression, so it drops straight
into CI.

---

## Dependencies

The core is pure standard library. Everything heavier is optional and imported
lazily; `dp.Caps.capabilities()` reports what is actually usable right now.

| Extra | Enables |
|---|---|
| `numpy` | all raster work |
| `opencv-python` | fast image ops — **NumPy fallbacks exist for most**, and the test suite runs both paths |
| `pymupdf` | PDF ingest and native text layers |
| `pillow` | image and multi-page TIFF ingest |
| `pytesseract` / `tesseract` | Tesseract backend (the binary alone is enough) |
| `paddleocr`, `rapidocr-onnxruntime`, `easyocr`, `python-doctr`, `surya-ocr` | OCR backends |
| `anthropic`, `openai` | vision backends and extraction clients |
| `pydantic` | schema-driven extraction (v1 and v2 both work) |

**Python 3.8+.** No `X | Y` annotations, no `match`, no walrus — this file is
meant to be dropped into old codebases.

### Cost accounting starts honest

`PRICING` ships **empty**. Token counts are always tracked; the monetary amount
reads `0.00` and warns once until you register what your account actually pays:

```python
dp.Pricing.set_pricing("claude-sonnet-5", input_per_mtok=3.00, output_per_mtok=15.00)
```

A wrong price is worse than a missing one — it produces a plausible budget
report that nobody rechecks.

---

## Testing

```bash
pip install "docpipe-core[dev]"
pytest                      # ~660 tests, about 75 seconds
```

The suite builds synthetic pages with Pillow and **records the true bounding
box of every word while drawing**, which gives exact ground truth. It then
pushes those pages through known degradation (Gaussian blur of known sigma,
known skew angle, known shadow gradient, known downsample) and asserts the
measured value recovers the known one — skew to within 0.3°, for instance.
A `TruthBackend` makes a *controlled* number of mistakes, so assertions like
"field accuracy is exactly 2/3" are meaningful rather than plausible.

Image ops are tested twice, once through OpenCV and once through the pure-NumPy
fallbacks, so the no-OpenCV promise is real.

No test touches a network or a paid API: `dp.EchoClient` is a deterministic
test double, and it ships in the library because every consuming project needs
the same thing to test its own schemas and validators.

---

## API reference

**[utkarsh5026.github.io/Docpipe](https://utkarsh5026.github.io/Docpipe/)** —
every class, function and constant, grouped by the same numbered layers the
source uses, with a filter box and a source link on each entry.

The page is generated from `docpipe.py` itself and rebuilt on every push to
`main`. To build it locally:

```bash
python docs/build_reference.py        # -> site/index.html
```

The generator reads the source with `ast` rather than importing it, so it needs
no dependencies — not even docpipe's optional ones — and the reference cannot
drift from the code it documents. CI fails if a name in `__all__` is missing
from the page.

### Namespaces

One file does not have to mean one flat heap of names. Functions that share a
subject are grouped onto a namespace class, and only the class is exported:

| | |
|---|---|
| `Caps` | what is importable right now; the OpenCV kill switch |
| `Util` | hashing, clamping, string distance, JSON coercion |
| `Image` | raster primitives — arrays in, arrays out |
| `Quality` | page-quality measures and estimators |
| `Ingest` | bytes of unknown provenance → `Document` |
| `Ops` | the preprocessing ops, as `Op` factories |
| `Policies` | measured page → the ops it needs |
| `Pricing` | token accounting and the optional price table |
| `Text` | script detection, normalisation, value parsing |
| `Confidence` | signal fusion and calibration metrics |
| `Validators` | domain-independent validator factories |

Stateful classes (`Page`, `BaseBackend`, `EvalSuite`) and the layer entry points
(`read`, `preprocess`, `extract`, `process`) stay top-level.

Every one of those staticmethods is *also* a module attribute under its bare
name — `dp.to_gray` and `dp.Image.to_gray` are the same object, so nothing
written against an earlier version breaks:

```python
dp.Image.to_gray(img)   # the documented path
dp.to_gray(img)         # identical, and still supported
```

What changed is `__all__`, which went from 221 names to 113. `from docpipe import *`
now gives you eleven namespaces instead of two hundred loose functions.

---

## Scope boundary

**In scope (library owns):** ingest, format detection, PDF repair,
rasterisation; quality measurement and adaptive preprocessing; layout, table
and stamp detection; OCR/VLM adapters, routing, retry, cost accounting; script,
numeral, date and currency normalisation; schema-driven extraction, prompt
assembly, structured-output parsing; confidence fusion, calibration,
provenance; caching, tracing, the eval harness.

**Explicitly out of scope (projects own):** domain schemas — no `HospitalBill`
in core, ever; business validation rules, tariff logic, policy checks;
downstream system integration; workflow, queues, human-in-the-loop review UI.

The test for a proposed addition: *could a team working on a completely
unrelated document type use this?* If no, it belongs in that project.

**Escape hatches are mandatory.** Every layer is usable standalone; anyone can
register a custom op, backend or router without touching this file. If a
consumer ever has to fork the library to get their job done, that is a defect
in the library.

---

## Honest limitations

- **`PRICING` is empty by design.** Cost reads 0.00 until you register prices.
- **VLM provenance is approximate.** A vision model returns a wall of text with
  no coordinates. Rather than fabricate bounding boxes, spans are marked
  `approximate_bbox` and tighter boxes are recovered by matching the value
  against a second, geometry-bearing read. Invented coordinates would be worse
  than none — they point a reviewer at the wrong part of the page.
- **180° rotation needs Tesseract's OSD.** The projection-profile fallback
  distinguishes portrait from landscape but cannot detect an upside-down page.
  This is reported in `page.meta["orientation_method"]` rather than hidden.
- **The default thresholds and fusion weights are priors, not constants.** They
  come from scanned Indian hospital bills and court filings. Recalibrate them
  per document class against a labelled set — that is what Layer 5 is for.
- **`estimate_effective_dpi` prefers line pitch** (blur-robust) and falls back
  to stroke width, which inflates under blur. Pages with no regular text lines
  get the weaker estimate.

## License

MIT.
