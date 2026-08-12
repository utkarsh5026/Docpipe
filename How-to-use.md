# How to use docpipe

A practical, plain-language guide. `README.md` explains *why* the library is shaped
the way it is; this file explains *what to type* when you have a document and a
problem.

The format below is deliberate. Each section starts with **the problem** in the
words you would actually use to describe it, then gives **the solution** and the
code. Skip to whichever problem is yours — the sections do not depend on each
other.

---

## Table of contents

- [First, the one-minute version](#first-the-one-minute-version)
- [Installing](#installing)
- [The mental model](#the-mental-model)
- [Use cases](#use-cases)
  1. [I just want the text out of a PDF](#1-i-just-want-the-text-out-of-a-pdf)
  2. [My PDF already has real text — don&#39;t pay to OCR it](#2-my-pdf-already-has-real-text--dont-pay-to-ocr-it)
  3. [My scans are terrible and the OCR output is garbage](#3-my-scans-are-terrible-and-the-ocr-output-is-garbage)
  4. [I need fields, not a wall of text](#4-i-need-fields-not-a-wall-of-text)
  5. [Which of these values can I actually trust?](#5-which-of-these-values-can-i-actually-trust)
  6. [An auditor asked me where this number came from](#6-an-auditor-asked-me-where-this-number-came-from)
  7. [This is costing too much](#7-this-is-costing-too-much)
  8. [Did my change actually make things better?](#8-did-my-change-actually-make-things-better)
  9. [I want to test my code without spending money](#9-i-want-to-test-my-code-without-spending-money)
  10. [My OCR engine isn&#39;t in the list](#10-my-ocr-engine-isnt-in-the-list)
  11. [I need an image correction that doesn&#39;t exist](#11-i-need-an-image-correction-that-doesnt-exist)
  12. [I can&#39;t use an LLM at all](#12-i-cant-use-an-llm-at-all)
  13. [The dates and amounts come out wrong for my country](#13-the-dates-and-amounts-come-out-wrong-for-my-country)
  14. [Two documents were photocopied onto one page](#14-two-documents-were-photocopied-onto-one-page)
  15. [I have 5,000 files, not one](#15-i-have-5000-files-not-one)
  16. [I can&#39;t install OpenCV here](#16-i-cant-install-opencv-here)
  17. [I want this inside a web service](#17-i-want-this-inside-a-web-service)
- [The CLI cookbook](#the-cli-cookbook)
- [Picking only the parts you need](#picking-only-the-parts-you-need)
- [Things that surprise people](#things-that-surprise-people)
- [Where to look next](#where-to-look-next)

---

## First, the one-minute version

**The problem.** You have scanned documents — bills, claims, invoices, court
filings — and you need reliable data out of them. Every project that tries this
rebuilds the same machinery: pick a DPI, notice the page is skewed, choose an OCR
engine, discover half the pages already had text, retry the failures, guess at
confidence, and then have no way to tell whether last week's change helped.

**The solution.** `docpipe` is that machinery, in one file, with nothing about
*your* documents baked in. You bring the schema and the business rules; it brings
everything from "here are some bytes" to "here is a typed value, with a confidence
number and the rectangle on the page it came from".

```python
import docpipe as dp

doc = dp.Ingest.ingest("scan.pdf")                       # path, bytes or stream
doc = dp.preprocess(doc)                                 # measure, then correct
doc = dp.read(doc)                                       # OCR / text layer / VLM
print(doc.text())
```

That is three lines and no configuration. Everything else in this guide is about
what to change when those three lines are not enough.

---

## Installing

There are two honest ways to get the library, and both are supported forever:

**1. Copy the file.** `docpipe.py` is the entire library. Drop it into your
project and import it. No build step, no package, no dependency resolution.

```bash
cp docpipe.py yourproject/
```

**2. Install it.**

```bash
pip install docpipe-core                    # core only: pure standard library
pip install "docpipe-core[core]"            # numpy, opencv, pymupdf, pillow
pip install "docpipe-core[core,schema]"     # + pydantic, for schema extraction
pip install "docpipe-core[all]"             # everything that installs without a GPU
```

The distribution is `docpipe-core`; the module you import is still `docpipe`.
The plain name `docpipe` on PyPI belongs to an unrelated project — installing
that one will not give you this library.

### What do I actually need installed?

The core imports with **zero** third-party packages. Every heavy thing is
optional and loaded only at the moment you use it — and if it is missing you get
a clear error naming the `pip install` that fixes it, raised where *you* called,
not from deep inside a backend.

| You want to…                                                       | You need                                                                                                               |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Parse the IR, normalise text, fuse confidence, run the eval harness | nothing                                                                                                                |
| Open a PDF                                                          | `pymupdf`                                                                                                            |
| Open an image or multi-page TIFF                                    | `pillow`                                                                                                             |
| Do anything with pixels (quality, preprocessing)                    | `numpy`                                                                                                              |
| Do it*fast*                                                       | `opencv-python-headless` (optional — pure-NumPy fallbacks exist)                                                    |
| OCR                                                                 | one of`pytesseract` · `paddleocr` · `rapidocr-onnxruntime` · `easyocr` · `python-doctr` · `surya-ocr` |
| Read pages with a vision model                                      | `anthropic` or `openai`                                                                                            |
| Extract into a typed model                                          | `pydantic` (v1 or v2 both work)                                                                                      |

Ask the library what it can do right now rather than guessing:

```bash
docpipe caps
```

```python
dp.Caps.capabilities()
# {'numpy': True, 'cv2': True, 'PIL': True, 'pymupdf': True,
#  'pytesseract': False, 'tesseract_binary': True, 'anthropic': True, ...}
```

**Python 3.8 or newer.** The file avoids `X | Y` annotations, `match`, and other
modern syntax on purpose, so it drops into old codebases unchanged.

---

## The mental model

Everything is a short pipeline over one data type. If you understand the diagram
below and the table under it, you understand the library.

```
   file / bytes
        |
        v
  [0] Ingest ........... Document (pages know their size and whether they
        |                          already contain real text; nothing is
        |                          rasterised yet)
        v
  [1] Measure .......... PageQuality per page: blur, skew, contrast, noise,
        |                effective DPI, illumination, a verdict
        v
  [2] Preprocess ....... only the corrections this page's measurements ask for
        |
        v
  [3] Read ............. TextSpans: text + the rectangle it sits in
        |
        v
  [4] Extract .......... typed fields + confidence + evidence rectangles
        |
        v
  [5] Evaluate ......... did any of that actually work? (run this one often)
```

Three types carry everything:

| Type         | What it is                                                                             |
| ------------ | -------------------------------------------------------------------------------------- |
| `Document` | A list of`Page`s plus metadata. `doc.text()` gives you the lot.                    |
| `Page`     | Geometry, a**lazy** raster, `quality`, `layout`, `spans`, and `history`. |
| `TextSpan` | A piece of text, its`BBox`, which engine read it, and that engine's confidence.      |

Two properties of `Page` matter more than they look:

- **`page.raster()` is lazy by contract.** A 300-page bundle at 400 DPI rendered
  eagerly is tens of gigabytes. Ingest never renders; pages render themselves when
  something asks.
- **`page.history` records every operation applied to the page**, with parameters
  and timings. That is what lets you explain later why one run differed from
  another.

---

## Use cases

### 1. I just want the text out of a PDF

**The problem.** Someone hands you a PDF. You do not know whether it is a real
digital PDF, a photograph of paper, or a mixture of both — which it usually is.
You just need the words.

**The solution.** The default path already handles the mixture. Ingest detects
which pages carry a genuine text layer; the default router reads those directly
and sends only the rest to OCR.

```python
import docpipe as dp

doc = dp.Ingest.ingest("scan.pdf")
doc = dp.preprocess(doc)                 # measures quality, corrects what needs it
doc = dp.read(doc)                       # routes each page to the right reader

print(doc.text())                        # the whole document
print(doc.pages[0].text())               # one page
for line in doc.pages[0].lines():        # spans grouped into visual lines
    print(" ".join(s.text for s in line))
```

Or, from the shell, with nothing to write at all:

```bash
docpipe read scan.pdf
```

**What to look at if it goes wrong.** `doc.pages[0].quality` tells you how bad
the page is; `doc.pages[0].kind` tells you what ingest decided the page *was*
(`DIGITAL_NATIVE`, `SCANNED`, `HYBRID`, `BLANK`). Most surprises are one of those
two being different from what you assumed.

---

### 2. My PDF already has real text — don't pay to OCR it

**The problem.** A large fraction of "scanned" PDFs are not scanned at all. They
were generated digitally and carry an exact text layer. Running OCR over them is
slower, costs money, and is *less* accurate — a model reading a picture of text
can hallucinate, while the embedded text layer is exact.

**The solution.** This is what `PageKind` is for, and the default router already
does the right thing. You can also be explicit:

```python
doc = dp.Ingest.ingest("statement.pdf")

native = doc.select(lambda p: p.kind is dp.PageKind.DIGITAL_NATIVE)
scanned = doc.select(lambda p: p.kind is not dp.PageKind.DIGITAL_NATIVE)

print(len(native.pages), "pages free,", len(scanned.pages), "pages need OCR")

doc = dp.read(doc, backend="pymupdf")    # force the text layer everywhere
```

To see what ingest decided, before committing to anything:

```bash
docpipe info statement.pdf
```

**The knob you may need.** A page counts as native when it carries at least
`MIN_NATIVE_CHARS` characters. A page with a text layer holding only a header
("Page 3 of 40") is really a scan; raise the threshold on ingest if your documents
do that:

```python
doc = dp.Ingest.ingest("statement.pdf", min_native_chars=400)
```

---

### 3. My scans are terrible and the OCR output is garbage

**The problem.** Photocopies of faxes of photocopies. Pages skewed by three
degrees because the clerk dropped a stack into the feeder. Shadow across a phone
photo. Faded thermal-printer ink. The OCR output looks like noise and you do not
know which of those problems to fix first.

**The solution.** Measure before you fix. This is the single most useful command
in the library:

```bash
docpipe quality bad_scan.pdf --show-policy
```

It prints, per page, what is actually wrong — and, with `--show-policy`, the exact
list of corrections the adaptive policy would apply and why. From Python:

```python
doc = dp.Ingest.ingest("bad_scan.pdf")
dp.Quality.measure_document(doc)

q = doc.pages[0].quality
print(q)
# PageQuality(verdict=degraded, blur=0.41, skew=2.30 deg, contrast=0.52, dpi~210, score=0.61)

print(dp.Policies.default_policy(doc.pages[0]))
# [Op(deskew), Op(ensure_dpi min_dpi=300), Op(unsharp amount=1.2)]
```

Then let the policy act:

```python
doc = dp.preprocess(doc, policy=dp.Policies.default_policy)
doc = dp.read(doc)
print(doc.pages[0].history_summary())   # exactly what was done, in order
```

**Which policy?** Three ship, and the choice matters:

| Policy                      | Use when                             | Note                                                           |
| --------------------------- | ------------------------------------ | -------------------------------------------------------------- |
| `Policies.default_policy` | you don't know yet                   | safe for both OCR and vision models                            |
| `Policies.ocr_policy`     | reading with Tesseract, Paddle, etc. | **binarises** — classical OCR likes hard black-on-white |
| `Policies.vlm_policy`     | reading with a vision model          | never binarises, caps the image size                           |
| `Policies.no_policy`      | measuring a baseline                 | applies nothing                                                |

That difference is the one design decision worth internalising: **binarisation
reliably helps classical OCR and reliably hurts vision models**, which use the
grey gradient to resolve faint strokes that a threshold has already destroyed.
`default_policy` therefore omits it.

**If you want to force a specific sequence** (you have measured, you are sure):

```python
policy = dp.Policies.fixed_policy(
    dp.Ops.deskew(),
    dp.Ops.normalize_illumination(),
    dp.Ops.binarize(method="sauvola"),
)
doc = dp.preprocess(doc, policy=policy)
```

A warning that is easy to ignore and expensive to learn: **a fixed sequence applied
to every page is usually worse than no preprocessing at all.** Sharpening a page
that was never blurred destroys signal that was intact. Corrections should be
driven by measurements, which is why the adaptive policies exist.

---

### 4. I need fields, not a wall of text

**The problem.** "Total amount" is somewhere on page 3, formatted as
`1,18,250.00`, next to a similar-looking number that is the gross. You need a
`Decimal`, not a paragraph.

**The solution.** Describe the shape you want and hand it over. The simplest
schema is a dict — no pydantic required:

```python
doc = dp.Ingest.ingest("bill.pdf")
doc = dp.read(dp.preprocess(doc))

schema = {
    "hospital_name": {"type": "string", "description": "name of the hospital"},
    "bill_number":   {"type": "string"},
    "admission_date": {"type": "date"},
    "discharge_date": {"type": "date", "required": False},
    "total":         {"type": "number", "description": "final amount payable"},
}

result = dp.extract(
    doc,
    schema=schema,
    context="Indian private hospital bill. Amounts in INR. Dates are DD-MM-YYYY.",
    client=dp.AnthropicClient(model="claude-sonnet-5"),
)

result.value("total")              # Decimal("118250.00")
result.confidence("total")         # 0.94
```

Field types are `string`, `number`, `integer`, `boolean`, `date`, `array`,
`object`. Values are coerced for you: `"1,18,250.00"` becomes a `Decimal`,
`"12-04-2024"` becomes a `date`.

**With pydantic**, you get a typed model back in `result.model`:

```python
from pydantic import BaseModel
from typing import Optional
from datetime import date
from decimal import Decimal

class HospitalBill(BaseModel):
    hospital_name: str
    patient_name: str
    admission_date: date
    discharge_date: Optional[date] = None
    total: Decimal

result = dp.extract(doc, schema=HospitalBill, client=client)
result.model.total          # Decimal("118250.00"), typed
```

Dataclasses work too. The library adapts to your schema rather than asking you to
adopt its own.

**The `context` string earns its keep.** It is where you say things the document
assumes and the model cannot know: the currency, the date order, that "IPD" means
inpatient. One sentence of context typically removes a whole class of error.

---

### 5. Which of these values can I actually trust?

**The problem.** You extracted forty fields from a bad scan. Some are certainly
right, a few are certainly wrong, and you cannot afford to have a human check all
forty. You need to know which ones to look at.

**The solution — and the thing to *not* do.** Do not ask the model how confident
it is. A model's self-reported confidence measures how confident the text
*sounds*. A cleanly-formatted hallucination scores high; a correct reading of a
smudged digit scores low. Handing that number to a reviewer labelled "confidence"
is worse than handing them nothing, because they will believe it.

`docpipe` instead fuses several signals that fail *independently*:

| Signal               | What it actually measures                                          |
| -------------------- | ------------------------------------------------------------------ |
| `cross_read_agree` | two different engines read the same string from the same rectangle |
| `validation`       | the value survived your arithmetic and cross-field rules           |
| `text_support`     | the value genuinely appears in text some engine read               |
| `backend_conf`     | the OCR engine's own confidence for that region                    |
| `page_quality`     | measured degradation, localised to the evidence rectangle          |
| `format_match`     | the value looks like the type it claims to be                      |

```python
result = dp.extract(
    doc, schema, client=client,
    validators=[
        dp.Validators.line_items_sum_to_total(),
        dp.Validators.date_order("admission_date", "discharge_date"),
        dp.Validators.required_fields("hospital_name", "total"),
        dp.Validators.field_matches("bill_number", r"\w"),
    ],
)

for field in result.low_confidence(0.8):                 # the review queue
    print(field.name, field.value, "%.2f" % field.confidence, field.signals)

result.is_valid            # did every validator pass?
result.issues              # which ones did not, and on which fields
result.mean_confidence     # one number for the whole extraction
```

A validator is any callable taking the extracted data and returning a
`ValidationIssue` or `None`:

```python
def gross_minus_discount_is_total(bill):
    gross, discount, total = bill.get("gross"), bill.get("discount") or 0, bill.get("total")
    if gross is None or total is None:
        return None
    if abs((gross - discount) - total) > decimal.Decimal("1.00"):
        return dp.ValidationIssue("gross - discount is not total",
                                  fields=["gross", "discount", "total"])
    return None
```

Note what happens next: a failed validator does not merely produce a warning, it
*pulls that field's confidence down*, so the field shows up in your review queue
automatically.

**The honest caveat.** A confidence number that has never been checked against
labelled data is decoration. To make `0.9` mean "right about ninety percent of the
time", calibrate it — see use case 8.

---

### 6. An auditor asked me where this number came from

**The problem.** Six months later, someone disputes a figure. "The model said so"
is not an answer. You need to point at the rectangle on the page.

**The solution.** Every field carries evidence, and it is recorded at read time
because it is unrecoverable afterwards.

```python
field = result.fields["total"]
field.value           # Decimal("118250.00")
field.evidence        # [BBox(page=3, 402.0, 688.0, 471.0, 699.0)]
field.pages           # [3]
field.method          # how the value was obtained
```

Those coordinates are in PDF points on the original page, so you can draw them
straight onto a rendered page in a review UI, or crop the region for a reviewer to
look at:

```python
box = field.evidence[0]
page = doc.pages[box.page]
x0, y0, x1, y1 = box.to_pixels(page.raster_dpi)
crop = page.raster()[y0:y1, x0:x1]        # hand this to a reviewer
```

**The limitation, stated plainly.** A vision model returns a wall of text with no
coordinates. Rather than invent boxes — which would point a reviewer at the wrong
part of the page, worse than no box at all — spans from a vision read are marked
`approximate_bbox`, and tighter geometry is recovered by matching the value against
a second, geometry-bearing read. If provenance matters to you, run a cheap OCR
pass alongside the vision pass rather than the vision pass alone.

---

### 7. This is costing too much

**The problem.** Sending every page of every document to a vision model works, and
the bill at the end of the month is indefensible. Most of those pages did not need
a model.

**The solution.** Three independent levers.

**Lever one — routing.** Send each page to the cheapest reader that can handle it.
This is what `default_router` already does:

```python
def default_router(page):
    if page.kind is dp.PageKind.DIGITAL_NATIVE:
        return "pymupdf"        # free and exact
    if page.quality.verdict is dp.Verdict.UNREADABLE or page.layout.has_handwriting:
        return "vlm:claude"     # only the genuinely hard pages
    return "paddle"             # clean print, especially ruled tables
```

Write your own with `RuleRouter`, first match wins:

```python
router = (dp.RuleRouter(fallback="tesseract")
          .add(lambda p: p.kind is dp.PageKind.DIGITAL_NATIVE, "pymupdf")
          .add(lambda p: p.layout.is_tabular, "paddle")
          .add(lambda p: not p.quality.is_readable, "vlm:claude"))

doc = dp.read(doc, router=router)
```

**Lever two — caching.** Reads are keyed on the raster's content, so re-running a
pipeline over documents you have already read costs nothing:

```python
cache = dp.DiskCache(os.path.expanduser("~/.cache/docpipe"))   # or DiskCache(), which
backend = dp.CachingBackend(dp.TesseractOCR(), cache=cache)    # uses $DOCPIPE_CACHE_DIR
doc = dp.read(doc, backend=backend)
```

**Lever three — a hard ceiling.** When you want a run to stop rather than overrun:

```python
try:
    doc = dp.read(doc, budget=2.50)          # USD; raises when passed
except dp.BudgetExceeded as exc:
    print("stopped early:", exc)

router = dp.BudgetRouter(dp.default_router, budget=5.00, cheap_backend="tesseract")
```

**About the numbers.** `PRICING` ships **empty**, on purpose. Token counts are
always tracked, but money reads `0.00` and warns until you say what your account
actually pays:

```python
dp.Pricing.set_pricing("claude-sonnet-5", input_per_mtok=3.00, output_per_mtok=15.00)
```

A wrong price is worse than a missing one, because it produces a plausible budget
report that nobody rechecks.

---

### 8. Did my change actually make things better?

**The problem.** You tuned a threshold, or switched OCR engines, or the model
provider shipped a new version underneath you. You looked at five documents, they
seemed fine, you shipped. This is not a measurement, and it is how accuracy
quietly rots.

**The solution.** The evaluation harness. If you use only one part of this
library, this should be it.

Lay a dataset out as documents beside their ground truth — `bill_001.pdf` next to
`bill_001.json`:

```json
{"hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL", "total": "118250.00"}
```

Then run two whole pipelines against it and compare:

```python
suite = dp.EvalSuite.from_dir("datasets/hospital_bills_v3")

baseline  = dp.Pipeline(name="baseline",  policy=dp.Policies.default_policy,
                        schema=schema, client=client)
candidate = dp.Pipeline(name="candidate", policy=dp.Policies.ocr_policy,
                        schema=schema, client=client)

report = suite.run({"baseline": baseline, "candidate": candidate})
print(report.summary())

report.by_field()                 # which fields moved, not just the average
report.by_page_quality()          # did we only improve the pages that were fine?
report.worst_cases(limit=10)      # where to look first
report.compare("baseline", "candidate")
report.save("reports/run-2026-08.json")
```

**`by_page_quality()` is the one to watch.** A change that lifts the average by
improving pages that already worked, while leaving the degraded pages exactly as
broken as before, has not solved the problem you set out to solve.

**Gate CI on it.** `docpipe eval --baseline` exits non-zero on a regression:

```bash
docpipe eval datasets/bills --schema schema.json \
        --report reports/run.json \
        --baseline reports/release-2026-08.json --tolerance 0.01
```

**And close the confidence loop.** This is what turns confidence from decoration
into a number you can set a threshold against:

```python
calibrator = report.fit_calibrator(kind="platt")        # or "isotonic"
result = dp.extract(doc, schema, client=client, calibrator=calibrator)
report.confidence_curve()                               # is a 0.9 right 90% of the time?
```

---

### 9. I want to test my code without spending money

**The problem.** Your test suite needs to exercise your schema, your validators
and your error handling. It must not call a paid API, must not need a network, and
must produce the same answer every time.

**The solution.** `EchoClient` is a deterministic stand-in for a model, and it
ships in the library precisely because every consuming project needs one.

```python
client = dp.EchoClient(response='{"total": "1,18,250.00", "patient_name": "R KUMAR"}')
result = dp.extract(doc, schema, client=client)
assert result.value("total") == decimal.Decimal("118250.00")
```

It also takes a callable, so you can vary the reply by prompt — useful for testing
retry and chunking paths:

```python
def fake(prompt):
    return '{"total": "0"}' if "page 2" in prompt else '{"total": "500"}'

client = dp.EchoClient(response=fake)
```

From the CLI:

```bash
docpipe extract bill.pdf --schema schema.json --client echo --echo '{"total": "1200"}'
```

No test in this repository touches a network or a paid API, and yours does not have
to either.

---

### 10. My OCR engine isn't in the list

**The problem.** You have a commercial engine, an internal service, or a
fine-tuned model that beats everything shipped here on your documents. You do not
want to fork the library to use it.

**The solution.** You never have to. Subclass `BaseBackend`, implement `_read` and
`is_available`, and register it. The base class handles timing, span sourcing and
script detection.

```python
class AcmeOCR(dp.BaseBackend):
    name = "acme"
    preferred_dpi = 300

    def is_available(self):
        return True

    def _read(self, page):
        img = page.raster(self.preferred_dpi)
        spans = []
        for word, (x0, y0, x1, y1), conf in acme_sdk.recognise(img):
            spans.append(dp.TextSpan(
                text=word,
                bbox=dp.BBox.from_pixels(page.index, x0, y0, x1, y1, page.raster_dpi),
                source=self.name,
                confidence=conf,
            ))
        return dp.ReadResult(spans=spans, backend=self.name)

dp.registry.register("acme", lambda: AcmeOCR())     # a factory, not an instance
doc = dp.read(doc, backend="acme")
```

**Register a factory, never an instance.** Constructing a heavyweight engine loads
hundreds of megabytes of model weights; a document that never routes to it must not
pay for that.

Once registered, your backend is a first-class citizen: routers can pick it,
`EnsembleBackend` can cross-check it against another engine, `CachingBackend` can
cache it, and the eval harness can A/B it against what you use today.

**Wrappers you get for free:**

```python
dp.CachingBackend(inner, cache=dp.DiskCache())          # skip repeat work
dp.RetryingBackend(inner, attempts=3)                   # survive flaky services
dp.EnsembleBackend(primary, secondary)                  # two reads -> agreement signal
```

`EnsembleBackend` is the one that pays for itself twice: it produces the
`cross_read_agree` confidence signal, which is the strongest evidence available
that a value is genuinely on the page.

---

### 11. I need an image correction that doesn't exist

**The problem.** Your documents have a specific defect — a printed watermark, a
perforation strip down one edge, a dot-matrix pattern — and no shipped op removes
it.

**The solution.** Write a function over the raster and decorate it. It becomes a
first-class op: composable, serialisable, recorded in `page.history`.

```python
@dp.register_op("remove_watermark", geometric=False, needs_raster=True)
def remove_watermark(img, page, threshold=200):
    """Drop the pale diagonal watermark that survives thresholding.

    The threshold is 200 because the watermark prints at about 20% grey while
    the lightest genuine ink on these forms measures around 150.
    """
    out = img.copy()
    out[out > threshold] = 255
    return out                 # or None, meaning "nothing to correct here"

policy = dp.Policies.fixed_policy(dp.Ops.deskew(), remove_watermark(threshold=210))
doc = dp.preprocess(doc, policy=policy)
```

Two things to get right:

- **Return `None` when there is nothing to fix.** The page is then left untouched.
  Returning a "corrected" version of an already-clean page is how you lose signal.
- **Set `geometric=True` if your op moves pixels** (rotation, cropping, perspective
  correction). Existing span coordinates then get remapped; see `deskew` for the
  pattern.

Ops are data, so a policy round-trips through JSON and can be reproduced exactly
from an eval report:

```python
recorded = op.to_dict()
same_op = dp.op_from_dict(recorded)
```

---

### 12. I can't use an LLM at all

**The problem.** Air-gapped deployment, a compliance rule, or simply a form so
fixed in layout that a model is overkill.

**The solution.** Label-anchored rules. Find a printed label, read the value
beside or below it. No model, no network.

```python
rules = [
    dp.FieldRule(name="bill_number", label=r"Bill\s*No", direction="right"),
    dp.FieldRule(name="total", label=r"Total Amount", direction="right",
                 value_pattern=r"([\d,]+\.\d{2})", type_name="number"),
    dp.FieldRule(name="admission_date", label=r"Admission Date",
                 direction="right", type_name="date"),
]

fields = dp.extract_with_rules(doc, rules)
fields["total"].value        # Decimal("118250.00")
fields["total"].evidence     # and you still get the rectangle
```

`direction` may be `right`, `same_line` or `below`. `occurrence` picks which match
you mean when a label repeats — a per-page "Total" on a multi-page bill, for
instance. A rule that matches nothing reports it in `warnings` rather than raising,
so one missing label does not cost you the document.

You can also mix the two: rules for the fields that sit in fixed positions, a model
for the ones that move.

---

### 13. The dates and amounts come out wrong for my country

**The problem.** `1,23,456.78` parsed as `1.23`. `04-12-2024` read as 4 December
when the document meant 12 April. Devanagari digits left as-is. Every one of these
is silent — you get a number, just the wrong one.

**The solution.** The `Text` namespace handles the culture-specific layer, and
`read()` applies the normalisation pass by default.

```python
dp.Text.parse_amount("1,23,456.78")            # Decimal("123456.78") — Indian grouping
dp.Text.parse_amount("1.234,56")               # Decimal("1234.56")   — European
dp.Text.parse_date("12-04-2024", dayfirst=True)   # date(2024, 4, 12)
dp.Text.normalize_digits("१२३४")                # "1234"
dp.Text.detect_script("नमस्ते")                  # "Deva" — the dominant script
dp.Text.detect_currency("₹ 1,18,250")
dp.Text.fix_ocr_confusions("1O5O", expect="numeric")   # O -> 0 where a digit belongs
```

Normalise an entire document in place:

```python
doc = dp.Text.normalize_document(doc)
```

**And say it in the context string.** When extracting, one line — "Amounts are in
INR and may use Indian digit grouping; dates are DD-MM-YYYY" — removes most of this
class of error before it happens.

---

### 14. Two documents were photocopied onto one page

**The problem.** Someone put two A5 bills on the glass and scanned them as one A4.
Read as a single page, the columns interleave and the text is nonsense.

**The solution.**

```python
doc = dp.Ops.split_document(doc)          # detects the gutter, splits each page
```

It finds the whitespace gutter and splits along it, producing one page per
document. `axis` may be `auto`, `horizontal` or `vertical`; `max_parts` caps how
many pieces a page may become. Pages with no gutter pass through untouched.

If you use the `Pipeline` object, set `split_pages=True` and it happens before
everything else.

---

### 15. I have 5,000 files, not one

**The problem.** A directory, or an inbox, or a nightly drop. Three of the files
are corrupt. You need the other 4,997 processed anyway, and you need it to finish
before morning.

**The solution.**

```python
docs = dp.Ingest.ingest_dir("inbox/", pattern=r"\.pdf$", recursive=True)
```

Unreadable files come back as an empty `Document` carrying a warning rather than
aborting the batch — so one corrupt scan cannot cost you the run.

Pages are processed in parallel where it helps:

```python
doc = dp.preprocess(doc, max_workers=8)
doc = dp.read(doc, max_workers=8)
```

And memory stays bounded, because rasters are lazy and can be dropped once you are
done with the pixels:

```python
doc.release_rasters()          # keep the text and geometry, free the images
doc.save_json("out/doc.json")  # the full IR, reloadable with Document.load_json
```

Email attachments ingest directly, which matters if your documents arrive as
`.eml`:

```python
doc = dp.Ingest.ingest("claim.eml")        # body plus every readable attachment
```

---

### 16. I can't install OpenCV here

**The problem.** A locked-down deployment target, a platform with no wheel, or a
security policy that will not approve the package. A library that hard-fails
without OpenCV cannot be used at all in those places.

**The solution.** OpenCV is optional. Image operations have pure-NumPy fallbacks,
and the test suite runs both paths, so this is a supported configuration rather
than a hopeful claim.

```bash
export DOCPIPE_DISABLE_OPENCV=1        # force the fallback everywhere
```

```python
dp.Caps.set_opencv_enabled(False)      # process-wide

with dp.Caps.without_opencv():         # or just for this block, to compare
    doc = dp.preprocess(doc)
```

The fallbacks are slower and occasionally differ slightly in the last decimal
place. They are not toys.

---

### 17. I want this inside a web service

**The problem.** You need one configured object you can build at start-up, reuse
per request, log the configuration of, and reproduce later.

**The solution.** `Pipeline` is the whole thing as data.

```python
PIPELINE = dp.Pipeline(
    name="claims-v3",
    policy=dp.Policies.vlm_policy,
    router=dp.default_router,
    schema=HospitalBill,
    context="Indian private hospital bill. Amounts in INR.",
    client=dp.AnthropicClient(model="claude-sonnet-5"),
    validators=[dp.Validators.line_items_sum_to_total()],
    calibrator=CALIBRATOR,
    render_dpi=300,
    max_workers=4,
    budget=1.00,
)

def handle(upload_bytes):
    result = PIPELINE.run(upload_bytes)         # bytes, path or file object
    return result.to_dict()
```

`PIPELINE.to_dict()` gives you a JSON record of exactly how it was configured —
policy, router, backend, validators, version — which is what you attach to a
result so that a disagreement six months later is settled by looking it up rather
than by arguing.

For a one-off, `process()` is the same thing in a single call:

```python
result = dp.process("bill.pdf", schema=schema, client=client,
                    context="...", validators=[...])
```

**Observability:**

```python
tracer = dp.Tracer()
with tracer.span("ingest"):
    doc = dp.Ingest.ingest(path)
tracer.to_dict()               # per-stage timings

tracker = dp.CostTracker()     # tokens and money, per backend
```

---

## The CLI cookbook

Every command works on PDFs, images, TIFFs and `.eml` files.

| Command                                                                      | What it answers                               |
| ---------------------------------------------------------------------------- | --------------------------------------------- |
| `docpipe caps`                                                             | What is installed and usable right now?       |
| `docpipe info scan.pdf`                                                    | How many pages, what kind, what metadata?     |
| `docpipe quality scan.pdf --show-policy`                                   | **Why is this extracting badly?**       |
| `docpipe preprocess scan.pdf --policy ocr --out ./debug`                   | What do the pages look like after correction? |
| `docpipe read scan.pdf --backend tesseract`                                | What text comes out?                          |
| `docpipe read scan.pdf --json`                                             | The full IR — spans, boxes, confidences      |
| `docpipe extract bill.pdf --schema s.json --client anthropic`              | What are the field values?                    |
| `docpipe extract bill.pdf --schema s.json --client echo --echo '{...}'`    | The same, with no API call                    |
| `docpipe eval datasets/bills --schema s.json --baseline reports/prev.json` | Did anything regress?                         |

Shared flags: `--dpi`, `--max-pages`, `--workers`, and `-v` / `-q` for logging.
`--policy` takes `default`, `ocr`, `vlm` or `none`.

`docpipe quality --show-policy` is the one to reach for first when a document
extracts badly. It usually turns "the OCR is bad" into "the page is at 180 DPI and
skewed by three degrees", which is a fixable problem.

---

## Picking only the parts you need

Every layer works standalone. You are not obliged to adopt the pipeline to use one
piece of it.

| You only want…                          | Use just this                               | Needs            |
| ---------------------------------------- | ------------------------------------------- | ---------------- |
| To open a file and know what it is       | `Ingest.ingest`, `doc.kinds()`          | pymupdf / pillow |
| To score how bad a scan is               | `Quality.measure_document`                | numpy            |
| To clean up images before your own OCR   | `preprocess`, `Ops.*`                   | numpy            |
| To swap OCR engines behind one interface | `read`, `registry`                      | your engine      |
| To normalise messy dates and amounts     | `Text.*`                                  | nothing          |
| To fuse your own confidence signals      | `Confidence.fuse_confidence`              | nothing          |
| To measure whether a change helped       | `EvalSuite`, `EvalReport`               | nothing          |
| To calibrate any probability you produce | `PlattCalibrator`, `IsotonicCalibrator` | nothing          |

The last four need no third-party packages at all — you can use the eval harness
and the confidence machinery against a pipeline that has nothing else to do with
this library.

---

## Things that surprise people

**"The cost says 0.00."** It will, until you call `Pricing.set_pricing` for your
models. Token counts are real; the money is not, and the library warns rather than
guessing, because a plausible-but-wrong budget report is worse than an obviously
missing one.

**"Confidence is stuck around 0.5."** That is the prior showing through: the
signals are mostly absent. Add validators, or run an `EnsembleBackend` so
`cross_read_agree` has something to say. A signal set to `None` means *absent*, not
zero — a backend that honestly declines to guess is not punished for it.

**"My confidence numbers don't mean anything."** Correct, until you calibrate them
against labelled data. `report.fit_calibrator()` is the step that makes `0.9` mean
"right about nine times in ten"; `report.confidence_curve()` shows you whether it
does.

**"The vision model gave me no bounding boxes."** By design. It does not produce
coordinates, so spans are marked `approximate_bbox` rather than being given
invented ones. Run a second, geometry-bearing read to recover tight boxes.

**"Binarising made things worse."** Very likely you binarised for a vision model.
Use `vlm_policy` there and `ocr_policy` for classical engines.

**"The page is upside down and `auto_orient` didn't fix it."** The
projection-profile fallback distinguishes portrait from landscape but cannot detect
a 180° flip; that needs Tesseract's OSD. The method used is reported in
`page.meta["orientation_method"]` rather than hidden.

**"The default thresholds are wrong for my documents."** They are priors, not
constants — they come from scanned Indian hospital bills and court filings.
Recalibrate them against your own labelled set. That is what the eval harness is
for, and `QualityThresholds` exists so you can pass your own.

**"I need something the library doesn't do."** Register an op, a backend or a
router. If you ever have to fork the library to get your job done, that is a defect
in the library, not in your requirements — please open an issue.

---

## Where to look next

- **[README.md](README.md)** — why the library is designed this way; the *Scope
  boundary* and *Honest limitations* sections are the authority on what belongs
  here and what does not.
- **[API reference](https://utkarsh5026.github.io/Docpipe/)** — every class,
  function and constant, generated from the source itself, with a filter box and a
  source link on each entry. Build it locally with
  `python docs/build_reference.py`.
- **[examples/hospital_bill.ipynb](examples/hospital_bill.ipynb)** — the whole
  library walked end to end, one capability per cell, with the page, the boxes and
  the tables rendered inline. Start here if you would rather see it than read it.
- **[examples/hospital_bill.py](examples/hospital_bill.py)** — the same worked
  example as a script, runnable with no API key and no network. It is the shortest
  demonstration of how little a domain project has to own: a schema, two
  validators and a context string.
- **[tests/](tests/)** — the closest thing to exhaustive usage documentation.
  Roughly 660 tests, one file per layer, all offline.

```bash
pip install -e ".[dev]"
pytest -q
python examples/hospital_bill.py

pip install -e ".[notebook]"                 # numpy, pillow, ipykernel, jupyterlab
jupyter lab examples/hospital_bill.ipynb     # or just open it in your editor
```
