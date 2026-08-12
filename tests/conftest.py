"""Shared fixtures for the docpipe test suite.

The central idea here is the *synthetic page with known ground truth*: pages
are rendered from text with Pillow, and the exact pixel box of every word is
recorded while drawing.  That gives three things the suite would otherwise have
to fake:

* a ground-truth span set, so provenance and evidence recovery can be asserted
  exactly rather than approximately;
* a deterministic "OCR engine" (:class:`TruthBackend`) that can be told to make
  a controlled number of mistakes, so confidence and eval metrics can be tested
  against a known error rate;
* realistic degradation, by pushing a clean render through blur, noise, skew
  and downsampling with known parameters.

Nothing here talks to a network or a paid API.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import docpipe as dp  # noqa: E402

np = pytest.importorskip("numpy")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]

requires_cv2 = pytest.mark.skipif(not dp.have("cv2"), reason="OpenCV not installed")
requires_pil = pytest.mark.skipif(not dp.have("PIL"), reason="Pillow not installed")
requires_pymupdf = pytest.mark.skipif(dp._fitz() is None, reason="PyMuPDF not installed")


def _font(size):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_text_page(lines, width=1240, height=1754, font_size=26, margin=70,
                     leading=1.6, page_index=0, dpi=150):
    """Render ``lines`` and return ``(image, spans)`` with true word boxes.

    The default canvas is A4 at 150 DPI.  ``spans`` are in canonical page
    points, matching what a backend is expected to produce.
    """
    from PIL import Image, ImageDraw

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    spans = []
    y = float(margin)
    step = font_size * leading

    for line_no, line in enumerate(lines):
        x = float(margin)
        for word in line.split(" "):
            if not word:
                x += font_size * 0.4
                continue
            draw.text((x, y), word, fill=0, font=font)
            try:
                box = draw.textbbox((x, y), word, font=font)
            except AttributeError:  # very old Pillow
                # textsize() was removed in Pillow 10, which is why this is the
                # fallback and not the primary path.
                w, h = draw.textsize(word, font=font)  # type: ignore[attr-defined]
                box = (x, y, x + w, y + h)
            spans.append(dp.TextSpan(
                text=word,
                bbox=dp.BBox.from_pixels(page_index, box[0], box[1], box[2], box[3], dpi),
                source="truth", confidence=0.99, line=line_no))
            x = box[2] + font_size * 0.35
        y += step

    return np.array(image), spans


BILL_LINES = [
    "SUNRISE MULTISPECIALITY HOSPITAL",
    "12 Nehru Road, Pune 411001    GSTIN 27AABCS1429B1Z0",
    "",
    "FINAL BILL CUM RECEIPT",
    "",
    "Patient Name : RAMESH KUMAR SHARMA",
    "UHID : SMH-2024-88213      Age / Sex : 54 Y / M",
    "Admission Date : 12-04-2024",
    "Discharge Date : 19-04-2024",
    "Bill No : INV-2024-04-7781",
    "",
    "Description                     Qty        Amount",
    "Room Rent Deluxe                  7      42000.00",
    "Consultant Visit Charges         11      16500.00",
    "Pharmacy and Consumables          1      31240.50",
    "Investigation Charges             1      18760.00",
    "Operation Theatre Charges         1      14750.00",
    "",
    "Gross Amount                           123250.50",
    "Discount                                 5000.50",
    "Total Amount Payable                   118250.00",
    "",
    "Amount in words : One Lakh Eighteen Thousand Two Hundred Fifty Only",
]

BILL_TRUTH = {
    "hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL",
    "patient_name": "RAMESH KUMAR SHARMA",
    "bill_number": "INV-2024-04-7781",
    "admission_date": "2024-04-12",
    "discharge_date": "2024-04-19",
    "total": 118250.00,
}


@pytest.fixture(scope="session")
def bill_page():
    """A clean, synthetic hospital bill: ``(image, spans)``."""
    return render_text_page(BILL_LINES)


@pytest.fixture
def bill_image(bill_page):
    return bill_page[0]


@pytest.fixture
def bill_spans(bill_page):
    return [dp.dataclasses.replace(s) for s in bill_page[1]]


@pytest.fixture
def bill_document(bill_page):
    """A one-page Document whose spans are the ground truth."""
    image, spans = bill_page
    doc = dp.document_from_images([image], dpi=150, source_uri="synthetic-bill")
    doc.pages[0].spans = [dp.dataclasses.replace(s) for s in spans]
    doc.pages[0].kind = dp.PageKind.SCANNED
    dp.measure_page(doc.pages[0])
    return doc


@pytest.fixture
def blank_image():
    return np.full((600, 800), 255, dtype=np.uint8)


def degrade(image, blur=0.0, noise=0.0, skew=0.0, scale=1.0, shadow=0.0,
            seed=1234):
    """Push a clean render through a controlled, known channel."""
    out = np.asarray(image).astype(np.float64)
    if scale != 1.0:
        out = dp.resize_image(dp.ensure_uint8(out), scale=scale).astype(np.float64)
        out = dp.resize_image(dp.ensure_uint8(out), scale=1.0 / scale).astype(np.float64)
    if blur > 0:
        out = dp.gaussian_blur(dp.ensure_uint8(out), blur).astype(np.float64)
    if shadow > 0:
        h, w = out.shape[:2]
        gradient = np.linspace(1.0, 1.0 - shadow, w)[None, :]
        out = out * gradient
    if noise > 0:
        rng = np.random.RandomState(seed)
        out = out + rng.normal(0.0, noise, out.shape)
    out = dp.ensure_uint8(np.clip(out, 0, 255))
    if skew:
        out = dp.rotate_image(out, skew, border_value=255)
    return out


class TruthBackend(dp.BaseBackend):
    """A deterministic OCR engine with a configurable, known error rate.

    Returns the page's ground-truth spans, corrupting every ``error_every``-th
    one by a single character substitution.  That makes it possible to assert
    that a metric reports (say) 80% field accuracy, rather than asserting that
    it reports "something plausible".
    """

    name = "truth"
    needs_raster = False

    def __init__(self, spans_by_page, error_every=0, confidence=0.95,
                 drop_every=0, name=None):
        dp.BaseBackend.__init__(self, name=name)
        self.spans_by_page = dict(spans_by_page)
        self.error_every = error_every
        self.drop_every = drop_every
        self.confidence = confidence
        self.calls = 0

    def supports(self, page):
        return page.index in self.spans_by_page

    def _read(self, page):
        self.calls += 1
        out = []
        for i, span in enumerate(self.spans_by_page.get(page.index, [])):
            if self.drop_every and i % self.drop_every == 0:
                continue
            text = span.text
            confidence = self.confidence
            if self.error_every and i % self.error_every == 0 and text:
                text = ("X" + text[1:]) if text[0] != "X" else ("Y" + text[1:])
                confidence = 0.42
            out.append(dp.TextSpan(text=text, bbox=span.bbox, source=self.name,
                                   confidence=confidence, line=span.line))
        return dp.ReadResult(spans=out, cost=dp.Cost(calls=1), backend=self.name)


@pytest.fixture
def truth_backend(bill_page):
    return TruthBackend({0: bill_page[1]})


@pytest.fixture
def native_pdf(tmp_path, bill_page):
    """A PDF with a real embedded text layer."""
    fitz = dp._fitz()
    if fitz is None:
        pytest.skip("PyMuPDF not installed")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 60.0
    for line in BILL_LINES:
        if line.strip():
            page.insert_text((50, y), line, fontsize=9)
        y += 14
    path = str(tmp_path / "native_bill.pdf")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path, bill_image):
    """A PDF that is a picture of a page -- no text layer at all."""
    fitz = dp._fitz()
    if fitz is None:
        pytest.skip("PyMuPDF not installed")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=dp.encode_png(bill_image))
    path = str(tmp_path / "scanned_bill.pdf")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def image_file(tmp_path, bill_image):
    path = str(tmp_path / "bill.png")
    dp.save_image(bill_image, path)
    return path


@pytest.fixture
def eval_dataset(tmp_path, bill_image):
    """A tiny labelled dataset laid out the way EvalSuite.from_dir expects."""
    directory = tmp_path / "dataset"
    directory.mkdir()
    for i in range(3):
        dp.save_image(bill_image, str(directory / ("case%d.png" % i)))
        truth = dict(BILL_TRUTH)
        if i == 2:
            truth["total"] = 999999.0  # one case the pipeline will get "wrong"
        with open(str(directory / ("case%d.json" % i)), "w", encoding="utf-8") as fh:
            json.dump({"truth": truth, "tags": ["synthetic"]}, fh)
    return str(directory)


@pytest.fixture
def isolated_cache(tmp_path):
    return dp.DiskCache(directory=str(tmp_path / "cache"))


#: Set by CI to simulate a deployment target where no OpenCV wheel exists.
OPENCV_DISABLED = os.environ.get("DOCPIPE_DISABLE_OPENCV", "").strip().lower() \
    in ("1", "true", "yes", "on")


@pytest.fixture(autouse=True)
def _restore_opencv_setting():
    """Undo any global OpenCV toggle a test performed.

    Several parity tests flip the switch directly to compare the two code
    paths.  Without this, one of them leaks its setting into every test that
    follows, and a genuine fallback failure would be masked by whichever test
    happened to run first.
    """
    previous = dp._USE_OPENCV
    yield
    dp.set_opencv_enabled(previous)


@pytest.fixture(params=["opencv", "numpy"])
def both_image_backends(request):
    """Run a test twice: once with OpenCV, once forced onto NumPy fallbacks.

    The fallbacks are the reason the library works where an OpenCV wheel is
    unavailable, so that promise is tested rather than asserted.
    """
    if request.param == "opencv":
        if not dp.have("cv2"):
            pytest.skip("OpenCV not installed")
        if OPENCV_DISABLED:
            pytest.skip("DOCPIPE_DISABLE_OPENCV is set")
        previous = dp.set_opencv_enabled(True)
        yield "opencv"
        dp.set_opencv_enabled(previous)
    else:
        previous = dp.set_opencv_enabled(False)
        yield "numpy"
        dp.set_opencv_enabled(previous)
