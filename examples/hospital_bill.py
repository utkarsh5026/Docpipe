"""A worked end-to-end example: extracting a scanned hospital bill.

Run it with no arguments to see the whole pipeline against a synthetic bill and
a deterministic stub model -- no API key, no network, no spend::

    python examples/hospital_bill.py

Point it at a real document to use a real model (needs ANTHROPIC_API_KEY)::

    python examples/hospital_bill.py path/to/bill.pdf

The point of this file is to show how little a *domain* project has to own:
the schema, two validators, and a context string.  Everything else is library.
"""

import decimal
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import docpipe as dp


# ---------------------------------------------------------------------------
# 1. The schema.  This is domain knowledge, so it lives here and never in core.
# ---------------------------------------------------------------------------

BILL_SCHEMA = {
    "hospital_name": {"type": "string", "description": "name of the hospital"},
    "patient_name": {"type": "string"},
    "bill_number": {"type": "string", "description": "invoice or bill number"},
    "admission_date": {"type": "date"},
    "discharge_date": {"type": "date", "required": False},
    "gross_amount": {"type": "number"},
    "discount": {"type": "number", "required": False},
    "total": {"type": "number", "description": "final amount payable"},
}

CONTEXT = (
    "Indian private hospital final bill. Amounts are in INR and may use Indian "
    "digit grouping (1,23,456.78). Dates are usually DD-MM-YYYY."
)


# ---------------------------------------------------------------------------
# 2. Business rules.  Also domain knowledge -- and they feed confidence fusion,
#    which is why a failing rule surfaces the field for human review.
# ---------------------------------------------------------------------------

def gross_minus_discount_is_total(bill):
    gross = bill.get("gross_amount")
    discount = bill.get("discount") or decimal.Decimal(0)
    total = bill.get("total")
    if gross is None or total is None:
        return None
    if abs((gross - discount) - total) > decimal.Decimal("1.00"):
        return dp.ValidationIssue(
            "gross %s minus discount %s is not total %s" % (gross, discount, total),
            fields=["gross_amount", "discount", "total"])
    return None


VALIDATORS = [
    gross_minus_discount_is_total,
    dp.date_order("admission_date", "discharge_date"),
    dp.field_matches("bill_number", r"\w"),
    dp.required_fields("hospital_name", "total"),
]


# ---------------------------------------------------------------------------
# 3. A synthetic bill, so the example runs anywhere.
# ---------------------------------------------------------------------------

BILL_TEXT = [
    "SUNRISE MULTISPECIALITY HOSPITAL",
    "12 Nehru Road, Pune 411001",
    "",
    "FINAL BILL CUM RECEIPT",
    "",
    "Patient Name : RAMESH KUMAR SHARMA",
    "Bill No : INV-2024-04-7781",
    "Admission Date : 12-04-2024",
    "Discharge Date : 19-04-2024",
    "",
    "Room Rent Deluxe                  7      42000.00",
    "Consultant Visit Charges         11      16500.00",
    "Pharmacy and Consumables          1      31240.50",
    "Investigation Charges             1      18760.00",
    "Operation Theatre Charges         1      14750.00",
    "",
    "Gross Amount                           123250.50",
    "Discount                                 5000.50",
    "Total Amount Payable                   118250.00",
]

STUB_RESPONSE = json.dumps({
    "hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL",
    "patient_name": "RAMESH KUMAR SHARMA",
    "bill_number": "INV-2024-04-7781",
    "admission_date": "12-04-2024",
    "discharge_date": "19-04-2024",
    "gross_amount": "1,23,250.50",
    "discount": "5,000.50",
    "total": "1,18,250.00",
})


def synthetic_scan():
    """Render the bill, then push it through a realistically nasty channel."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (1240, 1200), 255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    y = 60
    for line in BILL_TEXT:
        draw.text((70, y), line, fill=0, font=font)
        y += 42

    scan = np.array(image).astype(float)
    scan *= np.linspace(1.0, 0.6, scan.shape[1])[None, :]        # a shadow
    scan = dp.gaussian_blur(dp.ensure_uint8(scan), 1.1)          # soft focus
    scan = scan.astype(float) + np.random.RandomState(0).normal(0, 9, scan.shape)
    scan = dp.rotate_image(dp.ensure_uint8(np.clip(scan, 0, 255)),
                           -2.4, border_value=235)               # crooked feed
    return dp.resize_image(scan, scale=0.55)                     # low resolution


class StubOCR(dp.BaseBackend):
    """Stands in for Tesseract/Paddle so the example needs no OCR install."""

    name = "stub-ocr"
    needs_raster = False

    def supports(self, page):
        return True

    def _read(self, page):
        spans = []
        for i, line in enumerate(l for l in BILL_TEXT if l.strip()):
            y = 40.0 + i * 22.0
            x = 40.0
            for word in line.split():
                width = 7.0 * len(word)
                spans.append(dp.TextSpan(
                    text=word, bbox=dp.BBox(page.index, x, y, x + width, y + 16.0),
                    source=self.name, confidence=0.93, line=i))
                x += width + 7.0
        return dp.ReadResult(spans=spans, cost=dp.Cost(calls=1), backend=self.name)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None

    if source:
        client = dp.AnthropicClient(model="claude-sonnet-5")
        backend = None                       # let the router decide
        print("Extracting %s with a real model\n" % source)
    else:
        client = dp.EchoClient(STUB_RESPONSE)
        backend = StubOCR()
        source = dp.encode_png(synthetic_scan())
        print("Extracting a synthetic degraded scan with stubbed OCR and model\n")

    # -- ingest ------------------------------------------------------------
    doc = dp.ingest(source, filename="bill.png", dpi=150)
    print("ingested: %s" % doc)

    # -- measure -----------------------------------------------------------
    dp.measure_document(doc)
    page = doc.pages[0]
    print("\nmeasured quality:")
    print("  %s" % page.quality)
    print("  policy would apply: %s"
          % ([op.name for op in dp.default_policy(page)] or "nothing"))

    # -- preprocess and read ----------------------------------------------
    doc = dp.preprocess(doc, policy=dp.default_policy)
    print("\nafter preprocessing:")
    print("  %s" % doc.pages[0].quality)
    print("  history: %s" % doc.pages[0].history_summary())

    doc = dp.read(doc, backend=backend)
    doc = dp.normalize_document(doc, in_place=True)
    print("\nread %d characters via %s" % (doc.char_count, doc.meta["read_backends"]))

    # -- extract -----------------------------------------------------------
    result = dp.extract(doc, BILL_SCHEMA, context=CONTEXT, client=client,
                        validators=VALIDATORS)

    print("\nextracted fields:")
    for name, field in result.fields.items():
        evidence = ("page %d" % field.evidence[0].page) if field.evidence else "-"
        print("  %-16s %-34s conf=%.2f  evidence=%s"
              % (name, str(field.value)[:34], field.confidence, evidence))

    print("\nvalid: %s   mean confidence: %.2f   cost: %s"
          % (result.is_valid, result.mean_confidence, result.cost))
    for issue in result.issues:
        print("  issue: %s" % issue)

    needs_review = result.low_confidence(0.75)
    if needs_review:
        print("\nsend to a human reviewer:")
        for field in needs_review:
            print("  %-16s conf=%.2f  signals=%s"
                  % (field.name, field.confidence,
                     dict((k, v) for k, v in field.signals.items() if v is not None)))
    else:
        print("\nnothing below the review threshold.")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
