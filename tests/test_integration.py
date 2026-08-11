"""End-to-end pipelines, and the properties that only show up when the layers
are composed.

Each test here corresponds to one of the three consumers the library was
designed for: a scanned medical bill, a native-text claim PDF, and an inbound
email with mixed attachments.  The point is to prove the abstraction survives
all three without any of them needing to reach inside it.
"""

import decimal
import json

import pytest

import docpipe as dp
from conftest import TruthBackend, degrade, requires_pymupdf

np = pytest.importorskip("numpy")


SCHEMA = {
    "hospital_name": "string",
    "patient_name": "string",
    "admission_date": "date",
    "discharge_date": "date",
    "total": "number",
}

RESPONSE = json.dumps({
    "hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL",
    "patient_name": "RAMESH KUMAR SHARMA",
    "admission_date": "2024-04-12",
    "discharge_date": "2024-04-19",
    "total": "118250.00",
})


class TestScannedBillPipeline:
    def test_degraded_scan_survives_the_whole_pipeline(self, bill_image, bill_spans,
                                                       tmp_path):
        """The headline case: a blurry, skewed, badly-lit scan in; typed fields
        with evidence and confidence out."""
        scan = degrade(bill_image, blur=1.2, noise=12.0, skew=-2.5, shadow=0.4)
        path = str(tmp_path / "scan.png")
        dp.save_image(scan, path)

        pipeline = dp.Pipeline(
            policy=dp.default_policy,
            backend=TruthBackend({0: bill_spans}),
            schema=SCHEMA,
            context="Indian private hospital bill. Amounts in INR.",
            client=dp.EchoClient(RESPONSE),
            validators=[dp.date_order("admission_date", "discharge_date")],
        )
        result = pipeline.run(path)

        assert result.value("total") == decimal.Decimal("118250.00")
        assert result.is_valid
        assert result["patient_name"].evidence
        page = result.document.pages[0]
        assert "deskew" in [h.op for h in page.history]
        assert abs(page.quality.skew_deg) < 0.5

    def test_preprocessing_history_is_a_full_audit_trail(self, bill_image, tmp_path):
        path = str(tmp_path / "scan.png")
        dp.save_image(degrade(bill_image, blur=2.0, skew=-3.0), path)
        doc = dp.preprocess(dp.ingest(path))
        history = doc.pages[0].history_summary()
        assert "measure_quality" in history and "deskew" in history

    def test_the_same_document_can_drive_two_pipelines_independently(self, bill_image,
                                                                    bill_spans):
        """A/B-ing two pipelines over one ingested document has to be safe."""
        doc = dp.document_from_images([degrade(bill_image, skew=-3.0)], dpi=150)
        baseline = dp.preprocess(doc, policy=dp.no_policy)
        candidate = dp.preprocess(doc, policy=dp.default_policy)
        assert [h.op for h in baseline.pages[0].history] != \
               [h.op for h in candidate.pages[0].history]
        assert doc.pages[0].history == []      # the original is untouched


@requires_pymupdf
class TestNativePdfPipeline:
    def test_native_text_is_read_free_and_exactly(self, native_pdf):
        pipeline = dp.Pipeline(schema=SCHEMA, client=dp.EchoClient(RESPONSE))
        result = pipeline.run(native_pdf)
        assert result.document.meta["read_backends"] == {0: "pymupdf"}
        assert result.document.meta["read_cost"]["amount"] == 0.0
        assert result.value("patient_name") == "RAMESH KUMAR SHARMA"

    def test_evidence_points_at_the_right_page(self, native_pdf):
        result = dp.Pipeline(schema=SCHEMA,
                             client=dp.EchoClient(RESPONSE)).run(native_pdf)
        assert result["total"].evidence
        assert result["total"].evidence[0].page == 0

    def test_a_scanned_pdf_routes_away_from_the_text_layer(self, scanned_pdf,
                                                           bill_spans):
        doc = dp.ingest(scanned_pdf)
        assert doc.pages[0].kind is dp.PageKind.SCANNED
        out = dp.read(doc, backend=TruthBackend({0: bill_spans}))
        assert "SUNRISE" in out.text()


class TestEmailPipeline:
    def _message(self, parts):
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = "Claim 47812 documents"
        msg["From"] = "clerk@hospital.example"
        msg.set_content("Total payable is 118250.00 as per the attached bill.")
        for name, data, types in parts:
            msg.add_attachment(data, maintype=types[0], subtype=types[1], filename=name)
        return msg.as_bytes()

    def test_mixed_attachments_land_in_one_document(self, bill_image, native_pdf):
        with open(native_pdf, "rb") as fh:
            pdf_bytes = fh.read()
        raw = self._message([
            ("scan.png", dp.encode_png(bill_image), ("image", "png")),
            ("claim.pdf", pdf_bytes, ("application", "pdf")),
        ])
        doc = dp.ingest(raw, filename="claim.eml")
        attachments = {p.meta.get("attachment") for p in doc.pages}
        assert {"scan.png", "claim.pdf"} <= attachments
        assert [p.index for p in doc.pages] == list(range(len(doc.pages)))

    def test_each_page_is_routed_on_its_own_merits(self, bill_image, native_pdf,
                                                   bill_spans):
        """Native PDF pages take the free exact path; the scan does not."""
        with open(native_pdf, "rb") as fh:
            pdf_bytes = fh.read()
        raw = self._message([
            ("scan.png", dp.encode_png(bill_image), ("image", "png")),
            ("claim.pdf", pdf_bytes, ("application", "pdf")),
        ])
        doc = dp.ingest(raw, filename="claim.eml")
        spans = {p.index: bill_spans for p in doc.pages}
        out = dp.read(doc, backend=None,
                      router=lambda p: "pymupdf" if p.kind is dp.PageKind.DIGITAL_NATIVE
                      else "fallback")
        dp.registry.register("fallback", TruthBackend(spans), replace=True)
        out = dp.read(doc, router=lambda p: "pymupdf"
                      if p.kind is dp.PageKind.DIGITAL_NATIVE else "fallback")
        chosen = set(out.meta["read_backends"].values())
        assert "pymupdf" in chosen
        dp.registry.register("fallback", lambda: None, replace=True)

    def test_email_body_text_is_extractable(self, bill_image):
        raw = self._message([("scan.png", dp.encode_png(bill_image), ("image", "png"))])
        doc = dp.ingest(raw, filename="claim.eml")
        assert "118250.00" in doc.text()


class TestCrossReadAndConfidence:
    def test_two_agreeing_engines_beat_one(self, bill_image, bill_spans):
        """Cross-read agreement is the strongest signal available, so a field
        confirmed by two independent reads must outrank the same field read
        once."""
        doc = dp.document_from_images([bill_image], dpi=150)
        dp.measure_document(doc)

        single = dp.read(doc, backend=TruthBackend({0: bill_spans}, name="a"))
        ensemble = dp.read(doc, backend=dp.EnsembleBackend(
            TruthBackend({0: bill_spans}, name="a"),
            TruthBackend({0: bill_spans}, name="b")))

        one = dp.extract(single, SCHEMA, client=dp.EchoClient(RESPONSE))
        two = dp.extract(ensemble, SCHEMA, client=dp.EchoClient(RESPONSE))
        assert two["total"].confidence > one["total"].confidence

    def test_a_degraded_page_lowers_confidence_for_the_same_answer(self, bill_image,
                                                                   bill_spans):
        """Confidence is a property of the channel, not of how the answer reads."""
        clean = dp.document_from_images([bill_image], dpi=150)
        rough = dp.document_from_images(
            [degrade(bill_image, blur=3.5, noise=25.0, shadow=0.6)], dpi=150)
        for doc in (clean, rough):
            dp.measure_document(doc)
            doc.pages[0].spans = [dp.dataclasses.replace(s) for s in bill_spans]

        good = dp.extract(clean, SCHEMA, client=dp.EchoClient(RESPONSE))
        bad = dp.extract(rough, SCHEMA, client=dp.EchoClient(RESPONSE))
        assert good.value("total") == bad.value("total")
        assert good["total"].confidence > bad["total"].confidence

    def test_a_failed_arithmetic_check_surfaces_the_field_for_review(self,
                                                                     bill_document):
        schema = dict(SCHEMA)
        schema["line_items"] = {"type": "array", "items": {"type": "object"}}
        response = json.loads(RESPONSE)
        response["line_items"] = [{"description": "Room", "amount": 1.0}]
        result = dp.extract(bill_document, schema,
                            client=dp.EchoClient(json.dumps(response)),
                            validators=[dp.line_items_sum_to_total()])
        assert not result.is_valid
        assert "total" in [f.name for f in result.low_confidence(0.75)]


class TestRuleAndModelCrossCheck:
    def test_rules_and_a_model_can_be_compared_for_free(self, bill_document):
        """Running a deterministic rule set alongside a model gives a genuine
        second opinion at no marginal cost."""
        rules = [dp.FieldRule(name="total", label=r"Total\s*Amount\s*Payable",
                              value_pattern=r"([\d,\.]+)", type_name="number")]
        by_rule = dp.extract_with_rules(bill_document, rules)["total"].value
        by_model = dp.extract(bill_document, SCHEMA,
                              client=dp.EchoClient(RESPONSE)).value("total")
        assert dp.value_agreement(by_rule, by_model) == 1.0

    def test_disagreement_is_detectable(self, bill_document):
        rules = [dp.FieldRule(name="total", label=r"Gross\s*Amount",
                              value_pattern=r"([\d,\.]+)", type_name="number")]
        by_rule = dp.extract_with_rules(bill_document, rules)["total"].value
        by_model = dp.extract(bill_document, SCHEMA,
                              client=dp.EchoClient(RESPONSE)).value("total")
        assert by_rule != by_model
        assert dp.value_agreement(by_rule, by_model) == 0.0


class TestPipelineObject:
    def test_serialises_its_configuration(self):
        pipeline = dp.Pipeline(schema=SCHEMA, client=dp.EchoClient("{}"),
                               validators=[dp.date_order("a", "b")], name="claims-v3")
        payload = pipeline.to_dict()
        assert payload["name"] == "claims-v3"
        assert payload["policy"] == "default_policy"
        assert payload["client"] == "echo"
        assert payload["docpipe_version"] == dp.__version__

    def test_without_a_schema_it_is_a_text_pipeline(self, native_pdf):
        result = dp.Pipeline(schema=None).run(native_pdf)
        assert result.model is None
        assert "SUNRISE" in result.document.text()

    def test_process_is_a_one_call_wrapper(self, native_pdf):
        result = dp.process(native_pdf, schema=SCHEMA, client=dp.EchoClient(RESPONSE))
        assert result.value("total") == decimal.Decimal("118250.00")

    def test_callable_directly(self, native_pdf):
        pipeline = dp.Pipeline(schema=SCHEMA, client=dp.EchoClient(RESPONSE))
        assert pipeline(native_pdf).value("patient_name") == "RAMESH KUMAR SHARMA"


class TestLargeBundleBehaviour:
    @requires_pymupdf
    def test_a_long_bundle_is_not_rasterised_eagerly(self, tmp_path, bill_image):
        """A 300-page bundle rendered eagerly at 400 DPI is tens of gigabytes;
        laziness has to live in the type."""
        fitz = dp._fitz()
        pdf = fitz.open()
        for _ in range(40):
            page = pdf.new_page(width=595, height=842)
            page.insert_image(fitz.Rect(0, 0, 595, 842),
                              stream=dp.encode_png(bill_image))
        path = str(tmp_path / "bundle.pdf")
        pdf.save(path)
        pdf.close()

        doc = dp.ingest(path, render_dpi=400)
        assert len(doc) == 40
        assert all(p._raster is None for p in doc.pages)
        doc.pages[0].raster()
        assert sum(1 for p in doc.pages if p._raster is not None) == 1

    @requires_pymupdf
    def test_rasters_can_be_released_after_use(self, scanned_pdf):
        doc = dp.ingest(scanned_pdf)
        doc.pages[0].raster()
        assert doc.pages[0]._raster is not None
        doc.release_rasters()
        assert doc.pages[0]._raster is None
        assert doc.pages[0].raster() is not None      # re-rendered on demand

    def test_chunking_keeps_every_page(self):
        doc = dp.Document(source_uri="bundle.pdf")
        for i in range(30):
            page = dp.Page(index=i)
            page.spans = [dp.TextSpan("w" * 800, dp.BBox(i, 0, 0, 10, 10))]
            doc.pages.append(page)
        chunks = dp.chunk_pages(doc, max_chars=3000, overlap_pages=0)
        seen = set()
        for chunk in chunks:
            seen.update(p.index for p in chunk.pages)
        assert seen == set(range(30))


class TestTracingAndBudgets:
    def test_tracer_records_nested_spans(self):
        tracer = dp.Tracer()
        with tracer.span("ingest"):
            with tracer.span("render", dpi=300):
                pass
        payload = tracer.to_dict()
        assert payload["children"][0]["name"] == "ingest"
        assert payload["children"][0]["children"][0]["attributes"]["dpi"] == 300

    def test_cost_tracker_enforces_a_budget(self):
        tracker = dp.CostTracker(budget=1.0)
        tracker.add(dp.Cost(amount=0.6, calls=1), backend="vlm")
        with pytest.raises(dp.BudgetExceeded):
            tracker.add(dp.Cost(amount=0.6, calls=1), backend="vlm")

    def test_cost_tracker_breaks_spend_down_by_backend(self):
        tracker = dp.CostTracker()
        tracker.add(dp.Cost(amount=0.1, calls=1), backend="tesseract")
        tracker.add(dp.Cost(amount=0.9, calls=1), backend="vlm:claude")
        assert tracker.to_dict()["by_backend"]["vlm:claude"]["amount"] == \
            pytest.approx(0.9)


class TestDeterminism:
    def test_the_same_input_gives_the_same_output(self, bill_image, bill_spans,
                                                  tmp_path):
        path = str(tmp_path / "scan.png")
        dp.save_image(degrade(bill_image, blur=1.0, skew=-2.0), path)

        def once():
            pipeline = dp.Pipeline(backend=TruthBackend({0: bill_spans}),
                                   schema=SCHEMA, client=dp.EchoClient(RESPONSE))
            return pipeline.run(path)

        first, second = once(), once()
        assert first.to_dict()["fields"] == second.to_dict()["fields"]
        assert [h.op for h in first.document.pages[0].history] == \
               [h.op for h in second.document.pages[0].history]
