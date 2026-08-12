"""Layer 1: the intermediate representation.

The IR is the product, so these tests are deliberately picky about invariants
other layers rely on: bbox arithmetic, lazy rastering, history, and round-trip
serialisation.
"""

import json

import pytest

import docpipe as dp

np = pytest.importorskip("numpy")


# -- BBox --------------------------------------------------------------------

class TestBBox:
    def test_normalises_inverted_rectangles(self):
        # Several detectors emit quad vertices in arbitrary order.
        box = dp.BBox(0, 100.0, 200.0, 20.0, 40.0)
        assert box.x0 == 20.0 and box.x1 == 100.0
        assert box.y0 == 40.0 and box.y1 == 200.0
        assert box.width == 80.0 and box.height == 160.0

    def test_geometry(self):
        box = dp.BBox(1, 10, 20, 30, 60)
        assert box.width == 20 and box.height == 40
        assert box.area == 800
        assert box.center == (20.0, 40.0)

    def test_union_and_intersection(self):
        a = dp.BBox(0, 0, 0, 10, 10)
        b = dp.BBox(0, 5, 5, 15, 15)
        assert a.union(b) == dp.BBox(0, 0, 0, 15, 15)
        assert a.intersection(b) == dp.BBox(0, 5, 5, 10, 10)
        assert a.iou(b) == pytest.approx(25.0 / 175.0)

    def test_no_overlap_returns_none(self):
        a = dp.BBox(0, 0, 0, 10, 10)
        assert a.intersection(dp.BBox(0, 20, 20, 30, 30)) is None
        assert a.iou(dp.BBox(0, 20, 20, 30, 30)) == 0.0

    def test_cross_page_boxes_never_intersect_and_cannot_union(self):
        a = dp.BBox(0, 0, 0, 10, 10)
        b = dp.BBox(1, 0, 0, 10, 10)
        assert a.intersection(b) is None
        with pytest.raises(ValueError):
            a.union(b)

    def test_pixel_round_trip(self):
        box = dp.BBox.from_pixels(3, 300, 600, 900, 1200, dpi=300)
        assert box.x0 == pytest.approx(72.0)
        assert box.to_pixels(300) == (300, 600, 900, 1200)

    def test_from_quad_takes_axis_aligned_hull(self):
        quad = [[10, 12], [90, 8], [92, 40], [12, 44]]
        box = dp.BBox.from_quad(0, quad)
        assert (box.x0, box.y0, box.x1, box.y1) == (10, 8, 92, 44)

    def test_contains_and_expand(self):
        outer = dp.BBox(0, 0, 0, 100, 100)
        inner = dp.BBox(0, 10, 10, 20, 20)
        assert outer.contains(inner)
        assert not inner.contains(outer)
        assert inner.expand(5) == dp.BBox(0, 5, 5, 25, 25)

    def test_is_frozen(self):
        box = dp.BBox(0, 0, 0, 1, 1)
        with pytest.raises(Exception):
            box.x0 = 5  # type: ignore[misc]  # frozen on purpose -- that is the test

    def test_dict_round_trip(self):
        box = dp.BBox(2, 1.5, 2.5, 3.5, 4.5)
        assert dp.BBox.from_dict(box.to_dict()) == box

    def test_merge_bboxes_groups_per_page(self):
        boxes = [dp.BBox(0, 0, 0, 5, 5), dp.BBox(0, 10, 10, 15, 15),
                 dp.BBox(1, 2, 2, 4, 4)]
        merged = dp.merge_bboxes(boxes)
        assert len(merged) == 2
        assert merged[0] == dp.BBox(0, 0, 0, 15, 15)
        assert merged[1] == dp.BBox(1, 2, 2, 4, 4)


# -- TextSpan / PageQuality --------------------------------------------------

class TestTextSpan:
    def test_missing_confidence_stays_none(self):
        # "no evidence" must never be encoded as 0.0 or 1.0.
        span = dp.TextSpan("hello", dp.BBox(0, 0, 0, 10, 10))
        assert span.confidence is None
        assert "confidence" not in span.to_dict()

    def test_round_trip(self):
        span = dp.TextSpan("hello", dp.BBox(0, 0, 0, 10, 10), source="tesseract",
                           confidence=0.87, script="Latn", line=3, meta={"k": 1})
        restored = dp.TextSpan.from_dict(span.to_dict())
        assert restored.text == span.text
        assert restored.confidence == pytest.approx(0.87)
        assert restored.script == "Latn"
        assert restored.line == 3


class TestPageQuality:
    def test_score_is_bounded_and_monotone_in_blur(self):
        good = dp.PageQuality(blur=0.9, contrast=0.9, effective_dpi=300,
                              noise=0.0, illumination=1.0)
        bad = dp.PageQuality(blur=0.1, contrast=0.9, effective_dpi=300,
                             noise=0.0, illumination=1.0)
        assert 0.0 <= bad.score < good.score <= 1.0

    def test_round_trip(self):
        q = dp.PageQuality(blur=0.5, skew_deg=1.25, contrast=0.4, ink_coverage=0.07,
                           effective_dpi=210, verdict=dp.Verdict.DEGRADED)
        restored = dp.PageQuality.from_dict(q.to_dict())
        assert restored.verdict is dp.Verdict.DEGRADED
        assert restored.skew_deg == pytest.approx(1.25)

    def test_unreadable_is_not_readable(self):
        q = dp.PageQuality(verdict=dp.Verdict.UNREADABLE)
        assert not q.is_readable


# -- Page --------------------------------------------------------------------

class TestPageRaster:
    def test_provider_is_lazy_and_called_once_per_dpi(self):
        calls = []

        def provider(dpi):
            calls.append(dpi)
            return np.full((10, 10), 200, np.uint8)

        page = dp.Page(index=0)
        page.set_raster_provider(provider, dpi=150)
        assert calls == []                 # nothing rendered yet
        page.raster(150)
        page.raster(150)
        assert calls == [150]               # cached

    def test_different_dpi_rerenders_from_source_while_provider_exists(self):
        calls = []

        def provider(dpi):
            calls.append(dpi)
            size = int(dpi / 10)
            return np.full((size, size), 128, np.uint8)

        page = dp.Page(index=0)
        page.set_raster_provider(provider, dpi=100)
        assert page.raster(100).shape == (10, 10)
        assert page.raster(300).shape == (30, 30)
        assert calls == [100, 300]

    def test_processed_raster_is_authoritative_and_is_never_re_rendered(self):
        """Once an op has run, re-rendering from source would undo it."""
        def provider(dpi):
            return np.full((10, 10), 255, np.uint8)

        page = dp.Page(index=0)
        page.set_raster_provider(provider, dpi=100)
        page.raster(100)
        page.set_raster(np.zeros((10, 10), np.uint8), dpi=100)   # "an op ran"
        assert page._raster_provider is None
        assert page.raster(300).max() == 0        # rescaled, not re-rendered

    def test_raster_without_source_raises_a_useful_error(self):
        page = dp.Page(index=7)
        assert not page.has_raster()
        with pytest.raises(dp.DocpipeError) as exc:
            page.raster()
        assert "page 7" in str(exc.value)

    def test_release_raster_keeps_data_when_it_cannot_be_regenerated(self):
        page = dp.Page(index=0)
        page.set_raster(np.zeros((4, 4), np.uint8), dpi=100)
        page.release_raster()
        assert page._raster is not None      # no provider => must not drop it


class TestPageText:
    def _page(self):
        page = dp.Page(index=0)
        # Deliberately out of reading order.
        page.spans = [
            dp.TextSpan("world", dp.BBox(0, 60, 10, 100, 22)),
            dp.TextSpan("Hello", dp.BBox(0, 10, 10, 50, 22)),
            dp.TextSpan("second", dp.BBox(0, 10, 40, 60, 52)),
        ]
        return page

    def test_lines_are_grouped_and_ordered(self):
        assert self._page().lines() == ["Hello world", "second"]

    def test_text_joins_lines(self):
        assert self._page().text() == "Hello world\nsecond"

    def test_backend_line_ids_are_scoped_to_their_block(self):
        """PyMuPDF (and every engine with a block/line/word hierarchy) restarts
        its line counter inside each block.  Grouping on the line index alone
        welds the first line of every block into one enormous line, wrecking
        reading order and defeating provenance lookup."""
        page = dp.Page(index=0)
        page.spans = [
            dp.TextSpan("HOSPITAL", dp.BBox(0, 10, 10, 90, 22), block=0, line=0),
            dp.TextSpan("NAME", dp.BBox(0, 95, 10, 140, 22), block=0, line=0),
            dp.TextSpan("Patient", dp.BBox(0, 10, 200, 60, 212), block=1, line=0),
            dp.TextSpan("Ramesh", dp.BBox(0, 65, 200, 120, 212), block=1, line=0),
        ]
        assert page.lines() == ["HOSPITAL NAME", "Patient Ramesh"]

    def test_lines_fall_back_to_geometry_without_line_ids(self):
        page = dp.Page(index=0)
        page.spans = [
            dp.TextSpan("a", dp.BBox(0, 10, 10, 20, 22)),
            dp.TextSpan("b", dp.BBox(0, 30, 11, 40, 23)),
            dp.TextSpan("c", dp.BBox(0, 10, 60, 20, 72)),
        ]
        assert page.lines() == ["a b", "c"]

    def test_spans_in_respects_overlap_fraction(self):
        page = self._page()
        assert len(page.spans_in(dp.BBox(0, 0, 0, 120, 30))) == 2
        assert page.spans_in(dp.BBox(0, 0, 0, 20, 30)) == []

    def test_char_count(self):
        assert self._page().char_count == len("world") + len("Hello") + len("second")


class TestPageCopy:
    def test_copy_does_not_share_spans(self):
        page = dp.Page(index=0, spans=[dp.TextSpan("a", dp.BBox(0, 0, 0, 1, 1))])
        clone = page.copy()
        clone.spans[0].text = "changed"
        assert page.spans[0].text == "a"

    def test_copy_shares_the_raster_buffer(self):
        # Copying is on the hot path for every op; duplicating megabytes of
        # pixels per op would be unacceptable.
        page = dp.Page(index=0)
        page.set_raster(np.zeros((8, 8), np.uint8), dpi=100)
        assert page.copy()._raster is page._raster

    def test_history_is_recorded_and_rendered(self):
        page = dp.Page(index=0)
        page.record("deskew", {"angle": 1.5}, ms=12.0)
        assert "deskew(angle=1.5)" == page.history_summary()
        assert page.to_dict()["history"][0]["op"] == "deskew"


# -- Document ----------------------------------------------------------------

class TestDocument:
    def _doc(self):
        doc = dp.Document(source_uri="x.pdf")
        for i in range(3):
            page = dp.Page(index=i)
            page.spans = [dp.TextSpan("page%d" % i, dp.BBox(i, 0, 0, 10, 10))]
            doc.pages.append(page)
        return doc

    def test_sequence_protocol(self):
        doc = self._doc()
        assert len(doc) == 3
        assert doc[1].index == 1
        assert [p.index for p in doc] == [0, 1, 2]

    def test_page_lookup_is_by_index_attribute(self):
        doc = self._doc()
        doc.pages[0].index = 42
        assert doc.page(42) is doc.pages[0]
        with pytest.raises(KeyError):
            doc.page(0)

    def test_select_filters_pages(self):
        doc = self._doc()
        assert len(doc.select(lambda p: p.index > 0)) == 2

    def test_warn_deduplicates(self):
        doc = dp.Document()
        doc.warn("same")
        doc.warn("same")
        assert doc.warnings == ["same"]

    def test_json_round_trip(self, tmp_path):
        doc = self._doc()
        doc.meta["producer"] = "test"
        path = str(tmp_path / "doc.json")
        doc.save_json(path)
        restored = dp.Document.load_json(path)
        assert len(restored) == 3
        assert restored.text() == doc.text()
        assert restored.meta["producer"] == "test"

    def test_kinds_counts_page_types(self):
        doc = self._doc()
        doc.pages[0].kind = dp.PageKind.DIGITAL_NATIVE
        assert doc.kinds() == {"digital_native": 1, "scanned": 2}


# -- Cost --------------------------------------------------------------------

class TestCost:
    def test_addition_accumulates(self):
        total = dp.Cost(amount=1.5, input_tokens=100, calls=1) + \
                dp.Cost(amount=2.5, output_tokens=50, calls=1)
        assert total.amount == pytest.approx(4.0)
        assert total.input_tokens == 100 and total.output_tokens == 50
        assert total.calls == 2

    def test_sum_works_via_radd(self):
        total = sum([dp.Cost(amount=1.0, calls=1), dp.Cost(amount=2.0, calls=1)],
                    dp.Cost.zero())
        assert total.amount == pytest.approx(3.0)

    def test_mixing_currencies_is_refused(self):
        with pytest.raises(dp.ConfigError):
            _ = dp.Cost(currency="USD", amount=1.0) + dp.Cost(currency="INR", amount=1.0)

    def test_zero_cost_in_another_currency_is_harmless(self):
        total = dp.Cost(currency="USD", amount=1.0) + dp.Cost(currency="INR")
        assert total.currency == "USD" and total.amount == pytest.approx(1.0)


def test_to_json_handles_the_whole_object_graph():
    doc = dp.Document(source_uri="a.pdf",
                      pages=[dp.Page(index=0, spans=[
                          dp.TextSpan("x", dp.BBox(0, 0, 0, 1, 1))])])
    payload = json.loads(dp.to_json({"doc": doc, "cost": dp.Cost(amount=1.0),
                                     "kind": dp.PageKind.SCANNED}))
    assert payload["kind"] == "scanned"
    assert payload["doc"]["pages"][0]["spans"][0]["text"] == "x"
