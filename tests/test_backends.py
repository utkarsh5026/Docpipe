"""Layer 3: the backend protocol, registry, wrappers and routing.

No test here talks to a real OCR engine or a paid API.  The point is the
*plumbing* -- that backends are interchangeable, that the registry stays lazy
(constructing PaddleOCR loads hundreds of megabytes of weights), that retries
account honestly for calls a provider already billed, and that routing sends
native-text pages to the free exact path rather than to a model.
"""

import pytest

import docpipe as dp
from conftest import TruthBackend, requires_pymupdf

np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def restore_registry():
    """Each test gets the default registry back."""
    yield
    dp._register_default_backends()
    dp.registry.clear_instances()


class FlakyBackend(dp.BaseBackend):
    """Fails a fixed number of times, then succeeds."""

    name = "flaky"
    needs_raster = False

    def __init__(self, failures=2, exc=RuntimeError("provider timeout")):
        dp.BaseBackend.__init__(self)
        self.failures = failures
        self.attempts = 0
        self.exc = exc

    def supports(self, page):
        return True

    def _read(self, page):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.exc
        return dp.ReadResult(
            spans=[dp.TextSpan("ok", dp.BBox(page.index, 0, 0, 10, 10))],
            cost=dp.Cost(amount=0.01, calls=1), backend=self.name)


# -- protocol ----------------------------------------------------------------

class TestBaseBackend:
    def test_read_times_the_call_and_labels_the_spans(self, bill_page):
        backend = TruthBackend({0: bill_page[1]})
        page = dp.Page(index=0)
        result = backend.read(page)
        assert result.backend == "truth"
        assert result.latency_ms >= 0.0
        assert all(s.source == "truth" for s in result.spans)

    def test_read_tags_each_span_with_its_script(self, bill_page):
        result = TruthBackend({0: bill_page[1]}).read(dp.Page(index=0))
        assert result.spans[0].script == "Latn"

    def test_supports_is_false_without_a_raster_when_one_is_needed(self):
        class NeedsPixels(dp.BaseBackend):
            name = "pixels"

            def _read(self, page):
                return dp.ReadResult()

        assert not NeedsPixels().supports(dp.Page(index=0))

    def test_read_result_text_joins_spans(self):
        result = dp.ReadResult(spans=[
            dp.TextSpan("hello", dp.BBox(0, 0, 0, 1, 1)),
            dp.TextSpan("world", dp.BBox(0, 2, 0, 3, 1))])
        assert result.text == "hello world"


# -- registry ----------------------------------------------------------------

class TestRegistry:
    def test_factories_are_not_called_until_requested(self):
        constructed = []

        class Expensive(dp.BaseBackend):
            name = "expensive"

            def __init__(self):
                dp.BaseBackend.__init__(self)
                constructed.append(1)

            def _read(self, page):
                return dp.ReadResult()

        dp.registry.register("expensive", Expensive, replace=True)
        assert constructed == []
        dp.registry.get("expensive")
        dp.registry.get("expensive")
        assert constructed == [1]      # constructed once, cached

    def test_instances_may_be_registered_directly(self, bill_page):
        instance = TruthBackend({0: bill_page[1]})
        dp.registry.register("t", instance, replace=True)
        assert dp.registry.get("t") is instance

    def test_duplicate_registration_needs_replace(self, bill_page):
        dp.registry.register("dupe", TruthBackend({0: []}), replace=True)
        with pytest.raises(dp.ConfigError):
            dp.registry.register("dupe", TruthBackend({0: []}))

    def test_unknown_name_lists_what_is_registered(self):
        with pytest.raises(dp.ConfigError) as exc:
            dp.registry.get("clairvoyance")
        assert "pymupdf" in str(exc.value)

    def test_contains_and_names(self):
        assert "tesseract" in dp.registry
        assert "pymupdf" in dp.registry.names()

    def test_available_excludes_uninstalled_engines(self):
        available = dp.registry.available()
        if not dp.have("paddleocr"):
            assert "paddle" not in available

    def test_third_party_backends_need_no_core_change(self, bill_page):
        """If a consumer ever has to fork this file to add an engine, that is
        a defect here."""
        dp.registry.register("acme-ocr", TruthBackend({0: bill_page[1]}), replace=True)
        assert "acme-ocr" in dp.registry.available()


# -- native text layer -------------------------------------------------------

@requires_pymupdf
class TestPyMuPDFTextLayer:
    def test_reads_the_embedded_layer_for_free(self, native_pdf):
        doc = dp.ingest(native_pdf)
        result = dp.PyMuPDFTextLayer().read(doc.pages[0])
        assert "SUNRISE" in result.text
        assert result.cost.amount == 0.0

    def test_does_not_support_a_scanned_page(self, scanned_pdf):
        doc = dp.ingest(scanned_pdf)
        assert not dp.PyMuPDFTextLayer().supports(doc.pages[0])


# -- wrappers ----------------------------------------------------------------

class TestCachingBackend:
    def test_second_read_of_identical_pixels_is_a_hit(self, bill_image, isolated_cache):
        inner = TruthBackend({0: [dp.TextSpan("x", dp.BBox(0, 0, 0, 1, 1))]})
        cached = dp.CachingBackend(inner, cache=isolated_cache)
        page = dp.page_from_image(bill_image, dpi=150)
        cached.read(page)
        cached.read(page)
        assert (cached.hits, cached.misses) == (1, 1)
        assert inner.calls == 1

    def test_changed_pixels_miss(self, bill_image, isolated_cache):
        inner = TruthBackend({0: [dp.TextSpan("x", dp.BBox(0, 0, 0, 1, 1))]})
        cached = dp.CachingBackend(inner, cache=isolated_cache)
        cached.read(dp.page_from_image(bill_image, dpi=150))
        cached.read(dp.page_from_image(dp.rotate_image(bill_image, -5.0,
                                                       border_value=255), dpi=150))
        assert cached.misses == 2

    def test_cached_reads_report_no_cost(self, bill_image, isolated_cache):
        class Pricey(TruthBackend):
            def _read(self, page):
                return dp.ReadResult(spans=[dp.TextSpan("v", dp.BBox(0, 0, 0, 1, 1))],
                                     cost=dp.Cost(amount=0.5, calls=1))

        cached = dp.CachingBackend(Pricey({0: []}), cache=isolated_cache)
        page = dp.page_from_image(bill_image, dpi=150)
        assert cached.read(page).cost.amount == pytest.approx(0.5)
        assert cached.read(page).cost.amount == 0.0

    def test_cache_survives_a_new_cache_object_on_the_same_directory(self, bill_image,
                                                                    tmp_path):
        inner = TruthBackend({0: [dp.TextSpan("x", dp.BBox(0, 0, 0, 1, 1))]})
        page = dp.page_from_image(bill_image, dpi=150)
        directory = str(tmp_path / "shared")
        dp.CachingBackend(inner, cache=dp.DiskCache(directory)).read(page)
        second = dp.CachingBackend(inner, cache=dp.DiskCache(directory))
        second.read(page)
        assert second.hits == 1


class TestRetryingBackend:
    def test_succeeds_after_transient_failures(self, monkeypatch):
        monkeypatch.setattr(dp.time, "sleep", lambda s: None)
        inner = FlakyBackend(failures=2)
        result = dp.RetryingBackend(inner, attempts=3, base_delay=0).read(dp.Page(index=0))
        assert result.text == "ok"
        assert inner.attempts == 3

    def test_failed_attempts_are_reported_as_potentially_billed(self, monkeypatch):
        """A provider that times out after generating tokens has already
        charged for them; reporting the retry as free hides a doubled bill."""
        monkeypatch.setattr(dp.time, "sleep", lambda s: None)
        result = dp.RetryingBackend(FlakyBackend(failures=2), attempts=3,
                                    base_delay=0).read(dp.Page(index=0))
        assert result.cost.wasted_calls == 2
        assert any("billed" in w for w in result.warnings)

    def test_gives_up_after_the_attempt_budget(self, monkeypatch):
        monkeypatch.setattr(dp.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError):
            dp.RetryingBackend(FlakyBackend(failures=9), attempts=3,
                               base_delay=0).read(dp.Page(index=0))

    def test_non_retryable_errors_fail_immediately(self, monkeypatch):
        """Hammering a bad API key three times just bills three times."""
        monkeypatch.setattr(dp.time, "sleep", lambda s: None)
        inner = FlakyBackend(failures=9, exc=dp.ConfigError("bad api key"))
        with pytest.raises(dp.ConfigError):
            dp.RetryingBackend(inner, attempts=5, base_delay=0).read(dp.Page(index=0))
        assert inner.attempts == 1


class TestEnsembleBackend:
    def _pair(self, spans, corrupt_every):
        return (TruthBackend({0: spans}, name="a"),
                TruthBackend({0: spans}, error_every=corrupt_every, name="b"))

    def test_identical_reads_agree_completely(self, bill_page):
        primary, secondary = self._pair(bill_page[1], 0)
        result = dp.EnsembleBackend(primary, secondary).read(dp.Page(index=0))
        assert result.spans[0].meta["cross_read_agreement"] == pytest.approx(1.0)

    def test_agreement_falls_monotonically_with_the_error_rate(self, bill_page):
        def agreement(error_every):
            primary, secondary = self._pair(bill_page[1], error_every)
            result = dp.EnsembleBackend(primary, secondary).read(dp.Page(index=0))
            return result.spans[0].meta["cross_read_agreement"]

        scores = [agreement(n) for n in (0, 8, 4, 2, 1)]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0)
        assert scores[-1] < 0.9

    def test_a_backend_that_reads_nothing_scores_zero_agreement(self, bill_page):
        primary = TruthBackend({0: bill_page[1]}, name="a")
        silent = TruthBackend({0: []}, name="b")
        result = dp.EnsembleBackend(primary, silent).read(dp.Page(index=0))
        assert result.spans[0].meta["cross_read_agreement"] == 0.0

    def test_costs_are_summed(self, bill_page):
        primary, secondary = self._pair(bill_page[1], 0)
        assert dp.EnsembleBackend(primary, secondary).read(dp.Page(index=0)).cost.calls == 2

    def test_primary_geometry_is_the_one_returned(self, bill_page):
        primary, secondary = self._pair(bill_page[1], 3)
        result = dp.EnsembleBackend(primary, secondary).read(dp.Page(index=0))
        assert [s.bbox for s in result.spans] == [s.bbox for s in bill_page[1]]


# -- routing -----------------------------------------------------------------

class TestDefaultRouter:
    def test_native_text_never_pays_a_model(self):
        page = dp.Page(index=0, kind=dp.PageKind.DIGITAL_NATIVE,
                       spans=[dp.TextSpan("x" * 80, dp.BBox(0, 0, 0, 10, 10))])
        assert dp.default_router(page) == "pymupdf"

    def test_blank_pages_are_skipped(self):
        assert dp.default_router(dp.Page(index=0, kind=dp.PageKind.BLANK)) is None

    def test_returns_none_when_nothing_is_installed(self, monkeypatch):
        monkeypatch.setattr(dp.registry, "available", lambda: [])
        page = dp.Page(index=0, kind=dp.PageKind.SCANNED)
        page.quality.ink_coverage = 0.1
        assert dp.default_router(page) is None

    def test_degraded_pages_prefer_a_vision_model(self, monkeypatch):
        monkeypatch.setattr(dp.registry, "available",
                            lambda: ["tesseract", "vlm:claude"])
        page = dp.Page(index=0, kind=dp.PageKind.SCANNED)
        page.quality = dp.PageQuality(verdict=dp.Verdict.UNREADABLE, ink_coverage=0.1)
        assert dp.default_router(page) == "vlm:claude"

    def test_clean_pages_use_a_local_engine(self, monkeypatch):
        monkeypatch.setattr(dp.registry, "available",
                            lambda: ["tesseract", "vlm:claude"])
        page = dp.Page(index=0, kind=dp.PageKind.SCANNED)
        page.quality = dp.PageQuality(blur=0.9, contrast=0.9, effective_dpi=300,
                                      ink_coverage=0.1, verdict=dp.Verdict.CLEAN)
        assert dp.default_router(page) == "tesseract"

    def test_handwriting_forces_a_vision_model(self, monkeypatch):
        monkeypatch.setattr(dp.registry, "available",
                            lambda: ["tesseract", "vlm:claude"])
        page = dp.Page(index=0, kind=dp.PageKind.SCANNED)
        page.quality = dp.PageQuality(blur=0.9, contrast=0.9, effective_dpi=300,
                                      ink_coverage=0.1)
        page.layout.regions = [dp.LayoutRegion(dp.RegionKind.HANDWRITING,
                                               dp.BBox(0, 0, 0, 10, 10))]
        assert dp.default_router(page) == "vlm:claude"

    def test_tabular_pages_prefer_a_table_capable_engine(self, monkeypatch):
        monkeypatch.setattr(dp.registry, "available",
                            lambda: ["tesseract", "paddle"])
        page = dp.Page(index=0, kind=dp.PageKind.SCANNED)
        page.quality = dp.PageQuality(blur=0.9, contrast=0.9, effective_dpi=300,
                                      ink_coverage=0.1)
        page.layout.regions = [dp.LayoutRegion(dp.RegionKind.TABLE,
                                               dp.BBox(0, 0, 0, 10, 10))]
        assert dp.default_router(page) == "paddle"


class TestRuleRouter:
    def test_first_matching_rule_wins(self):
        router = dp.RuleRouter([
            (lambda p: p.index == 0, "first"),
            (lambda p: True, "rest"),
        ], fallback="none")
        assert router(dp.Page(index=0)) == "first"
        assert router(dp.Page(index=5)) == "rest"

    def test_fallback_applies_when_nothing_matches(self):
        assert dp.RuleRouter([], fallback="x")(dp.Page(index=0)) == "x"

    def test_a_raising_rule_is_skipped_not_fatal(self):
        def broken(page):
            raise ValueError("bad rule")

        router = dp.RuleRouter([(broken, "a"), (lambda p: True, "b")])
        assert router(dp.Page(index=0)) == "b"


class TestBudgetRouter:
    def test_downgrades_once_the_budget_is_spent(self, bill_image):
        class Pricey(dp.BaseBackend):
            name = "pricey"
            needs_raster = False

            def supports(self, page):
                return True

            def estimate_cost(self, page):
                return dp.Cost(amount=1.0)

            def _read(self, page):
                return dp.ReadResult()

        dp.registry.register("pricey", Pricey, replace=True)
        dp.registry.register("cheap", lambda: TruthBackend({0: []}, name="cheap"),
                             replace=True)
        router = dp.BudgetRouter(lambda p: "pricey", budget=1.5, cheap_backend="cheap")
        page = dp.page_from_image(bill_image, dpi=150)
        assert router(page) == "pricey"
        router.record(dp.Cost(amount=1.0))
        assert router(page) == "cheap"
        assert router.downgrades == 1

    def test_remaining_tracks_spend(self):
        router = dp.BudgetRouter(lambda p: None, budget=2.0)
        router.record(dp.Cost(amount=0.5))
        assert router.remaining() == pytest.approx(1.5)


# -- read() ------------------------------------------------------------------

class TestRead:
    def test_reads_every_page_and_replaces_spans(self, bill_image, bill_spans):
        doc = dp.document_from_images([bill_image], dpi=150)
        backend = TruthBackend({0: bill_spans})
        out = dp.read(doc, backend=backend)
        assert "SUNRISE" in out.text()
        assert out.meta["read_backends"] == {0: "truth"}

    def test_does_not_mutate_the_input(self, bill_image, bill_spans):
        doc = dp.document_from_images([bill_image], dpi=150)
        dp.read(doc, backend=TruthBackend({0: bill_spans}))
        assert doc.pages[0].spans == []

    def test_accumulates_cost(self, bill_image, bill_spans):
        doc = dp.document_from_images([bill_image, bill_image], dpi=150)
        backend = TruthBackend({0: bill_spans, 1: bill_spans})
        out = dp.read(doc, backend=backend)
        assert out.meta["read_cost"]["calls"] == 2

    def test_unsupported_page_produces_a_warning_not_an_exception(self, bill_image):
        doc = dp.document_from_images([bill_image], dpi=150)
        out = dp.read(doc, backend=TruthBackend({7: []}))
        assert any("cannot read" in w for w in out.warnings)

    def test_a_raising_backend_is_isolated_to_its_page(self, bill_image, bill_spans):
        class Exploding(dp.BaseBackend):
            name = "boom"
            needs_raster = False

            def supports(self, page):
                return True

            def _read(self, page):
                if page.index == 0:
                    raise RuntimeError("kaboom")
                return dp.ReadResult(spans=[dp.TextSpan("fine", dp.BBox(1, 0, 0, 1, 1))])

        doc = dp.document_from_images([bill_image, bill_image], dpi=150)
        out = dp.read(doc, backend=Exploding())
        assert any("kaboom" in w for w in out.warnings)
        assert out.pages[1].text() == "fine"

    def test_budget_stops_the_run(self, bill_image):
        class Pricey(dp.BaseBackend):
            name = "pricey"
            needs_raster = False

            def supports(self, page):
                return True

            def _read(self, page):
                return dp.ReadResult(cost=dp.Cost(amount=5.0, calls=1))

        doc = dp.document_from_images([bill_image] * 3, dpi=150)
        with pytest.raises(dp.BudgetExceeded):
            dp.read(doc, backend=Pricey(), budget=6.0)

    def test_on_page_callback_fires_per_page(self, bill_image, bill_spans):
        seen = []
        doc = dp.document_from_images([bill_image], dpi=150)
        dp.read(doc, backend=TruthBackend({0: bill_spans}),
                on_page=lambda p, r: seen.append(p.index))
        assert seen == [0]

    def test_backend_may_be_named(self, native_pdf):
        doc = dp.ingest(native_pdf)
        assert "SUNRISE" in dp.read(doc, backend="pymupdf").text()


# -- cost model --------------------------------------------------------------

class TestPricing:
    def test_unpriced_models_report_tokens_but_no_amount(self):
        """A wrong price is worse than a missing one: it produces a plausible
        budget report that nobody rechecks."""
        dp.PRICING.pop("mystery-model", None)
        cost = dp.price_tokens("mystery-model", 1000, 500)
        assert cost.amount == 0.0
        assert cost.input_tokens == 1000 and cost.output_tokens == 500

    def test_registered_prices_are_applied(self):
        dp.set_pricing("test-model", 3.0, 15.0)
        try:
            cost = dp.price_tokens("test-model", 1_000_000, 1_000_000)
            assert cost.amount == pytest.approx(18.0)
        finally:
            dp.PRICING.pop("test-model", None)

    def test_image_token_estimate_scales_with_area(self, bill_image):
        small = dp.resize_image(bill_image, scale=0.5)
        assert dp.estimate_image_tokens(bill_image) > dp.estimate_image_tokens(small) * 3


class TestVisionBackendPlumbing:
    class Fake(dp.VisionBackend):
        name = "vlm:fake"

        def __init__(self, text, **kw):
            dp.VisionBackend.__init__(self, model="fake-1", **kw)
            self.text = text
            self.sent = []

        def _call_model(self, image_bytes, media_type, prompt):
            self.sent.append((len(image_bytes), media_type))
            return self.text, dp.Cost(amount=0.02, input_tokens=900,
                                      output_tokens=120, calls=1)

    def test_transcription_becomes_one_span_per_line(self, bill_image):
        backend = self.Fake("line one\nline two\n\nline three")
        result = backend.read(dp.page_from_image(bill_image, dpi=150))
        assert [s.text for s in result.spans] == ["line one", "line two", "line three"]

    def test_spans_are_marked_as_approximate(self, bill_image):
        """Fabricated coordinates would point a reviewer at the wrong place;
        honest approximation is better than invented precision."""
        result = self.Fake("only line").read(dp.page_from_image(bill_image, dpi=150))
        assert result.spans[0].meta["approximate_bbox"] is True

    def test_illegible_markers_raise_a_warning(self, bill_image):
        result = self.Fake("total [illegible] rupees").read(
            dp.page_from_image(bill_image, dpi=150))
        assert any("illegible" in w for w in result.warnings)

    def test_oversized_pages_are_downscaled_before_sending(self, bill_image):
        backend = self.Fake("x")
        backend.max_side_px = 400
        page = dp.page_from_image(bill_image, dpi=150)
        assert max(dp.image_shape(backend._image_for(page))) == 400

    def test_cost_is_reported_from_the_call(self, bill_image):
        result = self.Fake("x").read(dp.page_from_image(bill_image, dpi=150))
        assert result.cost.input_tokens == 900
