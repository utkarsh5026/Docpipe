"""Layer 2: composable preprocessing ops and adaptive policies.

Two things are asserted throughout, beyond "the op does something":

* ops never mutate their input page (A/B-ing two pipelines over one ingested
  document has to be safe);
* geometric ops carry span coordinates through the transform rather than
  silently invalidating provenance.
"""

import pytest

import docpipe as dp
from conftest import degrade

np = pytest.importorskip("numpy")


@pytest.fixture
def page(bill_image):
    p = dp.page_from_image(bill_image, dpi=150)
    dp.measure_page(p)
    return p


# -- op machinery ------------------------------------------------------------

class TestOpMachinery:
    def test_op_returns_a_new_page_and_leaves_the_original_alone(self, page):
        before = page.pixel_size
        out = dp.crop_to_content()(page)
        assert out is not page
        assert page.pixel_size == before
        assert page.history == [] or page.history[-1].op != "crop_to_content"

    def test_history_records_op_and_parameters(self, page):
        out = dp.binarize(method="otsu")(page)
        record = out.history[-1]
        assert record.op == "binarize"
        assert record.params["method"] == "otsu"
        assert record.ms >= 0.0

    def test_compose_applies_in_order(self, page):
        pipeline = dp.compose(dp.to_grayscale(), dp.binarize(), dp.pad(margin_px=4))
        out = pipeline(page)
        assert [h.op for h in out.history[-3:]] == ["to_grayscale", "binarize", "pad"]

    def test_compose_drops_none_entries(self, page):
        assert len(dp.compose(dp.to_grayscale(), None, dp.binarize()).ops) == 2

    def test_ops_serialise_and_rebuild(self):
        original = dp.binarize(method="sauvola", window=41)
        rebuilt = dp.op_from_dict(original.to_dict())
        assert rebuilt.name == "binarize"
        assert rebuilt.params["window"] == 41

    def test_composite_serialises_and_rebuilds(self, page):
        original = dp.compose(dp.to_grayscale(), dp.unsharp(amount=1.5))
        rebuilt = dp.op_from_dict(original.to_dict())
        assert [o.name for o in rebuilt.ops] == ["to_grayscale", "unsharp"]
        assert rebuilt(page).history[-1].params["amount"] == 1.5

    def test_unknown_op_name_is_reported_with_the_known_set(self):
        with pytest.raises(dp.ConfigError) as exc:
            dp.op_from_dict({"op": "enhance_zoom"})
        assert "deskew" in str(exc.value)

    def test_factory_rejects_unknown_keywords(self):
        with pytest.raises(TypeError):
            dp.binarize(methodd="otsu")

    def test_factory_accepts_positional_arguments(self):
        assert dp.denoise("medium").params["strength"] == "medium"

    def test_op_on_a_page_without_a_raster_is_skipped_not_fatal(self):
        empty = dp.Page(index=0)
        out = dp.binarize()(empty)
        assert "skipped" in out.history[-1].note


# -- individual ops ----------------------------------------------------------

class TestToneOps:
    def test_to_grayscale_collapses_channels(self):
        page = dp.page_from_image(np.zeros((20, 20, 3), np.uint8), dpi=150)
        assert dp.to_grayscale()(page).raster().ndim == 2

    def test_invert_if_dark_flips_a_negative_scan(self, bill_image):
        page = dp.page_from_image(255 - np.asarray(bill_image), dpi=150)
        out = dp.invert_if_dark()(page)
        assert out.meta.get("inverted") is True
        assert out.raster().mean() > 128

    def test_invert_if_dark_leaves_a_normal_page_alone(self, page):
        assert dp.invert_if_dark()(page).meta.get("inverted") is None

    def test_autocontrast_expands_a_faded_page(self, bill_image, both_image_backends):
        faded = (np.asarray(bill_image).astype(float) * 0.3 + 170).astype(np.uint8)
        out = dp.autocontrast()(dp.page_from_image(faded, dpi=150))
        assert dp.measure_contrast(out.raster()) > dp.measure_contrast(faded)

    def test_gamma_darkens_above_one(self, page):
        assert dp.gamma(2.0)(page).raster().mean() < page.raster().mean()

    def test_gamma_of_one_is_a_no_op(self, page):
        assert np.array_equal(dp.gamma(1.0)(page).raster(), page.raster())

    def test_clahe_runs_in_both_backends(self, page, both_image_backends):
        assert dp.clahe()(page).raster().shape == page.raster().shape

    def test_normalize_illumination_flattens_a_shadow(self, bill_image,
                                                     both_image_backends):
        shadowed = degrade(bill_image, shadow=0.55)
        out = dp.normalize_illumination()(dp.page_from_image(shadowed, dpi=150))
        assert dp.measure_illumination(out.raster()) > dp.measure_illumination(shadowed)

    def test_remove_shadow_keeps_the_text(self, bill_image):
        shadowed = degrade(bill_image, shadow=0.5)
        out = dp.remove_shadow()(dp.page_from_image(shadowed, dpi=150))
        assert dp.measure_ink_coverage(out.raster()) > 0.005


class TestGeometryOps:
    def test_deskew_straightens_and_zeroes_the_measurement(self, bill_image):
        page = dp.page_from_image(dp.rotate_image(bill_image, -4.0, border_value=255),
                                  dpi=150)
        dp.measure_page(page)
        assert page.quality.skew_deg == pytest.approx(4.0, abs=0.3)
        out = dp.deskew()(page)
        assert out.quality.skew_deg == 0.0
        assert abs(dp.estimate_skew(out.raster())) < 0.4

    def test_deskew_declines_a_negligible_angle(self, page):
        page.quality.skew_deg = 0.15
        assert np.array_equal(dp.deskew(min_angle=0.4)(page).raster(), page.raster())

    def test_deskew_declines_beyond_max_angle(self, page):
        page.quality.skew_deg = 40.0
        assert np.array_equal(dp.deskew(max_angle=15.0)(page).raster(), page.raster())

    def test_deskew_carries_span_coordinates_through_the_rotation(self, bill_image):
        """Provenance is unrecoverable once dropped, so it must be transformed."""
        page = dp.page_from_image(bill_image, dpi=150)
        marker = dp.BBox(0, 100.0, 200.0, 160.0, 214.0)
        page.spans = [dp.TextSpan("total", marker)]
        out = dp.deskew(angle=5.0)(page)
        assert len(out.spans) == 1
        moved = out.spans[0].bbox
        assert moved != marker
        # A 5-degree rotation about the centre moves it, but not to another galaxy.
        assert abs(moved.x0 - marker.x0) < 200 and abs(moved.y0 - marker.y0) < 200

    def test_rotate_by_ninety_swaps_the_axes(self, page):
        out = dp.rotate(degrees=90)(page)
        assert out.pixel_size == (page.pixel_size[1], page.pixel_size[0])
        assert out.rotation == 90

    def test_rotate_four_times_is_the_identity(self, page):
        out = page
        for _ in range(4):
            out = dp.rotate(degrees=90)(out)
        assert np.array_equal(out.raster(), page.raster())
        assert out.rotation == 0

    def test_ensure_dpi_upsamples_a_low_resolution_page(self, bill_image):
        small = dp.resize_image(bill_image, scale=0.4)
        page = dp.page_from_image(small, dpi=120)
        dp.measure_page(page)
        out = dp.ensure_dpi(300)(page)
        assert out.pixel_size[0] > page.pixel_size[0]
        assert out.raster_dpi > page.raster_dpi

    def test_ensure_dpi_leaves_a_high_resolution_page_alone(self, page):
        page.quality.effective_dpi = 400
        assert np.array_equal(dp.ensure_dpi(300)(page).raster(), page.raster())

    def test_ensure_dpi_respects_the_ceiling(self, bill_image):
        page = dp.page_from_image(dp.resize_image(bill_image, scale=0.2), dpi=60)
        page.quality.effective_dpi = 60
        out = dp.ensure_dpi(min_dpi=300, max_dpi=150)(page)
        assert out.raster_dpi <= 150

    def test_resize_max_side_caps_the_longest_edge(self, page):
        out = dp.resize_max_side(400)(page)
        assert max(out.pixel_size) == 400

    def test_crop_to_content_trims_margins_and_shifts_spans(self, bill_image):
        padded = np.pad(np.asarray(bill_image), 200, mode="constant", constant_values=255)
        page = dp.page_from_image(padded, dpi=150)
        page.spans = [dp.TextSpan("x", dp.BBox(0, 120.0, 120.0, 140.0, 132.0))]
        out = dp.crop_to_content(margin_pt=3.0)(page)
        assert out.pixel_size[0] < page.pixel_size[0]
        assert out.spans[0].bbox.x0 < 120.0        # translated with the crop
        assert out.width_pt < page.width_pt        # geometry re-synced

    def test_crop_refuses_when_content_is_implausibly_small(self):
        img = np.full((400, 400), 255, np.uint8)
        img[10:14, 10:14] = 0        # a speck
        page = dp.page_from_image(img, dpi=150)
        out = dp.crop_to_content(min_keep=0.5)(page)
        assert out.pixel_size == page.pixel_size
        assert "crop_refused" in out.meta

    def test_remove_border_trims_a_dark_scan_frame(self, bill_image):
        framed = np.asarray(bill_image).copy()
        framed[:40, :] = 10
        framed[-40:, :] = 10
        page = dp.page_from_image(framed, dpi=150)
        out = dp.remove_border()(page)
        assert out.pixel_size[1] < page.pixel_size[1]

    def test_pad_grows_the_canvas_and_shifts_spans(self, page):
        page.spans = [dp.TextSpan("x", dp.BBox(0, 10.0, 10.0, 20.0, 20.0))]
        out = dp.pad(margin_px=30)(page)
        assert out.pixel_size[0] == page.pixel_size[0] + 60
        assert out.spans[0].bbox.x0 > 10.0

    @pytest.mark.skipif(not dp.have("cv2"), reason="needs OpenCV")
    def test_perspective_correct_is_a_no_op_without_a_convincing_quad(self, page):
        out = dp.perspective_correct()(page)
        assert out.pixel_size == page.pixel_size


class TestNoiseOps:
    @pytest.mark.parametrize("strength", ["light", "medium", "aggressive"])
    def test_denoise_strengths(self, bill_image, strength):
        noisy = dp.page_from_image(degrade(bill_image, noise=28.0), dpi=150)
        assert dp.measure_noise(dp.denoise(strength)(noisy).raster()) < \
               dp.measure_noise(noisy.raster())

    def test_denoise_rejects_an_unknown_strength(self, page):
        with pytest.raises(dp.ConfigError):
            dp.denoise("nuclear")(page)

    def test_denoise_rejects_an_unknown_method(self, page):
        with pytest.raises(dp.ConfigError):
            dp.denoise(method="wavelet")(page)

    def test_despeckle_removes_isolated_dots(self, bill_image, both_image_backends):
        speckled = np.asarray(bill_image).copy()
        rng = np.random.RandomState(7)
        ys, xs = rng.randint(0, speckled.shape[0], 500), rng.randint(0, speckled.shape[1], 500)
        speckled[ys, xs] = 0
        page = dp.page_from_image(speckled, dpi=150)
        out = dp.despeckle(min_area_px=6)(page)
        assert dp.measure_ink_coverage(out.raster()) < dp.measure_ink_coverage(speckled)

    def test_unsharp_recovers_gradient_energy(self, bill_image):
        soft = dp.page_from_image(degrade(bill_image, blur=1.6), dpi=150)
        sharpened = dp.unsharp(amount=1.5)(soft)
        assert dp.measure_blur(sharpened.raster()) > dp.measure_blur(soft.raster())

    def test_unsharp_threshold_suppresses_flat_areas(self, page):
        out = dp.unsharp(amount=2.0, threshold=200)(page)
        assert float(np.mean(np.abs(out.raster().astype(float)
                                    - page.raster().astype(float)))) < 3.0

    @pytest.mark.parametrize("operation", ["open", "close", "erode", "dilate"])
    def test_morphology_operations(self, page, operation, both_image_backends):
        assert dp.morphology(operation, ksize=3)(page).raster().shape == \
               page.raster().shape

    def test_morphology_rejects_an_unknown_operation(self, page):
        with pytest.raises(dp.ConfigError):
            dp.morphology("swirl")(page)


class TestBinarization:
    @pytest.mark.parametrize("method",
                             ["otsu", "adaptive", "sauvola", "niblack", "wolf",
                              "nick", "bradley"])
    def test_every_method_produces_a_two_level_image(self, bill_image, method,
                                                     both_image_backends):
        out = dp.binarize_array(bill_image, method=method)
        assert set(np.unique(out)).issubset({0, 255})
        coverage = float((out == 0).mean())
        assert 0.001 < coverage < 0.6, "%s produced %.4f ink" % (method, coverage)

    def test_sauvola_beats_otsu_under_a_shadow_gradient(self, bill_image):
        """The reason the default is a local method: a global threshold turns
        the shadowed part of the page into a solid block of 'ink'."""
        shadowed = degrade(bill_image, shadow=0.75)
        reference = dp.ink_mask(bill_image)
        otsu = dp.binarize_array(shadowed, method="otsu") == 0
        sauvola = dp.binarize_array(shadowed, method="sauvola", window=41) == 0

        def iou(mask):
            # Recall alone is useless here: marking the whole page black scores
            # a perfect 1.0. Overlap punishes that.
            return float((mask & reference).sum()) / float((mask | reference).sum())

        assert iou(sauvola) > iou(otsu) * 1.5

    def test_niblack_does_not_black_out_blank_paper(self, bill_image):
        """Its notorious failure mode, guarded by min_std."""
        assert float((dp.binarize_array(bill_image, method="niblack") == 0).mean()) < 0.4
        unguarded = dp.binarize_array(bill_image, method="niblack", min_std=0.0)
        assert float((unguarded == 0).mean()) > 0.5

    def test_unknown_method_is_rejected(self, bill_image):
        with pytest.raises(dp.ConfigError):
            dp.binarize_array(bill_image, method="magic")

    def test_binarize_op_records_its_method(self, page):
        assert dp.binarize(method="wolf")(page).history[-1].params["method"] == "wolf"


class TestLinesAndStamps:
    def test_remove_lines_erases_a_long_rule(self, both_image_backends):
        img = np.full((300, 600), 255, np.uint8)
        img[150:153, 20:580] = 0            # a horizontal rule
        img[60:90, 100:130] = 0             # a glyph-sized blob
        page = dp.page_from_image(img, dpi=150)
        out = dp.remove_lines(direction="horizontal")(page).raster()
        assert (out[150:153, 200:400] > 200).all()      # rule gone
        assert (out[65:85, 105:125] < 60).all()          # blob kept

    def test_remove_lines_handles_vertical_rules(self, both_image_backends):
        img = np.full((400, 300), 255, np.uint8)
        img[20:380, 150:153] = 0
        out = dp.remove_lines(direction="vertical")(
            dp.page_from_image(img, dpi=150)).raster()
        assert (out[100:300, 150:153] > 200).all()

    def test_remove_lines_rejects_an_unknown_direction(self, page):
        with pytest.raises(dp.ConfigError):
            dp.remove_lines(direction="diagonal")(page)

    def test_remove_lines_is_a_no_op_on_a_page_with_none(self, page):
        assert np.array_equal(dp.remove_lines()(page).raster(), page.raster())

    def test_remove_stamps_suppresses_coloured_overlay(self, both_image_backends):
        img = np.full((300, 400, 3), 255, np.uint8)
        img[100:120, 20:380] = 0                 # black print
        img[90:130, 250:330] = [180, 30, 160]    # a violet stamp over it
        page = dp.page_from_image(img, dpi=150)
        out = dp.remove_stamps()(page)
        assert out.raster().ndim == 2
        assert out.meta["stamp_pixels_removed"] > 0
        assert (out.raster()[105:115, 50:200] < 60).all()   # black print survived

    def test_remove_stamps_declines_on_a_colourful_page(self):
        rng = np.random.RandomState(1)
        img = rng.randint(0, 255, (200, 200, 3)).astype(np.uint8)
        out = dp.remove_stamps()(dp.page_from_image(img, dpi=150))
        assert "skipped" in out.meta.get("remove_stamps", "")

    def test_remove_stamps_is_a_no_op_on_grayscale(self, page):
        out = dp.remove_stamps()(page)
        assert "skipped" in out.meta.get("remove_stamps", "")


class TestSplitting:
    def test_splits_two_documents_side_by_side(self, bill_image):
        left = dp.resize_image(bill_image, scale=0.5)
        gutter = np.full((left.shape[0], 160), 255, np.uint8)
        combined = np.concatenate([left, gutter, left], axis=1)
        parts = dp.split_multi_bill_page(dp.page_from_image(combined, dpi=150))
        assert len(parts) == 2
        assert all(p.meta["split_from"] == 0 for p in parts)
        assert parts[0].width_pt < dp.page_from_image(combined, dpi=150).width_pt

    def test_does_not_split_an_ordinary_page(self, page):
        assert len(dp.split_multi_bill_page(page)) == 1

    def test_split_document_renumbers_pages(self, bill_image):
        half = dp.resize_image(bill_image, scale=0.5)
        gutter = np.full((half.shape[0], 160), 255, np.uint8)
        combined = np.concatenate([half, gutter, half], axis=1)
        doc = dp.document_from_images([combined], dpi=150)
        out = dp.split_document(doc)
        assert [p.index for p in out.pages] == list(range(len(out.pages)))


# -- policies ----------------------------------------------------------------

class TestPolicies:
    def test_clean_page_needs_almost_nothing(self, page):
        page.quality = dp.PageQuality(blur=0.9, contrast=0.9, effective_dpi=400,
                                      noise=0.05, illumination=0.95, skew_deg=0.05)
        assert dp.default_policy(page) == []

    def test_degraded_page_gets_targeted_corrections(self, page):
        page.quality = dp.PageQuality(blur=0.15, contrast=0.2, effective_dpi=140,
                                      noise=0.5, illumination=0.3, skew_deg=3.0)
        names = [op.name for op in dp.default_policy(page)]
        assert set(names) == {"normalize_illumination", "denoise", "ensure_dpi",
                              "deskew", "unsharp", "autocontrast"}

    def test_default_policy_never_binarises(self, page):
        """Binarisation helps classical OCR and hurts vision models, so it
        cannot live in the shared default."""
        page.quality = dp.PageQuality(blur=0.05, contrast=0.05, effective_dpi=80,
                                      noise=0.9, illumination=0.1, skew_deg=9.0)
        assert "binarize" not in [op.name for op in dp.default_policy(page)]

    def test_ocr_policy_binarises(self, page):
        assert "binarize" in [op.name for op in dp.ocr_policy(page)]

    def test_vlm_policy_does_not_binarise_and_caps_size(self, page):
        names = [op.name for op in dp.vlm_policy(page)]
        assert "binarize" not in names
        assert "resize_max_side" in names

    def test_no_policy_is_empty(self, page):
        assert dp.no_policy(page) == []

    def test_fixed_policy_ignores_page_state(self, page):
        policy = dp.fixed_policy(dp.to_grayscale())
        assert [op.name for op in policy(page)] == ["to_grayscale"]

    def test_thresholds_change_the_policy(self, page):
        page.quality = dp.PageQuality(blur=0.9, contrast=0.9, effective_dpi=250,
                                      noise=0.0, illumination=1.0)
        assert dp.default_policy(page, dp.QualityThresholds(min_dpi=200)) == []
        assert [o.name for o in dp.default_policy(page,
                                                  dp.QualityThresholds(min_dpi=400))] \
            == ["ensure_dpi"]


class TestPreprocessDocument:
    def test_measures_then_corrects_then_remeasures(self, bill_image):
        doc = dp.document_from_images([dp.rotate_image(bill_image, -3.0,
                                                       border_value=255)], dpi=150)
        out = dp.preprocess(doc)
        assert "deskew" in [h.op for h in out.pages[0].history]
        assert abs(out.pages[0].quality.skew_deg) < 0.5   # re-measured after

    def test_does_not_mutate_the_input_document(self, bill_image):
        doc = dp.document_from_images([bill_image], dpi=150)
        original = doc.pages[0].pixel_size
        dp.preprocess(doc, ops=[dp.resize_max_side(200)])
        assert doc.pages[0].pixel_size == original

    def test_in_place_mutates(self, bill_image):
        doc = dp.document_from_images([bill_image], dpi=150)
        out = dp.preprocess(doc, ops=[dp.resize_max_side(200)], in_place=True)
        assert out is doc
        assert max(doc.pages[0].pixel_size) == 200

    def test_explicit_ops_override_the_policy(self, bill_image):
        doc = dp.document_from_images([bill_image], dpi=150)
        out = dp.preprocess(doc, ops=[dp.to_grayscale()])
        assert [h.op for h in out.pages[0].history if h.op != "measure_quality"] == \
               ["to_grayscale"]

    def test_pages_without_rasters_pass_through(self):
        doc = dp.Document(pages=[dp.Page(index=0)])
        assert dp.preprocess(doc).pages[0].history == []

    def test_parallel_and_serial_agree(self, bill_image):
        images = [bill_image, dp.rotate_image(bill_image, -2.0, border_value=255)]
        serial = dp.preprocess(dp.document_from_images(images, dpi=150))
        parallel = dp.preprocess(dp.document_from_images(images, dpi=150), max_workers=2)
        assert [[h.op for h in p.history] for p in serial.pages] == \
               [[h.op for h in p.history] for p in parallel.pages]
