"""Page quality measurement.

These tests exist because the preprocessing policy and the confidence prior are
both *functions of these numbers*.  A metric that is merely plausible is not
good enough: each one is checked for the property it is supposed to have --
blur must be contrast-invariant, contrast must work on a sparse page, skew must
be accurate to a fraction of a degree, and the DPI estimate must survive blur.
"""

import pytest

import docpipe as dp
from conftest import degrade, render_text_page, BILL_LINES

np = pytest.importorskip("numpy")


class TestBlur:
    def test_sharp_scores_above_blurred(self, bill_image, both_image_backends):
        assert dp.measure_blur(bill_image) > dp.measure_blur(degrade(bill_image, blur=2.0))

    def test_monotone_in_blur_radius(self, bill_image):
        scores = [dp.measure_blur(degrade(bill_image, blur=s)) for s in (0.0, 1.0, 2.0, 4.0)]
        assert scores == sorted(scores, reverse=True)

    def test_is_invariant_to_contrast(self, bill_image):
        """Fading is measure_contrast's job to report, not this one's."""
        faded = (np.asarray(bill_image).astype(float) * 0.35 + 150).astype(np.uint8)
        assert dp.measure_blur(faded) == pytest.approx(dp.measure_blur(bill_image), abs=0.06)

    def test_noise_does_not_read_as_extra_sharpness(self, bill_image):
        """The classic variance-of-Laplacian failure: grain scoring as detail."""
        noisy = degrade(bill_image, noise=25.0)
        assert dp.measure_blur(noisy) <= dp.measure_blur(bill_image) + 0.02

    def test_sparse_page_is_not_reported_as_blurry(self):
        """The other VoL failure: a crisp page with little ink."""
        img = np.full((400, 600), 255, np.uint8)
        img[100:112, 50:550] = 0
        assert dp.measure_blur(img) > 0.6

    def test_blank_page_is_not_penalised(self, blank_image):
        assert dp.measure_blur(blank_image) == 1.0

    def test_bounded(self, bill_image):
        for image in (bill_image, degrade(bill_image, blur=6.0)):
            assert 0.0 <= dp.measure_blur(image) <= 1.0


class TestContrast:
    def test_faded_scores_lower(self, bill_image):
        faded = (np.asarray(bill_image).astype(float) * 0.3 + 170).astype(np.uint8)
        assert dp.measure_contrast(faded) < dp.measure_contrast(bill_image)

    def test_sparse_page_is_full_contrast_not_zero(self):
        """A 97%-white page has both the 5th and 95th percentile at 255; a
        naive percentile spread reports 0.0 for a perfectly crisp page."""
        img = np.full((400, 600), 255, np.uint8)
        img[100:110, 50:550] = 0
        assert dp.measure_contrast(img) > 0.9

    def test_flat_grey_has_no_contrast(self):
        assert dp.measure_contrast(np.full((50, 50), 128, np.uint8)) < 0.05


class TestSkew:
    @pytest.mark.parametrize("angle", [-9.0, -4.0, -1.5, 0.0, 0.8, 3.0, 7.5])
    def test_recovers_a_known_rotation(self, bill_image, angle):
        rotated = dp.rotate_image(bill_image, -angle, border_value=255)
        assert dp.estimate_skew(rotated) == pytest.approx(angle, abs=0.3)

    @pytest.mark.parametrize("method", ["projection", "hough", "minarearect", "auto"])
    def test_every_method_agrees_on_sign_and_magnitude(self, bill_image, method):
        # _cv2(), not have("cv2"): OpenCV may be installed but switched off.
        if method in ("hough", "minarearect") and dp._cv2() is None:
            pytest.skip("needs OpenCV enabled")
        rotated = dp.rotate_image(bill_image, -3.0, border_value=255)
        assert dp.estimate_skew(rotated, method=method) == pytest.approx(3.0, abs=0.6)

    def test_projection_alone_is_accurate_without_opencv(self, bill_image):
        rotated = dp.rotate_image(bill_image, -2.5, border_value=255)
        with dp.without_opencv():
            assert dp.estimate_skew(rotated) == pytest.approx(2.5, abs=0.3)

    def test_blank_page_reports_no_skew(self, blank_image):
        assert dp.estimate_skew(blank_image) == 0.0

    def test_sparse_and_dense_projection_paths_agree(self, bill_image):
        """estimate_skew shifts ink coordinates (forward map) while
        row_projection_at_angle shears the raster (inverse map).  They must
        peak at the same angle, or one of them has its sign backwards."""
        rotated = dp.rotate_image(bill_image, -3.0, border_value=255)
        small = dp.resize_image(rotated, scale=0.5)
        mask = dp.ink_mask(small).astype(float)
        angles = [a * 0.5 for a in range(-16, 17)]
        dense = max(angles,
                    key=lambda a: float(np.var(dp.row_projection_at_angle(mask, a))))
        assert dense == pytest.approx(dp.estimate_skew(rotated, method="projection"),
                                      abs=0.6)

    def test_a_straight_noisy_page_is_not_reported_as_wildly_skewed(self, bill_image):
        """Two traps meet here.  Clamping sheared-off ink heaps it onto the
        first and last row, and since more ink is sheared off at larger angles,
        that artefact hands ever-larger variance to ever-more-extreme angles.
        A noisy page then reports a confident 15 degrees, and deskewing on it
        destroys the page."""
        noisy = degrade(bill_image, noise=22.0, shadow=0.4)
        assert abs(dp.estimate_skew(noisy)) < 1.0

    def test_a_real_skew_still_beats_the_straight_page_prior(self, bill_image):
        noisy = degrade(bill_image, noise=15.0, skew=-6.0)
        assert dp.estimate_skew(noisy) == pytest.approx(6.0, abs=0.5)

    def test_extreme_angles_are_not_rewarded_by_shear_clipping(self, bill_image):
        """The projection score at the true angle must beat the score at the
        search limit, not merely tie with it."""
        rotated = dp.rotate_image(bill_image, -3.0, border_value=255)
        assert dp.estimate_skew(rotated, method="projection") == \
            pytest.approx(3.0, abs=0.3)

    def test_result_is_clamped_to_max_angle(self, bill_image):
        rotated = dp.rotate_image(bill_image, -30.0, border_value=255)
        assert abs(dp.estimate_skew(rotated, max_angle=15.0)) <= 15.0

    def test_unknown_method_is_rejected(self, bill_image):
        with pytest.raises(dp.ConfigError):
            dp.estimate_skew(bill_image, method="telepathy")


class TestResolution:
    def test_line_pitch_matches_the_rendered_leading(self):
        image, _ = render_text_page(BILL_LINES, font_size=26, leading=1.6)
        assert dp.estimate_line_pitch(image) == pytest.approx(26 * 1.6, abs=2.0)

    def test_line_pitch_returns_zero_on_an_aperiodic_page(self):
        rng = np.random.RandomState(3)
        assert dp.estimate_line_pitch(rng.randint(0, 255, (400, 400)).astype(np.uint8)) == 0.0

    def test_dpi_survives_blur(self, bill_image):
        """Stroke width inflates under blur; line pitch does not, which is why
        the estimator prefers it."""
        sharp = dp.estimate_effective_dpi(bill_image)
        blurred = dp.estimate_effective_dpi(degrade(bill_image, blur=3.0))
        assert blurred == pytest.approx(sharp, rel=0.35)

    def test_dpi_halves_when_the_page_is_halved(self, bill_image):
        half = dp.resize_image(bill_image, scale=0.5)
        full = dp.estimate_effective_dpi(bill_image)
        assert dp.estimate_effective_dpi(half) < full * 0.75

    def test_falls_back_when_there_is_no_ink(self, blank_image):
        assert dp.estimate_effective_dpi(blank_image, fallback=201) == 201

    def test_stroke_width_is_measured_in_pixels(self):
        img = np.full((100, 100), 255, np.uint8)
        img[40:44, 10:90] = 0        # a 4px-thick bar
        assert dp.estimate_stroke_width(img) == pytest.approx(4.0, abs=1.0)


class TestNoiseAndIllumination:
    def test_noise_rises_with_added_grain(self, bill_image):
        assert dp.measure_noise(degrade(bill_image, noise=30.0)) > dp.measure_noise(bill_image)

    def test_illumination_falls_under_a_shadow_gradient(self, bill_image):
        assert dp.measure_illumination(degrade(bill_image, shadow=0.6)) < \
               dp.measure_illumination(bill_image)

    def test_even_lighting_scores_high(self, bill_image):
        assert dp.measure_illumination(bill_image) > 0.7

    def test_ink_coverage_is_a_fraction(self, bill_image, blank_image):
        assert 0.0 < dp.measure_ink_coverage(bill_image) < 0.5
        assert dp.measure_ink_coverage(blank_image) < 0.001


class TestVerdicts:
    def test_clean_page_is_clean(self, bill_image):
        q = dp.measure_quality(bill_image, raster_dpi=150)
        q.effective_dpi = 300      # isolate the verdict from the synthetic DPI
        assert dp.classify_quality(q) is dp.Verdict.CLEAN

    def test_severe_blur_is_unreadable(self, bill_image):
        q = dp.measure_quality(degrade(bill_image, blur=10.0, scale=0.15), raster_dpi=150)
        assert q.verdict is dp.Verdict.UNREADABLE

    def test_skew_alone_makes_a_page_degraded(self, bill_image):
        q = dp.measure_quality(dp.rotate_image(bill_image, -3.0, border_value=255),
                               raster_dpi=300)
        assert q.verdict is dp.Verdict.DEGRADED

    def test_blank_page_is_clean_because_there_is_nothing_to_misread(self, blank_image):
        q = dp.measure_quality(blank_image, raster_dpi=300)
        assert q.verdict is dp.Verdict.CLEAN

    def test_inverted_page_is_flagged_unreadable(self, bill_image):
        q = dp.measure_quality(255 - np.asarray(bill_image), raster_dpi=300)
        assert q.verdict is dp.Verdict.UNREADABLE

    def test_thresholds_are_tunable(self, bill_image):
        q = dp.measure_quality(bill_image, raster_dpi=150)
        strict = dp.QualityThresholds(min_dpi=1200)
        assert dp.classify_quality(q, strict) is dp.Verdict.DEGRADED


class TestMeasurePage:
    def test_measures_and_records_history(self, bill_image):
        page = dp.page_from_image(bill_image, dpi=150)
        dp.measure_page(page)
        assert page.quality.raster_dpi == 150
        assert page.history[-1].op == "measure_quality"

    def test_page_without_a_raster_is_left_alone(self):
        page = dp.Page(index=0)
        dp.measure_page(page)
        assert page.history == []

    def test_measure_document_covers_every_page(self, bill_image):
        doc = dp.document_from_images([bill_image, bill_image], dpi=150)
        dp.measure_document(doc)
        assert all(p.quality.ink_coverage > 0 for p in doc.pages)

    def test_parallel_measurement_matches_serial(self, bill_image):
        images = [bill_image] * 3
        serial = dp.measure_document(dp.document_from_images(images, dpi=150))
        parallel = dp.measure_document(dp.document_from_images(images, dpi=150),
                                       max_workers=3)
        assert [p.quality.to_dict() for p in serial.pages] == \
               [p.quality.to_dict() for p in parallel.pages]
