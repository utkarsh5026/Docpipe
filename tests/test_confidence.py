"""Confidence fusion and calibration.

The premise these tests defend: a model's self-reported confidence measures
fluency, not correctness, so a defensible score has to fuse independent
evidence and then be calibrated against labelled data.  The properties that
matter are asserted directly -- absent signals must not be treated as zero, a
single strong disagreement must dominate rather than be averaged away, and a
fitted calibrator must actually reduce calibration error.
"""

import math
import random

import pytest

import docpipe as dp


class TestLogitSigmoid:
    def test_round_trip(self):
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert dp.sigmoid(dp.logit(p)) == pytest.approx(p, abs=1e-6)

    def test_extremes_are_clamped_not_infinite(self):
        assert math.isfinite(dp.logit(0.0)) and math.isfinite(dp.logit(1.0))

    def test_sigmoid_is_stable_at_both_tails(self):
        assert dp.sigmoid(-800) == pytest.approx(0.0)
        assert dp.sigmoid(800) == pytest.approx(1.0)


class TestFuseConfidence:
    def test_no_signals_returns_the_prior(self):
        assert dp.fuse_confidence({}) == 0.5
        assert dp.fuse_confidence({}, prior=0.3) == 0.3

    def test_all_signals_absent_is_the_same_as_no_signals(self):
        assert dp.fuse_confidence({"backend_conf": None, "validation": None}) == 0.5

    def test_absent_is_not_the_same_as_zero(self):
        """Encoding 'the engine declined to guess' as 0.0 would punish every
        backend that is honest about not knowing."""
        absent = dp.fuse_confidence({"cross_read_agree": 0.9, "backend_conf": None})
        zero = dp.fuse_confidence({"cross_read_agree": 0.9, "backend_conf": 0.0})
        assert absent > zero

    def test_agreeing_signals_reinforce(self):
        assert dp.fuse_confidence({"cross_read_agree": 0.95, "backend_conf": 0.95,
                                   "validation": 0.95}) > 0.9

    def test_a_failed_validation_pushes_a_field_below_the_review_threshold(self):
        """The operationally meaningful property: a field whose arithmetic does
        not check out ends up in front of a human, even when the page was clean
        and two engines agreed on the digits."""
        confident = {"cross_read_agree": 0.97, "backend_conf": 0.97, "format_match": 1.0}
        without = dp.fuse_confidence(confident)
        failed = dict(confident)
        failed["validation"] = 0.02
        with_failure = dp.fuse_confidence(failed)

        assert without > 0.9
        assert with_failure < 0.75                   # surfaces for review
        assert without - with_failure > 0.25         # and it is a large drop
        assert with_failure < (0.97 + 0.97 + 1.0 + 0.02) / 4   # beats averaging

    def test_no_single_signal_can_outvote_all_the_others(self):
        """An unclamped 1.0 is ~14 log-odds and would bury every measurement."""
        assert dp.fuse_confidence({"format_match": 1.0, "cross_read_agree": 0.05,
                                   "backend_conf": 0.05, "validation": 0.05}) < 0.3

    def test_result_is_bounded(self):
        assert 0.0 <= dp.fuse_confidence({"cross_read_agree": 0.0,
                                          "validation": 0.0}) <= 1.0
        assert 0.0 <= dp.fuse_confidence({"cross_read_agree": 1.0,
                                          "validation": 1.0}) <= 1.0

    def test_monotone_in_a_single_signal(self):
        scores = [dp.fuse_confidence({"cross_read_agree": v})
                  for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert scores == sorted(scores)

    def test_custom_weights_are_respected(self):
        signals = {"cross_read_agree": 0.99, "validation": 0.01}
        agreement_heavy = dp.fuse_confidence(signals, {"cross_read_agree": 1.0,
                                                       "validation": 0.01})
        validation_heavy = dp.fuse_confidence(signals, {"cross_read_agree": 0.01,
                                                        "validation": 1.0})
        assert agreement_heavy > 0.9 > validation_heavy

    def test_unknown_signal_names_are_ignored(self):
        assert dp.fuse_confidence({"vibes": 1.0}) == 0.5


class TestPageQualityPrior:
    def test_healthy_page_gives_a_high_prior(self):
        q = dp.PageQuality(blur=0.95, contrast=0.95, effective_dpi=400,
                           noise=0.0, illumination=1.0)
        assert dp.page_quality_prior(q) > 0.85

    def test_degraded_page_gives_a_low_prior(self):
        q = dp.PageQuality(blur=0.05, contrast=0.05, effective_dpi=90,
                           noise=0.9, illumination=0.2)
        assert dp.page_quality_prior(q) < 0.3

    def test_prior_is_localised_to_the_evidence_region(self, bill_image):
        """A page can be pristine at the top and destroyed at the bottom; a
        total read from the destroyed part must not inherit the average."""
        import numpy as np
        half = bill_image.shape[0] // 2
        damaged = np.asarray(bill_image).copy()
        damaged[half:] = dp.gaussian_blur(damaged[half:], 6.0)

        page = dp.page_from_image(damaged, dpi=150)
        dp.measure_page(page)
        top = dp.BBox.from_pixels(0, 0, 0, damaged.shape[1], half - 10, 150)
        bottom = dp.BBox.from_pixels(0, 0, half + 10, damaged.shape[1],
                                     damaged.shape[0], 150)
        assert dp.page_quality_prior(page.quality, top, page) > \
               dp.page_quality_prior(page.quality, bottom, page)

    def test_falls_back_to_the_page_score_without_a_raster(self):
        page = dp.Page(index=0)
        q = dp.PageQuality(blur=0.5, contrast=0.5, effective_dpi=200)
        assert dp.page_quality_prior(q, dp.BBox(0, 0, 0, 10, 10), page) == \
            pytest.approx(dp.page_quality_prior(q), abs=0.01)


class TestSpanAlignment:
    def test_aligns_overlapping_spans(self):
        a = [dp.TextSpan("total", dp.BBox(0, 0, 0, 40, 12))]
        b = [dp.TextSpan("tota1", dp.BBox(0, 1, 0, 41, 12))]
        pairs = dp.align_spans(a, b)
        assert len(pairs) == 1 and pairs[0][0] is a[0] and pairs[0][1] is b[0]

    def test_unmatched_spans_are_reported_not_dropped(self):
        """A backend that simply missed a field must be penalised."""
        a = [dp.TextSpan("x", dp.BBox(0, 0, 0, 10, 10))]
        b = [dp.TextSpan("y", dp.BBox(0, 500, 500, 510, 510))]
        pairs = dp.align_spans(a, b)
        assert len(pairs) == 2
        assert sum(1 for x, y in pairs if x is None or y is None) == 2


class TestCrossReadAgreement:
    def _spans(self, words, source="a"):
        return [dp.TextSpan(w, dp.BBox(0, i * 50, 0, i * 50 + 40, 12), source=source)
                for i, w in enumerate(words)]

    def test_identical_reads_agree(self):
        words = ["Total", "48250.00", "INR"]
        assert dp.cross_read_agreement(self._spans(words), self._spans(words)) == \
            pytest.approx(1.0)

    def test_completely_different_reads_disagree(self):
        assert dp.cross_read_agreement(self._spans(["alpha", "beta"]),
                                       self._spans(["zzzz", "yyyy"])) < 0.3

    def test_an_empty_read_is_maximal_disagreement(self):
        assert dp.cross_read_agreement(self._spans(["a"]), []) == 0.0
        assert dp.cross_read_agreement([], []) == 0.0

    def test_agreement_is_weighted_by_length_not_by_span_count(self):
        """Losing a twelve-digit amount must cost more than losing 'of'.  An
        unweighted mean over spans would score both misses identically."""
        base = ["Total", "123456789012", "of"]
        lost_the_number = dp.cross_read_agreement(
            self._spans(base), self._spans(["Total", "999999999999", "of"]))
        lost_a_stopword = dp.cross_read_agreement(
            self._spans(base), self._spans(["Total", "123456789012", "XY"]))
        assert lost_the_number < lost_a_stopword

    def test_single_character_errors_cost_the_same_wherever_they_land(self):
        """The corollary of length weighting: the score is effectively
        character-level edit distance over the whole page."""
        base = ["Total", "123456789012", "of"]
        assert dp.cross_read_agreement(
            self._spans(base), self._spans(["Total", "923456789012", "of"])) == \
            pytest.approx(dp.cross_read_agreement(
                self._spans(base), self._spans(["Total", "123456789012", "oX"])))

    def test_falls_back_to_token_comparison_for_approximate_boxes(self):
        """A vision model's page-level boxes cannot be matched geometrically."""
        vlm = [dp.TextSpan("Total 48250.00 INR", dp.BBox(0, 0, 0, 600, 800),
                           meta={"approximate_bbox": True})]
        ocr = self._spans(["Total", "48250.00", "INR"])
        assert dp.cross_read_agreement(vlm, ocr) > 0.9

    def test_normalisation_ignores_punctuation_and_case(self):
        assert dp.cross_read_agreement(self._spans(["Total:"]),
                                       self._spans(["total"])) == pytest.approx(1.0)


class TestValueAgreement:
    def test_numbers_compare_numerically(self):
        assert dp.value_agreement(48250, 48250.0) == 1.0
        assert dp.value_agreement(48250, 48251) == 0.0

    def test_numeric_tolerance(self):
        assert dp.value_agreement(1000, 1005, numeric_tolerance=0.01) == 1.0

    def test_dates_compare_as_dates(self):
        import datetime
        assert dp.value_agreement(datetime.date(2024, 4, 12),
                                  datetime.date(2024, 4, 12)) == 1.0

    def test_strings_compare_fuzzily(self):
        assert 0.7 < dp.value_agreement("SUNRISE HOSPITAL", "SUNRISE H0SPITAL") < 1.0

    def test_none_never_agrees(self):
        assert dp.value_agreement(None, 5) == 0.0


class TestFormatMatch:
    @pytest.mark.parametrize("value,kind,expected", [
        ("48250.00", "number", 1.0),
        ("not a number", "number", 0.0),
        ("2024-04-12", "date", 1.0),
        ("hello", "date", 0.0),
        ("yes", "boolean", 1.0),
        ("perhaps", "boolean", 0.0),
    ])
    def test_shapes(self, value, kind, expected):
        assert dp.format_match_score(value, kind) == expected

    def test_a_date_in_an_amount_field_is_rejected(self):
        """The off-by-one-row failure: fluent, high-confidence, and wrong.
        No other signal in the fusion sees it."""
        assert dp.format_match_score("12-04-2024", "number") == 0.0

    def test_ocr_garbage_scores_low_for_a_string_field(self):
        assert dp.format_match_score("|||###~~~@@@", "string") < 0.2

    def test_clean_text_scores_high_for_a_string_field(self):
        assert dp.format_match_score("RAMESH KUMAR SHARMA", "string") > 0.9

    def test_none_scores_zero(self):
        assert dp.format_match_score(None, "number") == 0.0

    def test_unknown_type_yields_no_signal(self):
        assert dp.format_match_score("x", "widget") is None


def _synthetic_scores(n=600, overconfidence=0.25, seed=11):
    """Scores that are systematically overconfident by a known margin."""
    rng = random.Random(seed)
    scores, labels = [], []
    for _ in range(n):
        true_p = rng.random()
        reported = min(0.999, true_p + overconfidence * (1.0 - true_p))
        scores.append(reported)
        labels.append(rng.random() < true_p)
    return scores, labels


class TestCalibration:
    def test_identity_calibrator_passes_through(self):
        assert dp.IdentityCalibrator().predict(0.73) == 0.73

    def test_platt_reduces_calibration_error(self):
        scores, labels = _synthetic_scores()
        before = dp.expected_calibration_error(scores, labels)
        calibrator = dp.PlattCalibrator().fit(scores, labels)
        after = dp.expected_calibration_error(calibrator.predict_many(scores), labels)
        assert before > 0.1
        assert after < before / 2.0

    def test_platt_recovers_the_identity_on_calibrated_input(self):
        scores, labels = _synthetic_scores(overconfidence=0.0)
        calibrator = dp.PlattCalibrator().fit(scores, labels)
        assert calibrator.a == pytest.approx(1.0, abs=0.4)
        assert calibrator.b == pytest.approx(0.0, abs=0.4)

    def test_platt_round_trips(self):
        calibrator = dp.PlattCalibrator(a=1.4, b=-0.6)
        restored = dp.Calibrator.from_dict(calibrator.to_dict())
        assert restored.predict(0.8) == calibrator.predict(0.8)

    def test_isotonic_reduces_calibration_error(self):
        scores, labels = _synthetic_scores(n=800)
        before = dp.expected_calibration_error(scores, labels)
        calibrator = dp.IsotonicCalibrator().fit(scores, labels)
        after = dp.expected_calibration_error(calibrator.predict_many(scores), labels)
        assert after < before / 2.0

    def test_isotonic_is_monotone(self):
        scores, labels = _synthetic_scores(n=400)
        calibrator = dp.IsotonicCalibrator().fit(scores, labels)
        predictions = [calibrator.predict(x / 40.0) for x in range(41)]
        assert predictions == sorted(predictions)

    def test_isotonic_refuses_to_overfit_a_tiny_sample(self):
        with pytest.raises(dp.ConfigError) as exc:
            dp.IsotonicCalibrator().fit([0.5, 0.6], [True, False])
        assert "Platt" in str(exc.value)

    def test_isotonic_round_trips(self):
        scores, labels = _synthetic_scores(n=200)
        calibrator = dp.IsotonicCalibrator().fit(scores, labels)
        restored = dp.Calibrator.from_dict(calibrator.to_dict())
        assert restored.predict(0.7) == calibrator.predict(0.7)

    def test_unknown_calibrator_kind_is_rejected(self):
        with pytest.raises(dp.ConfigError):
            dp.Calibrator.from_dict({"kind": "astrology"})


class TestCalibrationMetrics:
    def test_perfect_calibration_scores_zero_error(self):
        scores = [0.0] * 50 + [1.0] * 50
        labels = [False] * 50 + [True] * 50
        assert dp.expected_calibration_error(scores, labels) == pytest.approx(0.0, abs=0.01)

    def test_total_overconfidence_scores_one(self):
        assert dp.expected_calibration_error([1.0] * 20, [False] * 20) == \
            pytest.approx(1.0, abs=0.01)

    def test_reliability_curve_shape(self):
        scores, labels = _synthetic_scores(n=300)
        curve = dp.reliability_curve(scores, labels, bins=10)
        assert len(curve) == 10
        assert sum(b["count"] for b in curve) == 300
        assert all(0.0 <= b["accuracy"] <= 1.0 for b in curve)

    def test_reliability_curve_reports_the_overconfidence_gap(self):
        scores, labels = _synthetic_scores(overconfidence=0.35)
        populated = [b for b in dp.reliability_curve(scores, labels) if b["count"] > 10]
        assert sum(b["gap"] for b in populated) < 0     # accuracy below confidence

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(dp.ConfigError):
            dp.reliability_curve([0.5], [True, False])

    def test_brier_rewards_sharpness(self):
        labels = [True] * 50 + [False] * 50
        sharp = [1.0] * 50 + [0.0] * 50
        hedged = [0.5] * 100
        assert dp.brier_score(sharp, labels) < dp.brier_score(hedged, labels)

    def test_ece_can_be_gamed_by_hedging_but_brier_cannot(self):
        """Why both are reported."""
        labels = [True] * 50 + [False] * 50
        hedged = [0.5] * 100
        assert dp.expected_calibration_error(hedged, labels) == pytest.approx(0.0, abs=0.01)
        assert dp.brier_score(hedged, labels) == pytest.approx(0.25)
