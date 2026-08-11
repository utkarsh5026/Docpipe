"""Layer 5: the evaluation harness.

If only one component of this library were built, it should be this one --- so
it gets tested against *known* answers rather than plausible ones.  The
``TruthBackend`` fixture makes a controlled number of mistakes, so an assertion
like "field_exact is exactly 0.75" is meaningful.
"""

import json
import os

import pytest

import docpipe as dp
from conftest import BILL_TRUTH

np = pytest.importorskip("numpy")


SCHEMA = {
    "hospital_name": "string",
    "patient_name": "string",
    "bill_number": "string",
    "admission_date": "date",
    "discharge_date": "date",
    "total": "number",
}

PERFECT = json.dumps({
    "hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL",
    "patient_name": "RAMESH KUMAR SHARMA",
    "bill_number": "INV-2024-04-7781",
    "admission_date": "2024-04-12",
    "discharge_date": "2024-04-19",
    "total": "118250.00",
})


def make_pipeline(response=PERFECT, cost=None, fail_on=None):
    """A pipeline function of the shape EvalSuite.run expects."""
    def pipeline(path):
        if fail_on and fail_on in path:
            raise RuntimeError("simulated pipeline failure")
        doc = dp.ingest(path)
        return dp.extract(doc, SCHEMA,
                          client=dp.EchoClient(response,
                                               cost=cost or dp.Cost(amount=0.01, calls=1)))
    return pipeline


# -- value comparison --------------------------------------------------------

class TestCompareValues:
    def test_identical_strings(self):
        assert dp.compare_values("SUNRISE", "SUNRISE")[0] is True

    def test_case_and_punctuation_are_ignored_for_exactness(self):
        exact, fuzzy, _ = dp.compare_values("Sunrise Hospital.", "SUNRISE HOSPITAL")
        assert exact is True and fuzzy == 1.0

    def test_a_single_ocr_slip_is_not_exact_but_a_reviewer_would_accept_it(self):
        exact, fuzzy, within = dp.compare_values("SUNRISE MULTISPECIALITY HOSPITAL",
                                                 "SUNR1SE MULTISPECIALITY HOSPITAL")
        assert exact is False       # not what a downstream system needs
        assert fuzzy > 0.95         # but one character out of thirty-two
        assert within is True       # so a human would call it correct

    def test_a_slip_in_a_short_value_is_not_waved_through(self):
        """Proportionally, one wrong character in seven is a different claim."""
        assert dp.compare_values("SUNRISE", "SUNR1SE")[2] is False

    def test_garbage_scores_low(self):
        _, fuzzy, within = dp.compare_values("SUNRISE HOSPITAL", "###@@@!!!")
        assert fuzzy < 0.3 and within is False

    def test_numbers_compare_numerically_not_textually(self):
        exact, _, _ = dp.compare_values("118250.00", "118,250", "number")
        assert exact is True

    def test_numeric_tolerance(self):
        _, _, within = dp.compare_values(100000, 100400, "number",
                                         numeric_tolerance=0.01)
        assert within is True

    def test_dates_in_different_formats_match(self):
        assert dp.compare_values("2024-04-12", "12-04-2024", "date")[0] is True

    def test_both_absent_counts_as_correct(self):
        assert dp.compare_values(None, None)[0] is True

    def test_one_absent_counts_as_wrong(self):
        assert dp.compare_values("x", None)[0] is False

    def test_arrays_compare_element_wise(self):
        exact, fuzzy, _ = dp.compare_values([{"a": 1}, {"a": 2}],
                                            [{"a": 1}, {"a": 3}], "array")
        assert exact is False and 0.0 < fuzzy < 1.0


# -- suite loading -----------------------------------------------------------

class TestEvalSuite:
    def test_loads_documents_paired_with_ground_truth(self, eval_dataset):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        assert len(suite) == 3
        assert {c.case_id for c in suite.cases} == {"case0", "case1", "case2"}
        assert suite.cases[0].truth["total"] == BILL_TRUTH["total"]

    def test_tags_are_loaded(self, eval_dataset):
        assert dp.EvalSuite.from_dir(eval_dataset).cases[0].tags == ["synthetic"]

    def test_missing_directory_is_rejected(self):
        with pytest.raises(dp.ConfigError):
            dp.EvalSuite.from_dir("/no/such/dataset")

    def test_an_empty_directory_is_rejected(self, tmp_path):
        with pytest.raises(dp.ConfigError):
            dp.EvalSuite.from_dir(str(tmp_path))

    def test_truth_without_a_document_is_skipped(self, tmp_path, bill_image):
        dp.save_image(bill_image, str(tmp_path / "a.png"))
        for name in ("a.json", "orphan.json"):
            with open(str(tmp_path / name), "w") as fh:
                json.dump({"total": 1}, fh)
        assert len(dp.EvalSuite.from_dir(str(tmp_path))) == 1

    def test_filter_narrows_the_suite(self, eval_dataset):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        assert len(suite.filter(lambda c: c.case_id == "case0")) == 1


# -- running -----------------------------------------------------------------

class TestRun:
    def test_scores_a_perfect_pipeline(self, eval_dataset):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        report = suite.run({"perfect": make_pipeline()})
        metrics = report.compute("perfect")
        # case2's ground truth deliberately carries a wrong total: 1 of 18
        # field comparisons must fail.
        assert metrics["field_exact"] == pytest.approx(17.0 / 18.0, abs=0.01)

    def test_per_field_breakdown_isolates_the_failing_field(self, eval_dataset):
        """The aggregate hides everything that matters: a pipeline can gain two
        points overall while losing eight on the only field anyone uses."""
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        by_field = report.by_field("p")
        assert by_field["patient_name"]["correct"] == 1.0
        assert by_field["total"]["correct"] == pytest.approx(2.0 / 3.0, abs=1e-3)

    def test_by_page_quality_splits_the_results(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        buckets = report.by_page_quality("p")
        assert buckets
        assert sum(b["n"] for b in buckets.values()) == 18

    def test_a_failing_pipeline_is_recorded_not_fatal(self, eval_dataset):
        """A suite that aborts on the first bad scan measures nothing."""
        report = dp.EvalSuite.from_dir(eval_dataset).run(
            {"flaky": make_pipeline(fail_on="case1")})
        errors = report.errors("flaky")
        assert len(errors) == 1 and errors[0][0] == "case1"
        assert report.compute("flaky")["error_rate"] == pytest.approx(1.0 / 3.0, abs=1e-3)

    def test_cost_and_latency_are_measured(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run(
            {"p": make_pipeline(cost=dp.Cost(amount=0.25, calls=1))})
        metrics = report.compute("p")
        assert metrics["cost_per_doc"] == pytest.approx(0.25)
        assert metrics["latency_p95"] >= metrics["latency_p50"] >= 0.0

    def test_evidence_coverage_is_reported(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        assert 0.0 <= report.compute("p")["evidence_coverage"] <= 1.0

    def test_unknown_metric_is_rejected(self, eval_dataset):
        with pytest.raises(dp.ConfigError) as exc:
            dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()},
                                                    metrics=["vibes"])
        assert "field_exact" in str(exc.value)

    def test_worst_cases_surface_where_to_look(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        assert report.worst_cases("p")[0][0] == "case2"

    def test_parallel_and_serial_agree(self, eval_dataset):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        serial = suite.run({"p": make_pipeline()}).compute("p")
        parallel = suite.run({"p": make_pipeline()}, max_workers=3).compute("p")
        assert serial["field_exact"] == parallel["field_exact"]


# -- comparison and regression ----------------------------------------------

class TestComparison:
    def test_compare_reports_per_field_regressions(self, eval_dataset):
        degraded = json.dumps(dict(json.loads(PERFECT),
                                   patient_name="COMPLETELY WRONG NAME"))
        report = dp.EvalSuite.from_dir(eval_dataset).run({
            "baseline": make_pipeline(PERFECT),
            "candidate": make_pipeline(degraded),
        })
        comparison = report.compare("baseline", "candidate")
        assert comparison["regressed"] is True
        assert comparison["regressions"][0]["field"] == "patient_name"
        assert comparison["deltas"]["field_exact"] < 0

    def test_compare_reports_improvements(self, eval_dataset):
        broken = json.dumps(dict(json.loads(PERFECT), patient_name=None))
        report = dp.EvalSuite.from_dir(eval_dataset).run({
            "old": make_pipeline(broken),
            "new": make_pipeline(PERFECT),
        })
        comparison = report.compare("old", "new")
        assert comparison["regressed"] is False
        assert comparison["improvements"][0]["field"] == "patient_name"

    def test_identical_pipelines_show_no_movement(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({
            "a": make_pipeline(), "b": make_pipeline()})
        comparison = report.compare("a", "b")
        assert comparison["regressions"] == [] and comparison["improvements"] == []

    def test_regression_vs_a_saved_report_is_the_ci_gate(self, eval_dataset, tmp_path):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        baseline_path = str(tmp_path / "baseline.json")
        suite.run({"p": make_pipeline(PERFECT)}).save(baseline_path)

        degraded = json.dumps(dict(json.loads(PERFECT), patient_name="WRONG"))
        current = suite.run({"p": make_pipeline(degraded)})
        outcome = current.regression_vs(baseline_path)
        assert outcome["regressed"] is True
        assert any(r["metric"] == "field_exact" for r in outcome["regressions"])

    def test_no_regression_against_itself(self, eval_dataset, tmp_path):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        path = str(tmp_path / "baseline.json")
        report = suite.run({"p": make_pipeline()})
        report.save(path)
        assert report.regression_vs(path)["regressed"] is False

    def test_lower_is_better_metrics_flip_sign(self, eval_dataset, tmp_path):
        """A cost increase is a regression even though the number went up."""
        suite = dp.EvalSuite.from_dir(eval_dataset)
        path = str(tmp_path / "cheap.json")
        suite.run({"p": make_pipeline(cost=dp.Cost(amount=0.01, calls=1))}).save(path)
        pricey = suite.run({"p": make_pipeline(cost=dp.Cost(amount=1.00, calls=1))})
        regressions = pricey.regression_vs(path)["regressions"]
        assert any(r["metric"] == "cost_per_doc" for r in regressions)

    def test_tolerance_is_relative_for_unbounded_metrics(self, eval_dataset, tmp_path):
        """0.05 must mean 'five percent slower/pricier', not '0.05 milliseconds
        slower' -- otherwise scheduler noise fails every build."""
        suite = dp.EvalSuite.from_dir(eval_dataset)
        path = str(tmp_path / "b.json")
        suite.run({"p": make_pipeline(cost=dp.Cost(amount=0.010, calls=1))}).save(path)
        current = suite.run({"p": make_pipeline(cost=dp.Cost(amount=0.0102, calls=1))})
        assert current.regression_vs(path, tolerance=0.05)["regressed"] is False

    def test_a_real_cost_increase_still_fails(self, eval_dataset, tmp_path):
        suite = dp.EvalSuite.from_dir(eval_dataset)
        path = str(tmp_path / "b.json")
        suite.run({"p": make_pipeline(cost=dp.Cost(amount=0.010, calls=1))}).save(path)
        current = suite.run({"p": make_pipeline(cost=dp.Cost(amount=0.10, calls=1))})
        regressions = current.regression_vs(path, tolerance=0.05)["regressions"]
        assert any(r["metric"] == "cost_per_doc" for r in regressions)

    def test_metrics_filter_gates_on_accuracy_alone(self, eval_dataset, tmp_path):
        """Wall-clock varies on shared CI hardware regardless of the code."""
        suite = dp.EvalSuite.from_dir(eval_dataset)
        path = str(tmp_path / "b.json")
        suite.run({"p": make_pipeline()}).save(path)
        current = suite.run({"p": make_pipeline()})
        outcome = current.regression_vs(path, metrics=["field_exact",
                                                       "numeric_tolerance"])
        assert outcome["regressed"] is False
        assert all(r["metric"] in ("field_exact", "numeric_tolerance")
                   for r in outcome["regressions"])


# -- calibration loop --------------------------------------------------------

class TestCalibrationFromReport:
    def test_confidence_curve_is_produced(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        curve = report.confidence_curve("p", bins=5)
        assert len(curve) == 5
        assert sum(b["count"] for b in curve) == 18

    def test_a_calibrator_can_be_fitted_and_reused(self, eval_dataset, bill_document):
        """This closes the loop: measure, calibrate, ship the calibrator, and
        the numbers a reviewer sees start meaning what they say."""
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        calibrator = report.fit_calibrator("p", kind="platt")
        assert isinstance(calibrator, dp.PlattCalibrator)
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(PERFECT),
                            calibrator=calibrator)
        assert 0.0 <= result["total"].confidence <= 1.0

    def test_unknown_calibrator_kind_is_rejected(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        with pytest.raises(dp.ConfigError):
            report.fit_calibrator("p", kind="tea-leaves")


# -- report I/O --------------------------------------------------------------

class TestReportIO:
    def test_round_trips_through_disk(self, eval_dataset, tmp_path):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        path = report.save(str(tmp_path / "nested" / "report.json"))
        assert os.path.exists(path)
        restored = dp.EvalReport.load(path)
        assert restored.compute("p") == report.compute("p")
        assert restored.by_field("p") == report.by_field("p")

    def test_summary_is_human_readable(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({
            "baseline": make_pipeline(), "candidate": make_pipeline()})
        text = report.summary()
        assert "field_exact" in text
        assert "baseline" in text and "candidate" in text

    def test_summary_reports_errors(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run(
            {"p": make_pipeline(fail_on="case1")})
        assert "errored" in report.summary()

    def test_to_dict_includes_computed_metrics(self, eval_dataset):
        report = dp.EvalSuite.from_dir(eval_dataset).run({"p": make_pipeline()})
        assert "field_exact" in report.to_dict()["computed"]["p"]

    def test_empty_report_summary_does_not_crash(self):
        assert "no results" in dp.EvalReport(suite="empty").summary()
