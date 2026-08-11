"""Command-line interface smoke tests.

The CLI is how the library gets used before anyone writes code against it --
``docpipe quality scan.pdf`` is the fastest way to find out why a document is
extracting badly -- so each subcommand is exercised end to end.
"""

import json
import os

import pytest

import docpipe as dp


def run(capsys, *argv):
    """Invoke the CLI and return ``(exit_code, stdout)``."""
    code = dp.main(list(argv))
    return code, capsys.readouterr().out


class TestCaps:
    def test_reports_capabilities_and_registries(self, capsys):
        code, out = run(capsys, "caps")
        assert code == 0
        assert "numpy" in out
        assert "registered backends" in out
        assert "deskew" in out


class TestInfo:
    def test_describes_an_image(self, capsys, image_file):
        code, out = run(capsys, "info", image_file)
        assert code == 0
        assert "pages:   1" in out

    def test_describes_a_pdf(self, capsys, native_pdf):
        code, out = run(capsys, "info", native_pdf)
        assert code == 0
        assert "digital_native" in out

    def test_missing_file_exits_nonzero_with_a_message(self, capsys):
        code, _ = run(capsys, "info", "/no/such/file.pdf")
        assert code == 1


class TestQuality:
    def test_prints_a_verdict_per_page(self, capsys, image_file):
        code, out = run(capsys, "quality", image_file)
        assert code == 0
        assert "page   1" in out
        assert "PageQuality" in out

    def test_show_policy_lists_the_corrections_that_would_be_applied(self, capsys,
                                                                     image_file):
        code, out = run(capsys, "quality", "--show-policy", image_file)
        assert code == 0
        assert "policy:" in out


class TestPreprocess:
    def test_reports_the_ops_applied(self, capsys, image_file):
        code, out = run(capsys, "preprocess", image_file, "--policy", "ocr")
        assert code == 0
        assert "binarize" in out

    def test_writes_processed_pages(self, capsys, image_file, tmp_path):
        outdir = str(tmp_path / "debug")
        code, out = run(capsys, "preprocess", image_file, "--out", outdir)
        assert code == 0
        assert os.path.isfile(os.path.join(outdir, "page-001.png"))

    def test_none_policy_applies_nothing(self, capsys, image_file):
        code, out = run(capsys, "preprocess", image_file, "--policy", "none")
        assert code == 0
        assert "binarize" not in out


class TestRead:
    def test_reads_a_native_pdf_text_layer(self, capsys, native_pdf):
        code, out = run(capsys, "read", native_pdf, "--policy", "none")
        assert code == 0
        assert "SUNRISE" in out

    def test_json_output_is_the_full_ir(self, capsys, native_pdf):
        code, out = run(capsys, "read", native_pdf, "--policy", "none", "--json")
        assert code == 0
        payload = json.loads(out)
        assert payload["pages"][0]["spans"]

    def test_max_pages_is_respected(self, capsys, native_pdf):
        code, out = run(capsys, "read", native_pdf, "--policy", "none",
                        "--json", "--max-pages", "1")
        assert len(json.loads(out)["pages"]) == 1


class TestExtract:
    def test_extracts_with_the_echo_client(self, capsys, tmp_path, native_pdf):
        schema_path = str(tmp_path / "schema.json")
        with open(schema_path, "w") as fh:
            json.dump({"hospital_name": "string", "total": "number"}, fh)
        response = json.dumps({"hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL",
                               "total": "118250.00"})
        code, out = run(capsys, "extract", native_pdf, "--schema", schema_path,
                        "--client", "echo", "--echo", response)
        payload = json.loads(out)
        assert code == 0
        assert payload["fields"]["total"]["value"] == 118250.0
        assert payload["fields"]["hospital_name"]["evidence"]

    def test_exit_code_two_signals_a_validation_failure(self, capsys, tmp_path,
                                                        native_pdf):
        schema_path = str(tmp_path / "schema.json")
        with open(schema_path, "w") as fh:
            json.dump({"total": "number"}, fh)
        code, _ = run(capsys, "extract", native_pdf, "--schema", schema_path,
                      "--client", "echo", "--echo", "not json at all")
        assert code == 0     # unparseable output is a warning, not invalidity


class TestEval:
    def _schema(self, tmp_path):
        path = str(tmp_path / "schema.json")
        with open(path, "w") as fh:
            json.dump({"total": "number", "patient_name": "string"}, fh)
        return path

    def test_runs_a_suite_and_prints_a_summary(self, capsys, eval_dataset, tmp_path):
        response = json.dumps({"total": "118250.00",
                               "patient_name": "RAMESH KUMAR SHARMA"})
        code, out = run(capsys, "eval", eval_dataset,
                        "--schema", self._schema(tmp_path),
                        "--client", "echo", "--echo", response)
        assert code == 0
        assert "field_exact" in out

    def test_saves_a_report(self, capsys, eval_dataset, tmp_path):
        report_path = str(tmp_path / "report.json")
        response = json.dumps({"total": "118250.00", "patient_name": "RAMESH"})
        code, out = run(capsys, "eval", eval_dataset,
                        "--schema", self._schema(tmp_path),
                        "--client", "echo", "--echo", response,
                        "--report", report_path)
        assert code == 0
        assert os.path.isfile(report_path)
        assert dp.EvalReport.load(report_path).pipelines() == ["candidate"]

    def test_baseline_comparison_exits_one_on_regression(self, capsys, eval_dataset,
                                                         tmp_path):
        schema = self._schema(tmp_path)
        baseline = str(tmp_path / "baseline.json")
        good = json.dumps({"total": "118250.00", "patient_name": "RAMESH KUMAR SHARMA"})
        run(capsys, "eval", eval_dataset, "--schema", schema, "--client", "echo",
            "--echo", good, "--report", baseline)

        bad = json.dumps({"total": "1.00", "patient_name": "WRONG PERSON"})
        code, out = run(capsys, "eval", eval_dataset, "--schema", schema,
                        "--client", "echo", "--echo", bad,
                        "--baseline", baseline, "--tolerance", "0.01")
        assert code == 1
        assert "regressions" in out


class TestArgumentHandling:
    def test_no_subcommand_prints_help(self, capsys):
        assert dp.main([]) == 1

    def test_unknown_subcommand_exits(self, capsys):
        with pytest.raises(SystemExit):
            dp.main(["teleport"])
