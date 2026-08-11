"""Layer 4: schema adaptation, prompting, lenient parsing, provenance, rules.

Everything here runs against :class:`docpipe.EchoClient`, so the whole
extraction path is exercised deterministically with no network and no spend.
That is the same device a consuming project is expected to use to test its own
schemas and validators, which is why it ships in the library.
"""

import datetime
import decimal
import json

import pytest

import docpipe as dp

np = pytest.importorskip("numpy")


# -- schema adaptation -------------------------------------------------------

class TestSchemaAdapterDictSpec:
    def test_simple_type_map(self):
        adapter = dp.SchemaAdapter.from_schema({"total": "number", "name": "string"})
        assert {f.name for f in adapter.fields} == {"total", "name"}
        assert adapter.field("total").type_name == "number"

    def test_expanded_form_carries_descriptions(self):
        adapter = dp.SchemaAdapter.from_schema({
            "total": {"type": "number", "description": "grand total payable",
                      "required": False}})
        spec = adapter.field("total")
        assert spec.description == "grand total payable"
        assert spec.required is False

    def test_dates_are_declared_with_a_format(self):
        adapter = dp.SchemaAdapter.from_schema({"admitted": "date"})
        assert adapter.json_schema["properties"]["admitted"]["format"] == "date"

    def test_build_returns_a_plain_dict(self):
        adapter = dp.SchemaAdapter.from_schema({"total": "number"})
        assert adapter.build({"total": decimal.Decimal("5")}) == \
            {"total": decimal.Decimal("5")}


class TestSchemaAdapterDataclass:
    def test_fields_and_optionality(self):
        import dataclasses
        from typing import Optional

        @dataclasses.dataclass
        class Bill:
            hospital: str
            total: float
            discharge: Optional[datetime.date] = None

        adapter = dp.SchemaAdapter.from_schema(Bill)
        assert adapter.name == "Bill"
        assert adapter.field("hospital").required is True
        assert adapter.field("discharge").required is False
        assert adapter.field("total").type_name == "number"

    def test_build_instantiates_the_dataclass(self):
        import dataclasses

        @dataclasses.dataclass
        class Bill:
            hospital: str

        assert dp.SchemaAdapter.from_schema(Bill).build({"hospital": "X"}).hospital == "X"


@pytest.mark.skipif(not dp.have("pydantic"), reason="needs pydantic")
class TestSchemaAdapterPydantic:
    def _model(self):
        from pydantic import BaseModel
        from typing import List, Optional

        class LineItem(BaseModel):
            description: str
            amount: float

        class HospitalBill(BaseModel):
            hospital_name: str
            patient_name: str
            admission_date: datetime.date
            discharge_date: Optional[datetime.date] = None
            line_items: List[LineItem] = []
            total: float

        return HospitalBill

    def test_fields_are_discovered(self):
        adapter = dp.SchemaAdapter.from_schema(self._model())
        assert {f.name for f in adapter.fields} == {
            "hospital_name", "patient_name", "admission_date", "discharge_date",
            "line_items", "total"}

    def test_optional_fields_are_not_required(self):
        adapter = dp.SchemaAdapter.from_schema(self._model())
        assert adapter.field("hospital_name").required is True
        assert adapter.field("discharge_date").required is False

    def test_dates_and_arrays_are_typed(self):
        adapter = dp.SchemaAdapter.from_schema(self._model())
        assert adapter.field("admission_date").type_name == "date"
        assert adapter.field("line_items").type_name == "array"

    def test_build_validates(self):
        adapter = dp.SchemaAdapter.from_schema(self._model())
        model = adapter.build({"hospital_name": "A", "patient_name": "B",
                               "admission_date": datetime.date(2024, 4, 12),
                               "total": 100.0, "line_items": []})
        assert model.hospital_name == "A"

    def test_invalid_data_raises_extraction_error(self):
        adapter = dp.SchemaAdapter.from_schema(self._model())
        with pytest.raises(dp.ExtractionError):
            adapter.build({"hospital_name": "A"})


def test_unsupported_schema_type_is_rejected():
    with pytest.raises(dp.ConfigError):
        dp.SchemaAdapter.from_schema(42)


# -- coercion ----------------------------------------------------------------

class TestCoercion:
    def test_amount_strings_become_decimals(self):
        assert dp.coerce_value("Rs. 1,23,456.78", "number") == decimal.Decimal("123456.78")

    def test_money_never_becomes_a_float(self):
        assert isinstance(dp.coerce_value(48250.0, "number"), decimal.Decimal)

    def test_dates_are_parsed(self):
        assert dp.coerce_value("12-04-2024", "date") == datetime.date(2024, 4, 12)

    def test_integers_truncate(self):
        assert dp.coerce_value("7.0", "integer") == 7

    def test_booleans(self):
        assert dp.coerce_value("Yes", "boolean") is True

    def test_arrays_coerce_their_items(self):
        assert dp.coerce_value(["1,000", "2,000"], "array", "number") == \
            [decimal.Decimal("1000"), decimal.Decimal("2000")]

    def test_a_scalar_for_an_array_field_is_wrapped(self):
        assert dp.coerce_value("1,000", "array", "number") == [decimal.Decimal("1000")]

    def test_uncoercible_values_become_none(self):
        assert dp.coerce_value("not a number", "number") is None

    def test_none_stays_none(self):
        assert dp.coerce_value(None, "number") is None

    def test_empty_strings_become_none(self):
        assert dp.coerce_value("   ", "string") is None


# -- lenient JSON ------------------------------------------------------------

class TestParseJsonLenient:
    def test_plain_json(self):
        assert dp.parse_json_lenient('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert dp.parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert dp.parse_json_lenient('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_preamble(self):
        assert dp.parse_json_lenient(
            'Here is the extracted data:\n{"a": 1}\nHope that helps!') == {"a": 1}

    def test_trailing_commas(self):
        assert dp.parse_json_lenient('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_nan_and_infinity_become_null(self):
        """Python's json accepts these as floats, and a NaN in an amount field
        is worse than a null: every comparison against it silently fails."""
        assert dp.parse_json_lenient('{"a": NaN}') == {"a": None}
        assert dp.parse_json_lenient('{"a": Infinity, "b": -Infinity}') == \
            {"a": None, "b": None}

    def test_non_finite_values_are_stripped_from_nested_structures(self):
        assert dp.parse_json_lenient('{"items": [{"amount": NaN}]}') == \
            {"items": [{"amount": None}]}

    def test_smart_quotes(self):
        assert dp.parse_json_lenient('{“a”: 1}') == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        payload = '{"note": "amount {net} due", "a": 1}'
        assert dp.parse_json_lenient("blah " + payload)["a"] == 1

    def test_nested_objects_survive(self):
        assert dp.parse_json_lenient(
            'x {"a": {"b": [1, 2]}} y') == {"a": {"b": [1, 2]}}

    def test_empty_and_unparseable_raise(self):
        with pytest.raises(dp.ExtractionError):
            dp.parse_json_lenient("")
        with pytest.raises(dp.ExtractionError):
            dp.parse_json_lenient("I could not read this document.")
        with pytest.raises(dp.ExtractionError):
            dp.parse_json_lenient(None)


# -- prompting and chunking --------------------------------------------------

class TestPrompting:
    def test_prompt_contains_the_schema_and_the_text(self):
        adapter = dp.SchemaAdapter.from_schema({"total": "number"})
        prompt = dp.build_extraction_prompt(adapter, "TOTAL 48250",
                                            context="Indian hospital bill, INR")
        assert '"total"' in prompt
        assert "TOTAL 48250" in prompt
        assert "Indian hospital bill" in prompt

    def test_system_prompt_forbids_invention(self):
        assert "Never" in dp.DEFAULT_EXTRACTION_SYSTEM
        assert "null" in dp.DEFAULT_EXTRACTION_SYSTEM

    def test_page_markers_are_included(self, bill_document):
        assert "--- page 1 ---" in dp.format_document_text(bill_document)

    def test_max_chars_truncates(self, bill_document):
        assert len(dp.format_document_text(bill_document, max_chars=200)) <= 200


class TestChunking:
    def _doc(self, pages, chars_per_page):
        doc = dp.Document(source_uri="big.pdf")
        for i in range(pages):
            page = dp.Page(index=i)
            page.spans = [dp.TextSpan("w" * chars_per_page, dp.BBox(i, 0, 0, 10, 10))]
            doc.pages.append(page)
        return doc

    def test_short_documents_are_not_split(self):
        assert len(dp.chunk_pages(self._doc(3, 100), max_chars=10000)) == 1

    def test_long_documents_split_on_page_boundaries(self):
        chunks = dp.chunk_pages(self._doc(10, 1000), max_chars=2500, overlap_pages=0)
        assert len(chunks) > 1
        assert sum(len(c.pages) for c in chunks) == 10

    def test_overlap_repeats_the_boundary_page(self):
        """Totals and discharge dates live exactly where a page breaks."""
        chunks = dp.chunk_pages(self._doc(6, 1000), max_chars=2500, overlap_pages=1)
        first_last = chunks[0].pages[-1].index
        assert chunks[1].pages[0].index == first_last

    def test_merge_prefers_the_first_non_null_scalar(self):
        merged = dp.merge_extracted_data([{"a": None, "b": 1}, {"a": 2, "b": 9}])
        assert merged == {"a": 2, "b": 1}

    def test_merge_concatenates_lists_without_duplicates(self):
        merged = dp.merge_extracted_data([{"items": [{"x": 1}]},
                                          {"items": [{"x": 1}, {"x": 2}]}])
        assert merged["items"] == [{"x": 1}, {"x": 2}]


# -- provenance --------------------------------------------------------------

class TestLocateValue:
    def test_finds_a_value_present_in_the_text(self, bill_document):
        evidence = dp.locate_value(bill_document, "RAMESH")
        assert evidence and evidence[0].score > 0.9
        assert evidence[0].bbox.page == 0

    def test_spans_a_multi_word_value(self, bill_document):
        evidence = dp.locate_value(bill_document, "RAMESH KUMAR SHARMA")
        assert evidence
        assert evidence[0].bbox.width > 40      # union of several word boxes

    def test_finds_a_number_despite_currency_and_grouping(self, bill_document):
        """The text prints 118250.00; the extracted value is Decimal 118250."""
        evidence = dp.locate_value(bill_document, decimal.Decimal("118250.00"))
        assert evidence and evidence[0].score > 0.85

    def test_finds_an_iso_date_printed_in_local_order(self, bill_document):
        """The prompt asks the model for ISO dates and the page prints
        DD-MM-YYYY.  Without matching across renderings, every correctly-read
        date scores as a hallucination."""
        evidence = dp.locate_value(bill_document, datetime.date(2024, 4, 12))
        assert evidence and evidence[0].score > 0.9
        assert "12-04-2024" in evidence[0].text

    def test_finds_an_iso_date_supplied_as_a_string(self, bill_document):
        assert dp.locate_value(bill_document, "2024-04-19")

    def test_a_date_is_not_matched_against_a_bare_year(self, bill_document):
        """'2024-04-12' parses as the amount 2024; treating it numerically
        would find spurious evidence anywhere a 2024 appears."""
        keys = dp._comparison_keys("2024-04-12")
        assert "2024" not in keys

    def test_absent_values_produce_no_evidence(self, bill_document):
        assert dp.locate_value(bill_document, "COMPLETELY ABSENT VALUE") == []

    def test_none_produces_no_evidence(self, bill_document):
        assert dp.locate_value(bill_document, None) == []

    def test_results_are_sorted_by_score(self, bill_document):
        evidence = dp.locate_value(bill_document, "Discharge", min_score=0.3)
        assert [e.score for e in evidence] == sorted([e.score for e in evidence],
                                                     reverse=True)

    def test_pages_filter_restricts_the_search(self, bill_document):
        assert dp.locate_value(bill_document, "RAMESH", pages=[7]) == []

    def test_backend_confidence_averages_over_supporting_spans(self, bill_document):
        evidence = dp.locate_value(bill_document, "RAMESH")
        assert dp.backend_confidence_for(bill_document, evidence) == pytest.approx(0.99)

    def test_backend_confidence_is_none_when_no_span_reports_one(self, bill_document):
        for page in bill_document.pages:
            for span in page.spans:
                span.confidence = None
        evidence = dp.locate_value(bill_document, "RAMESH")
        assert dp.backend_confidence_for(bill_document, evidence) is None


# -- validators --------------------------------------------------------------

class TestValidators:
    def _bill(self, total, items):
        return {"total": total,
                "line_items": [{"description": "x", "amount": a} for a in items]}

    def test_sum_check_passes(self):
        validator = dp.line_items_sum_to_total()
        assert validator(self._bill(300, [100, 200])) is None

    def test_sum_check_fails_and_names_the_fields(self):
        issue = dp.line_items_sum_to_total()(self._bill(999, [100, 200]))
        assert issue.severity == "error"
        assert set(issue.fields) == {"line_items", "total"}

    def test_sum_check_allows_rounding_tolerance(self):
        assert dp.line_items_sum_to_total()(self._bill("300.01", [100, "200.00"])) is None

    def test_sum_check_is_silent_when_there_is_nothing_to_check(self):
        assert dp.line_items_sum_to_total()({"total": None, "line_items": []}) is None

    def test_date_order(self):
        validator = dp.date_order("admission", "discharge")
        assert validator({"admission": datetime.date(2024, 4, 12),
                          "discharge": datetime.date(2024, 4, 19)}) is None
        issue = validator({"admission": datetime.date(2024, 4, 19),
                           "discharge": datetime.date(2024, 4, 12)})
        assert issue is not None and "after" in issue.message

    def test_field_matches(self):
        validator = dp.field_matches("bill_number", r"^INV-\d{4}")
        assert validator({"bill_number": "INV-2024-04-7781"}) is None
        assert validator({"bill_number": "nonsense"}) is not None

    def test_required_fields(self):
        issues = dp.required_fields("a", "b")({"a": "x", "b": None})
        assert len(issues) == 1 and issues[0].fields == ["b"]

    def test_a_raising_validator_is_reported_not_fatal(self):
        """A buggy business rule must not cost you the document."""
        def broken(model):
            raise ZeroDivisionError("oops")

        issues = dp.run_validators({}, [broken])
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "ZeroDivisionError" in issues[0].message

    def test_a_plain_string_return_becomes_an_error(self):
        issues = dp.run_validators({}, [lambda m: "something is wrong"])
        assert issues[0].severity == "error"

    def test_validators_work_on_objects_as_well_as_dicts(self):
        import dataclasses

        @dataclasses.dataclass
        class Bill:
            total: int

        assert dp.field_matches("total", r"^\d+$")(Bill(total=5)) is None


# -- extract() ---------------------------------------------------------------

SCHEMA = {
    "hospital_name": "string",
    "patient_name": "string",
    "admission_date": "date",
    "total": "number",
}

GOOD_RESPONSE = json.dumps({
    "hospital_name": "SUNRISE MULTISPECIALITY HOSPITAL",
    "patient_name": "RAMESH KUMAR SHARMA",
    "admission_date": "2024-04-12",
    "total": "118250.00",
})


class TestExtract:
    def test_end_to_end_with_a_canned_response(self, bill_document):
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(GOOD_RESPONSE))
        assert result.value("patient_name") == "RAMESH KUMAR SHARMA"
        assert result.value("total") == decimal.Decimal("118250.00")
        assert result.value("admission_date") == datetime.date(2024, 4, 12)

    def test_client_is_required(self, bill_document):
        with pytest.raises(dp.ConfigError) as exc:
            dp.extract(bill_document, SCHEMA)
        assert "EchoClient" in str(exc.value)

    def test_values_found_in_the_page_get_evidence(self, bill_document):
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(GOOD_RESPONSE))
        assert result["patient_name"].evidence
        assert result["patient_name"].evidence[0].page == 0

    def test_a_hallucinated_value_is_flagged_and_scored_low(self, bill_document):
        """It cannot be located anywhere in what was actually read."""
        response = json.dumps(dict(json.loads(GOOD_RESPONSE),
                                   patient_name="ENTIRELY INVENTED PERSON"))
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(response))
        field = result["patient_name"]
        assert any("hallucinated" in w for w in field.warnings)
        assert field.confidence < result["hospital_name"].confidence

    def test_missing_required_field_warns_and_scores_zero(self, bill_document):
        response = json.dumps({"hospital_name": None, "patient_name": None,
                               "admission_date": None, "total": None})
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(response))
        assert result["total"].value is None
        assert result["total"].confidence == 0.0
        assert any("required" in w for w in result["total"].warnings)

    def test_unparseable_output_yields_nulls_and_a_warning(self, bill_document):
        result = dp.extract(bill_document, SCHEMA,
                            client=dp.EchoClient("I am unable to read this."))
        assert all(f.value is None for f in result.fields.values())
        assert any("could not parse" in w for w in result.warnings)

    def test_validators_run_and_are_reported(self, bill_document):
        def always_fails(model):
            return dp.ValidationIssue("total looks wrong", fields=["total"])

        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(GOOD_RESPONSE),
                            validators=[always_fails])
        assert not result.is_valid
        assert result["total"].confidence < result["patient_name"].confidence

    def test_signals_are_recorded_for_auditing(self, bill_document):
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(GOOD_RESPONSE))
        signals = result["total"].signals
        assert "cross_read_agree" in signals and "page_quality" in signals

    def test_cost_is_accumulated(self, bill_document):
        client = dp.EchoClient(GOOD_RESPONSE, cost=dp.Cost(amount=0.03, calls=1))
        assert dp.extract(bill_document, SCHEMA, client=client).cost.amount == \
            pytest.approx(0.03)

    def test_low_confidence_lists_what_a_human_should_check(self, bill_document):
        response = json.dumps(dict(json.loads(GOOD_RESPONSE),
                                   patient_name="INVENTED"))
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(response))
        assert "patient_name" in [f.name for f in result.low_confidence(0.7)]

    def test_calibrator_is_applied_to_the_fused_score(self, bill_document):
        raw = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(GOOD_RESPONSE))
        calibrated = dp.extract(bill_document, SCHEMA,
                                client=dp.EchoClient(GOOD_RESPONSE),
                                calibrator=dp.PlattCalibrator(a=1.0, b=-3.0))
        assert calibrated["total"].confidence < raw["total"].confidence

    def test_retry_on_a_transient_client_failure(self, bill_document, monkeypatch):
        monkeypatch.setattr(dp.time, "sleep", lambda s: None)
        state = {"calls": 0}

        class Flaky(dp.EchoClient):
            def complete(self, prompt, system=None, images=None):
                state["calls"] += 1
                if state["calls"] == 1:
                    raise RuntimeError("gateway timeout")
                return GOOD_RESPONSE, dp.Cost(calls=1)

        result = dp.extract(bill_document, SCHEMA, client=Flaky(), retries=2)
        assert result.value("total") == decimal.Decimal("118250.00")
        assert state["calls"] == 2

    def test_extra_keys_in_the_response_are_kept(self, bill_document):
        response = json.dumps(dict(json.loads(GOOD_RESPONSE), surprise="extra"))
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(response))
        assert result.model["surprise"] == "extra"

    def test_a_list_response_takes_the_first_object(self, bill_document):
        response = json.dumps([json.loads(GOOD_RESPONSE)])
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(response))
        assert result.value("patient_name") == "RAMESH KUMAR SHARMA"

    def test_extraction_serialises(self, bill_document):
        result = dp.extract(bill_document, SCHEMA, client=dp.EchoClient(GOOD_RESPONSE))
        payload = json.loads(dp.to_json(result.to_dict()))
        assert payload["fields"]["total"]["value"] == 118250.0
        assert payload["valid"] is True


class TestEchoClient:
    def test_records_prompts(self, bill_document):
        client = dp.EchoClient(GOOD_RESPONSE)
        dp.extract(bill_document, SCHEMA, client=client)
        assert len(client.prompts) == 1
        assert "SUNRISE" in client.prompts[0]

    def test_callable_responses(self):
        client = dp.EchoClient(lambda prompt: {"seen": len(prompt)})
        text, _ = client.complete("abcd")
        assert json.loads(text)["seen"] == 4

    def test_dict_responses_are_serialised(self):
        text, _ = dp.EchoClient({"a": 1}).complete("x")
        assert json.loads(text) == {"a": 1}


# -- rule-based extraction ---------------------------------------------------

class TestRuleExtraction:
    def test_reads_a_labelled_value_to_the_right(self, bill_document):
        rules = [dp.FieldRule(name="patient_name", label=r"Patient\s*Name",
                              value_pattern=r"[:\s]*([A-Z][A-Z\s]{4,})")]
        results = dp.extract_with_rules(bill_document, rules)
        assert "RAMESH" in results["patient_name"].value

    def test_typed_rules_coerce_their_value(self, bill_document):
        rules = [dp.FieldRule(name="total", label=r"Total\s*Amount\s*Payable",
                              value_pattern=r"([\d,\.]+)", type_name="number")]
        assert dp.extract_with_rules(bill_document, rules)["total"].value == \
            decimal.Decimal("118250.00")

    def test_a_date_rule(self, bill_document):
        rules = [dp.FieldRule(name="admitted", label=r"Admission\s*Date",
                              value_pattern=r"([\d\-/]{8,10})", type_name="date")]
        assert dp.extract_with_rules(bill_document, rules)["admitted"].value == \
            datetime.date(2024, 4, 12)

    def test_evidence_and_confidence_are_produced(self, bill_document):
        rules = [dp.FieldRule(name="bill_number", label=r"Bill\s*No",
                              value_pattern=r"(INV-[\d\-]+)")]
        result = dp.extract_with_rules(bill_document, rules)["bill_number"]
        assert result.evidence and result.confidence > 0.5
        # A single read cannot claim cross-read agreement.
        assert result.signals["cross_read_agree"] is None

    def test_a_missing_label_is_reported(self, bill_document):
        rules = [dp.FieldRule(name="x", label=r"NoSuchLabelAnywhere")]
        result = dp.extract_with_rules(bill_document, rules)["x"]
        assert result.value is None
        assert any("not found" in w for w in result.warnings)
