"""Script, digit, date and amount normalisation.

Domain-independent but not culture-independent.  Indian digit grouping
(``1,23,456.78``) breaks every parser that assumes groups of three, and it is
the default on the documents this library targets -- so it gets the most tests
here.  An ambiguous ``03/04/2024`` silently read as March 4th is the kind of
error that reaches a claims reviewer looking entirely plausible.
"""

import datetime
import decimal

import pytest

import docpipe as dp


class TestScriptDetection:
    @pytest.mark.parametrize("text,script", [
        ("Hospital", "Latn"),
        ("अस्पताल", "Deva"),
        ("மருத்துவமனை", "Taml"),
        ("ಆಸ್ಪತ್ರೆ", "Knda"),
        ("হাসপাতাল", "Beng"),
        ("مستشفى", "Arab"),
        ("больница", "Cyrl"),
        ("医院", "Hans"),
    ])
    def test_detects_the_dominant_script(self, text, script):
        assert dp.detect_script(text) == script

    def test_digits_and_punctuation_carry_no_script(self):
        assert dp.detect_script("12,345.00 -/-") is None
        assert dp.detect_script("") is None

    def test_mixed_text_returns_the_majority(self):
        assert dp.detect_script("अस्पताल Hospital अस्पताल अस्पताल") == "Deva"


class TestDigitNormalisation:
    @pytest.mark.parametrize("text,expected", [
        ("४८२५०", "48250"),          # Devanagari
        ("١٢٣٤٥", "12345"),          # Arabic-Indic
        ("۱۲۳", "123"),              # Extended Arabic-Indic
        ("১২৩", "123"),              # Bengali
        ("௧௨௩", "123"),              # Tamil
        ("１２３", "123"),             # Fullwidth
        ("48250", "48250"),          # already ASCII
    ])
    def test_converts_to_ascii(self, text, expected):
        assert dp.normalize_digits(text) == expected

    def test_leaves_letters_alone(self):
        assert dp.normalize_digits("Total ४८२५० INR") == "Total 48250 INR"


class TestTextNormalisation:
    def test_collapses_whitespace_but_keeps_lines(self):
        assert dp.normalize_whitespace("a   b\n\n\n\nc  d") == "a b\n\nc d"

    def test_can_collapse_newlines_too(self):
        assert dp.normalize_whitespace("a\nb", collapse_newlines=True) == "a b"

    def test_handles_non_breaking_and_zero_width_spaces(self):
        assert dp.normalize_whitespace("a ​b") == "a b"

    def test_nfkc_folds_compatibility_forms(self):
        assert dp.normalize_unicode("ﬁle") == "file"

    def test_normalize_text_does_all_three(self):
        assert dp.normalize_text("Total  ४८२५० INR") == "Total 48250 INR"


class TestOCRConfusions:
    def test_numeric_repairs(self):
        assert dp.fix_ocr_confusions("l23O5", expect="numeric") == "12305"

    def test_alpha_repairs(self):
        assert dp.fix_ocr_confusions("S0NIA", expect="alpha") == "SONIA"

    def test_rejects_an_unknown_expectation(self):
        with pytest.raises(dp.ConfigError):
            dp.fix_ocr_confusions("abc", expect="vibes")

    def test_empty_input_is_returned_unchanged(self):
        assert dp.fix_ocr_confusions("", expect="numeric") == ""


class TestParseAmount:
    @pytest.mark.parametrize("text,expected", [
        ("48250", "48250"),
        ("48,250", "48250"),
        ("48,250.75", "48250.75"),
        ("1,23,456.78", "123456.78"),      # Indian grouping
        ("12,34,56,789", "123456789"),     # crore grouping
        ("Rs. 1,23,456.78", "123456.78"),
        ("₹48,250.00", "48250.00"),
        ("INR 48250/-", "48250"),
        ("  48250.00  ", "48250.00"),
        ("0.50", "0.50"),
    ])
    def test_indian_and_plain_formats(self, text, expected):
        assert dp.parse_amount(text) == decimal.Decimal(expected)

    def test_european_convention_is_detected_by_separator_position(self):
        assert dp.parse_amount("1.234,56") == decimal.Decimal("1234.56")

    def test_us_convention(self):
        assert dp.parse_amount("1,234.56") == decimal.Decimal("1234.56")

    def test_accounting_parentheses_are_negative(self):
        assert dp.parse_amount("(1,234.00)") == decimal.Decimal("-1234.00")

    def test_explicit_minus(self):
        assert dp.parse_amount("-500.25") == decimal.Decimal("-500.25")

    def test_debit_and_credit_markers(self):
        assert dp.parse_amount("1,200.00 DR") == decimal.Decimal("-1200.00")
        assert dp.parse_amount("1,200.00 CR") == decimal.Decimal("1200.00")

    def test_devanagari_digits(self):
        assert dp.parse_amount("₹ ४८,२५०.००") == decimal.Decimal("48250.00")

    def test_returns_none_when_there_is_no_number(self):
        assert dp.parse_amount("no amount here") is None
        assert dp.parse_amount("") is None
        assert dp.parse_amount(None) is None

    def test_returns_decimal_not_float(self):
        """Money and binary floating point should not meet."""
        assert isinstance(dp.parse_amount("0.1"), decimal.Decimal)
        assert dp.parse_amount("0.1") + dp.parse_amount("0.2") == decimal.Decimal("0.3")

    def test_explicit_separator_overrides_inference(self):
        assert dp.parse_amount("1.234", decimal_separator="comma") == decimal.Decimal("1234")
        assert dp.parse_amount("1,234", decimal_separator="dot") == decimal.Decimal("1234")

    def test_typographic_spaces_group_digits(self):
        assert dp.parse_amount("1\u2009234\u2009567") == decimal.Decimal("1234567")
        assert dp.parse_amount("1\u00a0234\u00a0567") == decimal.Decimal("1234567")

    def test_a_plain_space_does_not_glue_two_numbers_together(self):
        """'12 500' in a table row is as likely to be two cells as one
        amount; inventing 12500 from it would be worse than reading 12."""
        assert dp.parse_amount("12 500") == decimal.Decimal("12")


class TestDetectCurrency:
    @pytest.mark.parametrize("text,code", [
        ("₹48,250", "INR"), ("Rs. 500", "INR"), ("INR 500", "INR"),
        ("$1,200", "USD"), ("€900", "EUR"), ("£40", "GBP"),
    ])
    def test_symbols_and_codes(self, text, code):
        assert dp.detect_currency(text) == code

    def test_no_currency(self):
        assert dp.detect_currency("48250") is None


class TestParseDate:
    @pytest.mark.parametrize("text,expected", [
        ("12-04-2024", (2024, 4, 12)),        # DD-MM-YYYY, the local default
        ("12/04/2024", (2024, 4, 12)),
        ("2024-04-12", (2024, 4, 12)),        # ISO always wins
        ("31/12/2023", (2023, 12, 31)),       # unambiguous: day > 12
        ("12 Apr 2024", (2024, 4, 12)),
        ("12-April-2024", (2024, 4, 12)),
        ("April 12, 2024", (2024, 4, 12)),
        ("Apr 12 2024", (2024, 4, 12)),
        ("12042024", (2024, 4, 12)),
    ])
    def test_common_formats(self, text, expected):
        assert dp.parse_date(text) == datetime.date(*expected)

    def test_dayfirst_default_resolves_ambiguity_the_local_way(self):
        assert dp.parse_date("03/04/2024") == datetime.date(2024, 4, 3)

    def test_dayfirst_can_be_turned_off(self):
        assert dp.parse_date("03/04/2024", dayfirst=False) == datetime.date(2024, 3, 4)

    def test_unambiguous_input_ignores_the_flag(self):
        assert dp.parse_date("25/04/2024", dayfirst=False) == datetime.date(2024, 4, 25)

    def test_two_digit_years_expand_to_the_recent_past(self):
        parsed = dp.parse_date("12-04-24")
        assert parsed.year == 2024 and parsed.month == 4 and parsed.day == 12

    def test_devanagari_digits(self):
        assert dp.parse_date("१२-०४-२०२४") == datetime.date(2024, 4, 12)

    def test_finds_a_date_inside_a_sentence(self):
        assert dp.parse_date("Admission Date : 12-04-2024 ward B") == \
            datetime.date(2024, 4, 12)

    def test_returns_none_rather_than_guessing(self):
        assert dp.parse_date("no date here") is None
        assert dp.parse_date("") is None
        assert dp.parse_date(None) is None

    def test_impossible_dates_are_rejected(self):
        assert dp.parse_date("32-13-2024") is None

    def test_out_of_range_years_are_rejected(self):
        assert dp.parse_date("12-04-1204") is None


class TestParseBool:
    @pytest.mark.parametrize("text,expected", [
        ("yes", True), ("Y", True), ("true", True), ("1", True), ("X", True),
        ("✓", True), ("no", False), ("N", False), ("false", False), ("0", False),
        ("-", False),
    ])
    def test_checkbox_values(self, text, expected):
        assert dp.parse_bool(text) is expected

    def test_unrecognised_is_none(self):
        assert dp.parse_bool("maybe") is None


class TestSpanNormalisation:
    def test_normalises_text_and_tags_scripts(self):
        spans = [dp.TextSpan("४८२५०", dp.BBox(0, 0, 0, 10, 10)),
                 dp.TextSpan("अस्पताल", dp.BBox(0, 0, 20, 10, 30))]
        out = dp.normalize_spans(spans)
        assert out[0].text == "48250"
        assert out[1].script == "Deva"

    def test_does_not_mutate_the_input(self):
        spans = [dp.TextSpan("४८२५०", dp.BBox(0, 0, 0, 10, 10))]
        dp.normalize_spans(spans)
        assert spans[0].text == "४८२५०"

    def test_normalize_document_covers_every_page(self):
        doc = dp.Document(pages=[
            dp.Page(index=i, spans=[dp.TextSpan("४८२५०", dp.BBox(i, 0, 0, 1, 1))])
            for i in range(2)])
        out = dp.normalize_document(doc)
        assert all(p.spans[0].text == "48250" for p in out.pages)
