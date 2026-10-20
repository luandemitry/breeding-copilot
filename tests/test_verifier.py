import pytest

from app.verifier import collect_numbers, extract_numbers, is_supported, redact, unsupported_numbers

RESULTS = [
    {"rows": [["L042", 6.384, 3], ["L007", 6.1, 4]], "row_count": 2, "truncated": False},
    {"predictions": [{"p10": 4.21, "p50": 5.03, "p90": 6.5, "interval_width": 2.29}]},
]


def test_extract_numbers_ignores_identifiers_and_list_markers():
    text = "1. L042 in Cerrado-2027 yields 6.4 t/ha (marker m12, p90 = 7.1)\n2) F3 has 3 lines"
    assert extract_numbers(text) == ["6.4", "7.1", "3"]


def test_extract_numbers_handles_decimal_comma_and_negatives():
    assert extract_numbers("media de 6,38 e delta -0.5") == ["6.38", "-0.5"]


def test_collect_numbers_walks_nested_structures_and_skips_bools():
    assert sorted(collect_numbers({"a": [1, {"b": 2.5}], "c": True, "d": "7"})) == [1.0, 2.5]


@pytest.mark.parametrize(
    ("number", "supported"),
    [("6.38", True), ("6.4", True), ("6", True), ("6.39", False), ("6.5", False), ("5.03", True), ("5.0", True)],
)
def test_is_supported_rounding(number, supported):
    assert is_supported(number, [6.384, 5.03]) is supported


def test_faithful_answer_has_no_unsupported_numbers():
    answer = "L042 averages 6.38 t/ha; predicted 5.03 (p10 4.21, p90 6.5), 2 lines found. 80% interval."
    assert unsupported_numbers(answer, RESULTS) == []


def test_fabricated_number_is_flagged():
    answer = "L042 averages 7.9 t/ha, 6.38 elsewhere, and 12 trials."
    assert unsupported_numbers(answer, RESULTS) == ["7.9", "12"]


def test_numbers_from_the_question_are_allowed():
    assert unsupported_numbers("Here are the top 5 lines", [], question="Which 5 lines are best?") == []
    assert unsupported_numbers("Here are the top 6 lines", [], question="Which 5 lines are best?") == ["6"]


def test_answer_with_numbers_but_no_tool_results_is_flagged():
    assert unsupported_numbers("Yield is 6.2", []) == ["6.2"]


def test_numberless_answer_is_always_faithful():
    assert unsupported_numbers("L042 is the best line in Cerrado-2027.", []) == []


def test_redact_replaces_only_listed_numbers():
    out = redact("Yield is 7.9 vs 6.38; Cerrado-2027 had 12 trials, L012 too", ["7.9", "12"])
    assert out == "Yield is [unverified] vs 6.38; Cerrado-2027 had [unverified] trials, L012 too"


def test_redact_does_not_break_longer_numbers():
    assert redact("values 12.5 and 12", ["12"]) == "values 12.5 and [unverified]"


def test_model_version_strings_are_identifiers_not_numbers():
    text = "Predicted 5.03 t/ha with model version qgbr-2714e999 for L042 in Cerrado-2027."
    assert extract_numbers(text) == ["5.03"]
