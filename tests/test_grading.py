from __future__ import annotations

import json
from pathlib import Path

import pytest

from vivatrace.database import get_assignment, init_database, list_assignments
from vivatrace.grading import (
    accepted_variants,
    grade_numbered_answer,
    grade_structured_answer,
    normalize_answer,
    parse_numbered_items,
)


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setenv("VIVATRACE_DB_PATH", str(tmp_path / "grading.db"))
    init_database()


def test_modal_attempt_scores_three_correct_and_one_blank():
    assignment = get_assignment(6)

    result = grade_structured_answer(
        assignment,
        "1) can't\n2) must\n3) might\n4)",
    )

    assert result is not None
    assert result["score"] == 0.75
    assert result["correct"] is False
    assert [slot["correct"] for slot in result["slots"]] == [True, True, True, False]
    assert result["slots"][3]["expected_phrase"] == "must have"


def test_modal_key_accepts_explicit_alternatives():
    assignment = get_assignment(6)

    with_might = grade_structured_answer(
        assignment, "1) cannot\n2) must\n3) might\n4) must have"
    )
    with_could = grade_structured_answer(
        assignment, "1) can't\n2) must\n3) could because I am unsure\n4) must have"
    )

    assert with_might is not None and with_might["score"] == 1
    assert with_could is not None and with_could["score"] == 1


def test_modal_key_rejects_reversed_deductions():
    assignment = get_assignment(6)

    result = grade_structured_answer(
        assignment,
        "1) must\n2) can't\n3) must\n4) might have",
    )

    assert result is not None
    assert result["score"] == 0
    assert result["correct"] is False


def test_contractions_are_normalized_without_llm_judgement():
    assignment = get_assignment(12)

    result = grade_structured_answer(
        assignment,
        "1) have not seen\n2) called\n3) have already sent\n4) moved",
    )

    assert result is not None
    assert result["score"] == 1


def test_numbered_parser_does_not_split_years_or_explanations():
    items = parse_numbered_items(
        "1) The vaccine was developed in 2020.\n"
        "2) A new system is being tested.\n"
        "3) My bicycle has been stolen."
    )

    assert len(items) == 3
    assert items[0].endswith("2020")


def test_legacy_starter_suffix_does_not_corrupt_last_answer():
    assignment = get_assignment(8)

    result = grade_structured_answer(
        assignment,
        "1) meeting\n2) to send\n3) smoking\n4) to buy Meaning difference:",
    )

    assert result is not None
    assert result["score"] == 1
    assert result["slots"][3]["prompt"] == "He stopped ___ (buy) some water"


def test_reported_speech_near_misses_receive_component_scores():
    assignment = get_assignment(5)
    answer = (
        "1) Maya said she was working from home yesterday\n"
        "2) Leo said he finished the task the day before yesterday\n"
        "3) Nina asked if I have seen this document"
    )

    result = grade_structured_answer(assignment, answer)

    assert result is not None
    assert 0.55 <= result["score"] <= 0.70
    assert [slot["diagnostic"]["code"] for slot in result["slots"]] == [
        "reference_shift",
        "backshift",
        "backshift",
    ]


def test_reported_request_diagnoses_reporting_construction_not_generic_backshift():
    assignments = list_assignments()
    assignment = next(
        item
        for item in assignments
        if item.get("topic_key") == "eng_reported_speech" and item.get("variant") == 2
    )

    result = grade_structured_answer(
        assignment,
        "1) Ada said to Ben to send her that file\n2)\n3)",
    )

    assert result is not None
    first = result["slots"][0]
    assert 0.60 <= first["score"] <= 0.75
    assert first["diagnostic"]["code"] == "request_reporting_verb"
    assert "ask + object + to-infinitive" in first["diagnostic"]["rule_focus"]


def test_reference_slashes_expand_to_real_answers():
    assert accepted_variants("might/could have") == ["might have", "could have"]
    assert set(accepted_variants("She asked whether/if I knew.")) >= {
        "She asked whether I knew.",
        "She asked if I knew.",
    }


def test_open_writing_task_is_left_to_llm():
    assignment = get_assignment(10)

    assert grade_numbered_answer(assignment, "A coherent paragraph.") is None


def test_normalization_preserves_semantic_tokens():
    assert normalize_answer("She hasn't finished.") == normalize_answer(
        "She has not finished"
    )


def test_every_numbered_assignment_accepts_its_canonical_key():
    checked = 0
    for assignment in list_assignments():
        reference = str((assignment.get("rubric") or {}).get("reference_answer") or "")
        items = parse_numbered_items(reference)
        if not items:
            continue
        answer = "\n".join(
            f"{index}) {accepted_variants(item)[0]}"
            for index, item in enumerate(items, start=1)
        )
        result = grade_numbered_answer(assignment, answer)
        assert result is not None
        assert result["score"] == 1, assignment["title"]
        checked += 1

    assert checked >= 19


def test_adversarial_benchmark_has_no_false_scores():
    benchmark_path = Path(__file__).parents[1] / "data" / "grading_benchmark.json"
    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))

    for case in cases:
        result = grade_structured_answer(
            get_assignment(int(case["assignment_id"])), str(case["answer"])
        )
        assert result is not None, case["name"]
        assert float(result["score"]) == pytest.approx(
            float(case["expected_score"]), abs=1e-6
        ), case["name"]

    assert len(cases) >= 24


def test_assignment_bank_has_90_unique_titles_and_nine_variants_per_topic():
    path = Path(__file__).parents[1] / "data" / "english_b2_assignments.json"
    assignments = json.loads(path.read_text(encoding="utf-8"))

    assert len(assignments) == 90
    assert len({item["title"] for item in assignments}) == 90
    topics = {item["topic_key"] for item in assignments}
    assert len(topics) == 10
    for topic in topics:
        variants = sorted(
            item["variant"] for item in assignments if item["topic_key"] == topic
        )
        assert variants == list(range(1, 10))
