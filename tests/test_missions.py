from __future__ import annotations

import pytest

from vivatrace.missions import (
    detect_mission_features,
    grounded_error_items,
    load_missions,
    missions_by_topic,
    validate_mission_evaluation,
)


@pytest.fixture(scope="module")
def mission_bank() -> list[dict]:
    return load_missions()


def test_mission_bank_has_one_valid_scenario_for_every_english_topic(mission_bank):
    by_topic = missions_by_topic(mission_bank)

    assert len(mission_bank) == 10
    assert len(by_topic) == 10
    assert all(mission["max_turns"] == 3 for mission in mission_bank)
    assert all(len(mission["required_features"]) == 2 for mission in mission_bank)


@pytest.mark.parametrize(
    ("topic_key", "answer"),
    [
        ("eng_articles", "I saw a black suitcase. The suitcase belongs to Anna."),
        ("eng_present_perfect", "I have worked in EdTech for two years. I started in 2024."),
        ("eng_modals_deduction", "It must be Lena's bag, but it might belong to her colleague."),
        ("eng_gerund_infinitive", "I stopped smoking and decided to join a gym."),
        ("eng_conditionals", "If we finish today, we will ship. If I had more time, I would test it."),
        ("eng_passive", "The report was published. A correction will be added tomorrow."),
        ("eng_reported_speech", "Maya asked me to send it that day."),
        ("eng_relative_clauses", "The woman who called works at the café where we met."),
        ("eng_linking", "However, it is expensive. Therefore, we should wait."),
        ("eng_formal_writing", "Would it be possible to move it due to a medical appointment?"),
    ],
)
def test_feature_detector_recognizes_required_language(mission_bank, topic_key, answer):
    mission = missions_by_topic(mission_bank)[topic_key]

    signal = detect_mission_features(mission, [answer])

    assert signal["coverage"] == 1
    assert not signal["missing"]
    assert all(item["evidence"] for item in signal["features"])


def test_feature_detector_does_not_reward_irrelevant_confident_text(mission_bank):
    mission = missions_by_topic(mission_bank)["eng_modals_deduction"]

    signal = detect_mission_features(mission, ["I definitely know the answer. This is obvious."])

    assert signal["coverage"] == 0
    assert len(signal["missing"]) == 2


def test_llm_error_must_quote_the_students_literal_answer():
    verified, discarded = grounded_error_items(
        [
            {
                "fragment": "I stopped smoke",
                "explanation": "После stop нужна форма -ing.",
                "correction": "I stopped smoking.",
            },
            {
                "fragment": "I never wrote this",
                "explanation": "Выдуманная ошибка.",
                "correction": "—",
            },
        ],
        "I stopped smoke last month and decided to exercise.",
    )

    assert [item["fragment"] for item in verified] == ["I stopped smoke"]
    assert discarded == 1


def test_mission_cannot_complete_without_all_deterministic_features(mission_bank):
    mission = missions_by_topic(mission_bank)["eng_articles"]
    answer = "I found a suitcase near reception."
    signal = detect_mission_features(mission, [answer])

    result = validate_mission_evaluation(
        mission,
        {
            "npc_reply": "Thank you. Which suitcase do you mean?",
            "positive_feedback": "Сообщение понятно.",
            "guidance": "Уточните уже упомянутый предмет.",
            "errors": [],
            "grammar_score": 1,
            "communicative_score": 1,
            "state_summary": "Чемодан пока не идентифицирован.",
            "suggested_next_action": "Использовать the.",
            "ready_to_finish": True,
        },
        answer,
        signal,
        1,
    )

    assert result["status"] == "active"
    assert result["success"] is False
    assert result["score"] <= 0.74


def test_mission_completes_only_on_grounded_hybrid_score(mission_bank):
    mission = missions_by_topic(mission_bank)["eng_articles"]
    answer = "I found a suitcase. The suitcase has a blue tag."
    signal = detect_mission_features(mission, [answer])

    result = validate_mission_evaluation(
        mission,
        {
            "npc_reply": "Great, I can identify it now.",
            "positive_feedback": "Оба артикля применены по задаче.",
            "guidance": "Миссия завершена.",
            "errors": [],
            "grammar_score": 0.95,
            "communicative_score": 0.9,
            "state_summary": "Владелец и чемодан установлены.",
            "suggested_next_action": "Завершить диалог.",
            "ready_to_finish": True,
        },
        answer,
        signal,
        2,
    )

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["score"] >= mission["success_threshold"]


def test_invalid_first_mention_article_correction_is_filtered():
    verified, discarded = grounded_error_items(
        [
            {
                "fragment": "The suitcase has a blue tag.",
                "explanation": "Use the definite article for a specific tag.",
                "correction": "The suitcase has the blue tag.",
            }
        ],
        "I found a suitcase. The suitcase has a blue tag.",
    )

    assert verified == []
    assert discarded == 1


@pytest.mark.parametrize(
    ("topic_key", "answer", "error"),
    [
        (
            "eng_gerund_infinitive",
            "I want to stop checking my phone, and I plan to study English instead.",
            {
                "fragment": "I want to stop checking my phone, and I plan to study English instead.",
                "explanation": "stop checking instead of stop doing, and plan to instead of decide to.",
                "correction": "I want to stop checking my phone, and I plan to decide to study English instead.",
            },
        ),
        (
            "eng_relative_clauses",
            "The library is a place where you can study quietly.",
            {
                "fragment": "The library is a place where you can study quietly.",
                "explanation": "Use a non-defining clause instead.",
                "correction": "The library, where you can study quietly, is the perfect place for you.",
            },
        ),
    ],
)
def test_valid_target_form_is_not_rejected_as_a_stylistic_preference(
    mission_bank, topic_key, answer, error
):
    mission = missions_by_topic(mission_bank)[topic_key]

    verified, discarded = grounded_error_items([error], answer, mission)

    assert verified == []
    assert discarded == 1


@pytest.mark.parametrize(
    ("topic_key", "answer", "error"),
    [
        (
            "eng_gerund_infinitive",
            "I want to stop check my phone and I plan to studying English.",
            {
                "fragment": "stop check",
                "explanation": "После stop для прекращения действия нужна форма -ing.",
                "correction": "stop checking",
            },
        ),
        (
            "eng_relative_clauses",
            "The library where is quiet closes at ten.",
            {
                "fragment": "The library where is quiet",
                "explanation": "После where нужна полноценная придаточная часть.",
                "correction": "The library where you can study is quiet",
            },
        ),
    ],
)
def test_subject_guard_keeps_real_local_grammar_errors(
    mission_bank, topic_key, answer, error
):
    mission = missions_by_topic(mission_bank)[topic_key]

    verified, discarded = grounded_error_items([error], answer, mission)

    assert verified == [error]
    assert discarded == 0


@pytest.mark.parametrize(
    "error",
    [
        {
            "fragment": "The suitcase",
            "explanation": "Здесь уже нужен определённый артикль.",
            "correction": "the suitcase",
        },
        {
            "fragment": "However, live discussion is important.",
            "explanation": "However, live discussion is important. As a result, a mixed format may work best.",
            "correction": "However, live discussion is important. As a result, a mixed format may work best.",
        },
    ],
)
def test_noop_or_malformed_correction_is_not_shown_as_an_error(error):
    answer = (
        "I lost a black suitcase. The suitcase has a red tag. "
        "However, live discussion is important. As a result, a mixed format may work best."
    )

    verified, discarded = grounded_error_items([error], answer)

    assert verified == []
    assert discarded == 1
