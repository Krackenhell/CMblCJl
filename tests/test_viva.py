from vivatrace.models import ArtifactFinding
from vivatrace.viva import follow_up_question, select_questions


def test_questions_follow_findings_and_assignment_skills():
    finding = ArtifactFinding(
        skill_id="data_leakage",
        severity="high",
        evidence="fit_transform вызван до split",
        hypothesis="Возможна утечка",
        confidence=0.95,
    )

    questions = select_questions(
        [finding],
        {"data_leakage": 0.35, "metrics": 0.35},
        limit=2,
        allowed_skills=["data_leakage", "metrics"],
        seed_key="student-one",
    )

    assert questions[0].skill_id == "data_leakage"
    assert {question.skill_id for question in questions} == {"data_leakage", "metrics"}


def test_weak_answer_can_receive_a_different_follow_up():
    question = select_questions(
        [],
        {"cross_validation": 0.2},
        limit=1,
        allowed_skills=["cross_validation"],
        seed_key="student-two",
    )[0]

    follow_up = follow_up_question(question, seed_key="student-two")

    assert follow_up is not None
    assert follow_up.skill_id == question.skill_id
    assert follow_up.id != question.id
