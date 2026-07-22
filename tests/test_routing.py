from meaning_trainer.models import Evidence, Route, Skill
from meaning_trainer.routing import choose_route


SKILL = Skill(id="x", name="Навык", description="", target_mastery=0.8)


def test_low_confidence_goes_to_human() -> None:
    evidence = Evidence("x", 0.5, 0.4, "", "")
    assert choose_route(SKILL, 0.5, evidence).route == Route.HUMAN_REVIEW


def test_mastery_routes() -> None:
    assert choose_route(SKILL, 0.3, None).route == Route.REPAIR
    assert choose_route(SKILL, 0.6, None).route == Route.PRACTICE
    assert choose_route(SKILL, 0.9, None).route == Route.TRANSFER

