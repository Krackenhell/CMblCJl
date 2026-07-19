from vivatrace.bkt import BKTModel, BKTParameters


def test_strong_evidence_increases_mastery() -> None:
    model = BKTModel(BKTParameters(learn=0.05))
    assert model.update(0.35, 0.95) > 0.35


def test_wrong_evidence_can_reduce_high_mastery() -> None:
    model = BKTModel(BKTParameters(learn=0.05))
    assert model.update(0.85, 0.05) < 0.85


def test_fractional_update_stays_bounded() -> None:
    model = BKTModel()
    assert 0 <= model.update(0.4, 0.5) <= 1

