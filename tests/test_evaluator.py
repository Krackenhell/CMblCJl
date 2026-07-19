from vivatrace.evaluator import RubricEvaluator
from vivatrace.viva import QUESTION_BANK


def test_good_answer_scores_higher_than_misconception() -> None:
    question = QUESTION_BANK["data_leakage"][0]
    good = RubricEvaluator().evaluate(
        question,
        "Статистики test попадут в scaling. Нужно сначала разделить и fit только train через Pipeline.",
    )
    bad = RubricEvaluator().evaluate(
        question,
        "Это безопасно, лучше обучить на всех данных, потому что так среднее точнее.",
    )
    assert good.score > bad.score
    assert bad.misconception == "test_can_train_preprocessing"

