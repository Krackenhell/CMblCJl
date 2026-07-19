from __future__ import annotations

from .models import Evidence, Recommendation, Route, Skill


def choose_route(skill: Skill, mastery: float, evidence: Evidence | None) -> Recommendation:
    if evidence and evidence.confidence < 0.58:
        return Recommendation(
            route=Route.HUMAN_REVIEW,
            title="Нужна проверка преподавателя",
            action="Посмотреть ответ и подтвердить уровень освоения без автоматического штрафа.",
            duration_minutes=3,
            skill_id=skill.id,
        )
    if mastery < 0.45:
        return Recommendation(
            route=Route.REPAIR,
            title=f"Восстановить: {skill.name}",
            action="Разобрать контрпример, исправить один фрагмент работы и пройти повторный вопрос.",
            duration_minutes=10,
            skill_id=skill.id,
        )
    if mastery < skill.target_mastery:
        return Recommendation(
            route=Route.PRACTICE,
            title=f"Закрепить: {skill.name}",
            action="Решить одно задание с изменёнными условиями и объяснить выбор решения.",
            duration_minutes=8,
            skill_id=skill.id,
        )
    return Recommendation(
        route=Route.TRANSFER,
        title=f"Усложнить: {skill.name}",
        action="Найти аналогичный риск в новом датасете и защитить предложенный эксперимент.",
        duration_minutes=12,
        skill_id=skill.id,
    )
