from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from html import escape
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from vivatrace.bkt import combine_mastery_evidence
from vivatrace.cohort import mastery_frame
from vivatrace.curriculum import load_curriculum
from vivatrace.database import (
    create_assignment,
    get_mastery,
    init_database,
    latest_topic_attempts,
    latest_mission_attempt,
    latest_voice_topic_sessions,
    list_assignments,
    list_students,
    mission_history,
    reset_learning_data,
    save_attempt,
    save_mission_turn,
    start_mission_attempt,
    student_attempts,
    student_history,
    student_progress,
    student_voice_sessions,
    update_assignment,
    update_mastery,
)
from vivatrace.demo import DATA_DIR
from vivatrace.grading import grade_structured_answer, parse_numbered_items
from vivatrace.local_llm import LLMTrace, LocalLLM, LocalLLMError
from vivatrace.missions import detect_mission_features, load_missions, missions_by_topic
from vivatrace.models import ArtifactFinding, Curriculum, Evidence, StudentState
from vivatrace.rulebook import load_rulebook
from vivatrace.review import build_review_plan
from vivatrace.voice import ensure_voice_server, voice_component_html


ROOT = Path(__file__).resolve().parent
CURRICULUM = load_curriculum(DATA_DIR / "curriculum.json")
LLM = LocalLLM()
ASSESSMENT_VERSION = 8
RELATIVE_CLAUSE_GRADER_VERSION = 2
RULEBOOK = load_rulebook()
MISSIONS = load_missions()
MISSIONS_BY_TOPIC = missions_by_topic(MISSIONS)
DIFFICULTY_LABELS = {1: "Базовый", 2: "Средний", 3: "Продвинутый"}
COLORS = {
    "green": "#176B50",
    "lime": "#C9F26B",
    "amber": "#F4B860",
    "coral": "#EC7565",
}

st.set_page_config(
    page_title="VivaTrace",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
.stApp { background: #F5F6F1; color: #18231F; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stSidebarCollapseButton"] {
  display: none !important;
}
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1420px; }
[data-testid="stSidebar"] { background: #102D23; border-right: 0; }
[data-testid="stSidebar"] * { color: #F4F8F5 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {
  color: #18231F !important;
}
h1, h2, h3 { color: #18231F; letter-spacing: -0.035em; }
.brand { font-size: 1.35rem; font-weight: 800; letter-spacing: -.03em; margin: .3rem 0 .15rem; }
.brand-sub { color: #AFC2BA; font-size: .78rem; margin-bottom: 1.6rem; }
.hero { border-radius: 24px; padding: 28px 32px; color: white; margin-bottom: 22px;
  background: radial-gradient(circle at 88% 18%, rgba(201,242,107,.32), transparent 25%),
              linear-gradient(120deg, #123D2F, #1A7053); box-shadow: 0 14px 40px rgba(23,61,48,.13); }
.hero .eyebrow { color: #C9F26B; font-weight: 800; font-size: .76rem; letter-spacing: .11em; }
.hero h1 { color: white; margin: 7px 0; font-size: 2.2rem; }
.hero p { color: #E7F1EC; max-width: 920px; margin: 0; line-height: 1.55; }
.card, .metric-card { background: white; border: 1px solid #E2E7E2; border-radius: 18px; padding: 18px 20px; margin-bottom: 13px; }
.metric-card { min-height: 118px; }
.metric-card .label { color: #64726C; font-size: .73rem; font-weight: 800; text-transform: uppercase; letter-spacing: .055em; }
.metric-card .value { color: #18231F; font-size: 1.9rem; font-weight: 800; margin-top: 8px; }
.metric-card .hint, .muted { color: #64726C; font-size: .78rem; line-height: 1.45; }
.finding { border-left: 4px solid #EC7565; background: #FFF4F1; border-radius: 10px; padding: 12px 14px; margin: 9px 0; }
.finding-ok { border-left: 4px solid #176B50; background: #EFF8F3; border-radius: 10px; padding: 12px 14px; margin: 9px 0; }
.question { background: white; border: 1px solid #DDE5DF; border-radius: 20px; padding: 22px 24px; margin: 14px 0; }
.question-number { color: #176B50; font-size: .75rem; font-weight: 800; letter-spacing: .08em; }
.evidence { border-left: 4px solid #176B50; background: white; border-radius: 12px; padding: 14px 16px; margin: 10px 0; border: 1px solid #E2E7E2; border-left-width: 4px; }
.decision { background: #18362C; color: white; border-radius: 18px; padding: 20px 22px; }
.decision strong { color: #C9F26B; }.decision h3 { color: white; margin: .35rem 0; }.decision p { color: #E4EEE9; }
.chip { display: inline-block; background: #DEF3E8; color: #16533E; border-radius: 999px; padding: 5px 10px; margin: 2px 4px 2px 0; font-size: .74rem; font-weight: 700; }
.trace { background:#EEF2FF; color:#33416A; border-radius:10px; padding:9px 12px; font-size:.72rem; margin:7px 0; word-break:break-all; }
.llm-ok { background:#174E3B; border:1px solid #28785D; border-radius:12px; padding:11px; font-size:.74rem; }
.llm-off { background:#5A302B; border:1px solid #A75B50; border-radius:12px; padding:11px; font-size:.74rem; }
.empty { background: white; border: 1px dashed #BCC9C2; border-radius: 22px; padding: 42px; text-align: center; }
.review-item { background:white; border:1px solid #E2E7E2; border-radius:14px; padding:14px 16px; margin:9px 0; }
.review-item.ok { border-left:4px solid #176B50; }.review-item.partial { border-left:4px solid #F4B860; }.review-item.bad { border-left:4px solid #EC7565; }
.gap-card { background:white; border:1px solid #E2E7E2; border-radius:16px; padding:16px 18px; margin:10px 0; box-shadow:0 6px 18px rgba(16,45,35,.05); }
.gap-card .count { float:right; background:#FFF0EC; color:#9D382E; border-radius:999px; padding:4px 9px; font-weight:800; font-size:.72rem; }
.rule-card { background:#F0F5FF; border:1px solid #D8E2F7; border-radius:14px; padding:14px 16px; margin:10px 0; color:#273A63; }
.mission-hero { background:linear-gradient(130deg,#172A45,#244F66); color:white; border-radius:22px; padding:24px 26px; margin:14px 0 18px; box-shadow:0 14px 34px rgba(19,42,56,.14); }
.mission-hero h2 { color:white; margin:.3rem 0; }.mission-hero p { color:#E5F1F5; margin:.35rem 0; }
.mission-objective { color:#BFEA8C !important; font-weight:800; font-size:.78rem; letter-spacing:.07em; }
.mission-signal { border:1px solid #D8E5DF; border-radius:14px; padding:12px 14px; background:#FBFCFA; margin:8px 0; }
.mission-result { border:1px solid #B7DAC9; background:#EFF9F4; border-radius:18px; padding:18px 20px; margin:14px 0; }
.stButton > button, .stFormSubmitButton > button { border-radius: 12px; font-weight: 750; border: 0; background: #176B50; color: white; }
[data-testid="stExpander"] { background: white; border: 1px solid #E2E7E2; border-radius: 16px; }
</style>
""",
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def trace_card(trace: dict | LLMTrace, label: str | None = None) -> None:
    item = asdict(trace) if isinstance(trace, LLMTrace) else trace
    st.markdown(
        f'<div class="trace"><b>Локальная LLM · {label or item.get("stage", "вызов")}</b><br>'
        f'{item.get("model", "—")} · llama.cpp · {item.get("duration_ms", 0) / 1000:.1f} с<br>'
        f'ID: {item.get("trace_id", "—")}<br>SHA-256 весов: {item.get("model_sha256", "—")}</div>',
        unsafe_allow_html=True,
    )


def assignment_curriculum(assignment: dict) -> Curriculum:
    allowed = set(assignment["skill_ids"])
    return replace(CURRICULUM, skills=tuple(s for s in CURRICULUM.skills if s.id in allowed))


def assignment_topic_key(assignment: dict) -> str:
    return str(assignment.get("topic_key") or assignment["skill_ids"][0])


def difficulty_label(assignment: dict) -> str:
    return DIFFICULTY_LABELS.get(int(assignment.get("difficulty") or 1), "Базовый")


def criterion_results_from_check(check: dict) -> list[dict]:
    return [
        {
            "criterion": f'Пункт {slot["position"]}',
            "status": (
                "correct"
                if slot["correct"]
                else "partial"
                if float(slot.get("score", 0)) >= 0.4
                else "incorrect"
            ),
            "score": float(slot.get("score", int(slot["correct"]))),
            "student_evidence": slot["student_evidence"],
            "issue": slot["issue"],
            "correction": slot.get("correction") or slot["expected_phrase"],
        }
        for slot in check["slots"]
    ]


def regrade_relative_clause_attempt(attempt: dict, assignment: dict) -> dict:
    assessment = dict(attempt.get("assessment") or {})
    if "eng_relative_clauses" not in assignment.get("skill_ids", []) or int(
        assessment.get("relative_clause_grader_version") or 0
    ) >= RELATIVE_CLAUSE_GRADER_VERSION:
        return attempt
    check = grade_structured_answer(assignment, str(attempt.get("artifact") or ""))
    if not check:
        return attempt
    result = dict(attempt)
    wrong_positions = [str(slot["position"]) for slot in check["slots"] if not slot["correct"]]
    assessment.update(
        {
            "relative_clause_grader_version": RELATIVE_CLAUSE_GRADER_VERSION,
            "submission_score": float(check["score"]),
            "is_correct": bool(check["correct"]),
            "feedback": (
                "Все ответы грамматически корректны и сохраняют исходные факты."
                if check["correct"]
                else f'Нужно уточнить пункты: {", ".join(wrong_positions)}.'
            ),
            "mode": "viva" if check["correct"] else "diagnostic",
            "objective_check": check,
            "criterion_results": criterion_results_from_check(check),
        }
    )
    result["submission_score"] = float(check["score"])
    result["submission_correct"] = bool(check["correct"])
    result["assessment_mode"] = str(assessment["mode"])
    result["assessment"] = assessment
    result["objective_regraded"] = True
    return result


def regrade_legacy_attempt(attempt: dict, assignment: dict) -> dict:
    stored_assessment = attempt.get("assessment") or {}
    if int(stored_assessment.get("grader_version") or 0) >= 4:
        upgraded = upgrade_mastery_model(attempt, assignment)
        return regrade_relative_clause_attempt(upgraded, assignment)
    check = grade_structured_answer(assignment, str(attempt.get("artifact") or ""))
    if not check:
        return upgrade_mastery_model(attempt, assignment)
    result = dict(attempt)
    wrong_positions = [
        str(slot["position"]) for slot in check["slots"] if not slot["correct"]
    ]
    unit = "позициях" if "eng_articles" in assignment["skill_ids"] else "пунктах"
    feedback = (
        "Все ответы совпадают с проверяемым ключом."
        if check["correct"]
        else f'Ошибки или пропуски в {unit}: {", ".join(wrong_positions)}.'
    )
    result["submission_score"] = float(check["score"])
    result["submission_correct"] = bool(check["correct"])
    result["assessment_mode"] = "viva" if check["correct"] else "diagnostic"
    result["assessment"] = {
        "grader_version": 4,
        "submission_score": float(check["score"]),
        "is_correct": bool(check["correct"]),
        "feedback": feedback,
        "mode": result["assessment_mode"],
        "objective_check": check,
        "criterion_results": criterion_results_from_check(check),
    }
    result["evidence"] = []
    result["next_activity"] = {}
    result["teacher_recommendation"] = {}
    result["regraded_legacy"] = True
    return upgrade_mastery_model(result, assignment)


def upgrade_mastery_model(attempt: dict, assignment: dict) -> dict:
    assessment = dict(attempt.get("assessment") or {})
    if int(assessment.get("mastery_model_version") or 0) >= 2:
        return attempt
    result = dict(attempt)
    submission_score = float(
        assessment.get("submission_score", attempt.get("submission_score") or 0)
    )
    skill_scores = {
        str(item.get("skill_id")): float(item.get("score", submission_score))
        for item in assessment.get("skill_results") or []
    }
    previous_mastery = attempt.get("mastery") or {}
    combined = {}
    for skill_id in assignment["skill_ids"]:
        viva_scores = [
            float(item.get("score", 0))
            for item in result.get("evidence") or []
            if item.get("skill_id") == skill_id
        ]
        combined[skill_id] = combine_mastery_evidence(
            0.35,
            skill_scores.get(skill_id, submission_score),
            viva_scores,
        )
    result["mastery"] = {**previous_mastery, **combined}
    assessment["mastery_model_version"] = 2
    result["assessment"] = assessment
    result["mastery_recalculated"] = True
    return result


def rule_for_evidence(evidence: Evidence | dict) -> dict:
    rule_id = evidence.rule_id if isinstance(evidence, Evidence) else evidence.get("rule_id")
    return RULEBOOK.get(str(rule_id or ""), {})


def render_criterion_results(assessment: dict) -> None:
    items = assessment.get("criterion_results") or []
    if not items:
        return
    with st.expander("Разбор исходного решения", expanded=not assessment["is_correct"]):
        if assessment.get("objective_check"):
            st.caption(
                "Баллы определены предметным grader по атомарным компонентам правила. Локальная LLM "
                "не меняет этот результат: она получает только проверенные пробелы и использует их для Viva."
            )
        for item in items:
            status = item["status"]
            css = "ok" if status == "correct" else "partial" if status == "partial" else "bad"
            label = "Верно" if status == "correct" else "Частично" if status == "partial" else "Нужно исправить"
            if item.get("score") is not None:
                label += f' · {float(item["score"]):.0%} пункта'
            evidence = escape(str(item.get("student_evidence") or "—"))
            if status == "correct":
                details = '<p><b>Критерий выполнен.</b> Исправления не требуются.</p>'
            else:
                issue = escape(str(item.get("issue") or "—"))
                correction = escape(str(item.get("correction") or "—"))
                details = (
                    f'<p><b>Точная ошибка:</b> {issue}<br>'
                    f'<b>Исправленный вариант:</b> {correction}</p>'
                )
            st.markdown(
                f'<div class="review-item {css}"><b>{label} · {escape(str(item["criterion"]))}</b>'
                f'<p><span class="muted">В ответе:</span> {evidence}</p>{details}</div>',
                unsafe_allow_html=True,
            )


def reset_flow() -> None:
    st.session_state.pop("learning_flow", None)
    st.session_state["attempt_nonce"] = st.session_state.get("attempt_nonce", 0) + 1


def render_llm_status() -> None:
    identity = LLM.identity()
    css_class = "llm-ok" if identity["ready"] else "llm-off"
    status = "готова" if identity["ready"] else "не установлена"
    active_model = identity["model"] if identity["quality_mode"] else identity["fast_model"]
    active_hash = (
        identity["model_sha256"]
        if identity["quality_mode"]
        else identity["fast_model_sha256"]
    )
    st.markdown(
        f'<div class="{css_class}"><b>Локальная LLM: {status}</b><br>{active_model}<br>'
        f'llama.cpp · без API и облака</div>',
        unsafe_allow_html=True,
    )
    if identity["ready"]:
        short_hash = str(active_hash)[:16]
        st.caption(f"SHA-256: {short_hash}…")
    else:
        st.caption(r"Установка: scripts\setup_local_llm.ps1")


def render_sidebar() -> tuple[str, dict | None]:
    students = list_students()
    with st.sidebar:
        st.markdown('<div class="brand">◉ VivaTrace</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">Адаптивная обратная связь для учебной группы</div>', unsafe_allow_html=True)
        role = st.selectbox("Роль", ["Студент", "Преподаватель"])
        selected_student = None
        if role == "Студент":
            by_name = {item["name"]: item for item in students}
            selected_student = by_name[st.selectbox("Аккаунт студента", list(by_name))]
            st.caption("Демонстрационное переключение аккаунтов без авторизации")
        else:
            st.markdown("**Кабинет преподавателя**")
            st.caption("Только результаты реальных попыток")
        st.markdown("---")
        render_llm_status()
    return role, selected_student


def start_assessment(student: dict, assignment: dict, artifact: str) -> None:
    skill_names = {skill_id: CURRICULUM.skill_by_id[skill_id].name for skill_id in assignment["skill_ids"]}
    assessment, traces = LLM.assess_submission(assignment, artifact, skill_names)
    assessment["grader_version"] = 4
    assessment["mastery_model_version"] = 2
    if "eng_relative_clauses" in assignment.get("skill_ids", []):
        assessment["relative_clause_grader_version"] = RELATIVE_CLAUSE_GRADER_VERSION
    findings = [
        ArtifactFinding(
            skill_id=item["skill_id"],
            severity="high" if item["score"] < 0.4 else "medium",
            evidence=item["diagnosis"],
            hypothesis="Проверить понимание в диагностическом вопросе",
            confidence=0.9,
        )
        for item in assessment["skill_results"]
        if item["score"] < 0.75
    ]
    mastery_before = get_mastery(student["id"], assignment["skill_ids"])
    mastery_after_task = dict(mastery_before)
    for item in assessment["skill_results"]:
        skill_id = item["skill_id"]
        mastery_after_task[skill_id] = combine_mastery_evidence(
            mastery_before.get(skill_id, 0.35),
            float(item["score"]),
            [],
        )
    st.session_state.learning_flow = {
        "assessment_version": ASSESSMENT_VERSION,
        "student_id": student["id"],
        "assignment_id": assignment["id"],
        "artifact": artifact,
        "assessment": assessment,
        "findings": findings,
        "questions": assessment["questions"],
        "current": 0,
        "evidence": [],
        "mastery_before": mastery_before,
        "mastery_after_task": dict(mastery_after_task),
        "mastery": mastery_after_task,
        "completed": False,
        "traces": [asdict(trace) for trace in traces],
    }


def render_assignment_answer(student: dict, assignment: dict) -> str:
    """Render one compact field per deterministic numbered item."""
    reference_items = parse_numbered_items(
        str((assignment.get("rubric") or {}).get("reference_answer") or "")
    )
    instruction_items = parse_numbered_items(str(assignment.get("instructions") or ""))
    nonce = st.session_state.get("attempt_nonce", 0)
    if len(reference_items) >= 2 and len(instruction_items) == len(reference_items):
        st.caption("Введите ответ отдельно для каждого пункта. Номера и служебные подсказки добавятся автоматически.")
        answers = []
        for index, prompt in enumerate(instruction_items, start=1):
            answers.append(
                st.text_input(
                    f"{index}. {prompt}",
                    key=f'artifact-item-{student["id"]}-{assignment["id"]}-{nonce}-{index}',
                    placeholder="Ваш ответ",
                ).strip()
            )
        return "\n".join(f"{index}) {answer}" for index, answer in enumerate(answers, start=1))
    return st.text_area(
        "Ваш ответ",
        value=assignment["starter_code"],
        height=330,
        key=f'artifact-{student["id"]}-{assignment["id"]}-{nonce}',
    )


def verified_error_facts(assessment: dict, evidence_items: list) -> list[dict]:
    facts = []
    objective = (assessment or {}).get("objective_check") or {}
    for slot in objective.get("slots", []):
        if not slot.get("correct"):
            facts.append(
                {
                    "source": "rubric_check",
                    "position": slot.get("position"),
                    "student_form": slot.get("student_evidence"),
                    "expected_form": slot.get("correction") or slot.get("expected_phrase"),
                    "error_component": (slot.get("diagnostic") or {}).get("label"),
                }
            )
    for item in evidence_items:
        score = item.score if isinstance(item, Evidence) else float(item.get("score", 0))
        if score >= 0.75:
            continue
        rule_id = (
            item.rule_id or item.skill_id
            if isinstance(item, Evidence)
            else item.get("rule_id") or item.get("skill_id")
        )
        facts.append(
            {
                "source": "viva_score",
                "rule_id": rule_id,
                "score": score,
            }
        )
    return facts


def cohort_context_for_llm(assignment: dict, current_student: dict, flow: dict) -> list[dict]:
    rows = [
        {
            "student_id": row["student_id"],
            "assignment_title": row.get("assignment_title"),
            "submission_score": row.get("submission_score"),
            "viva_score": row["overall_score"],
            "errors": verified_error_facts(row.get("assessment") or {}, row["evidence"]),
        }
        for row in latest_topic_attempts(assignment_topic_key(assignment))
        if row["student_id"] != current_student["id"]
    ]
    rows.append(
        {
            "student_id": current_student["id"],
            "assignment_title": assignment["title"],
            "submission_score": flow["assessment"]["submission_score"],
            "viva_score": sum(item.score for item in flow["evidence"]) / max(len(flow["evidence"]), 1),
            "errors": verified_error_facts(flow["assessment"], flow["evidence"]),
        }
    )
    return rows


def complete_attempt(student: dict, assignment: dict, flow: dict) -> None:
    final_result, traces = LLM.finalize_learning(
        assignment,
        flow["assessment"],
        flow["evidence"],
        cohort_context_for_llm(assignment, student, flow),
    )
    flow["traces"].extend(asdict(trace) for trace in traces)
    flow["next_activity"] = final_result["student_activity"]
    flow["branch"] = final_result["branch"]
    flow["teacher_recommendation"] = final_result["teacher_recommendation"]
    save_attempt(
        student_id=student["id"],
        assignment_id=assignment["id"],
        artifact=flow["artifact"],
        findings=flow["findings"],
        evidence=flow["evidence"],
        mastery=flow["mastery"],
        submission_score=float(flow["assessment"]["submission_score"]),
        submission_correct=bool(flow["assessment"]["is_correct"]),
        assessment_mode=str(flow["assessment"]["mode"]),
        next_activity=flow["next_activity"],
        teacher_recommendation=flow["teacher_recommendation"],
        traces=flow["traces"],
        assessment={
            key: value
            for key, value in flow["assessment"].items()
            if key != "questions"
        },
    )
    flow["completed"] = True


def completed_flow_from_attempt(student: dict, assignment: dict, attempt: dict) -> dict:
    assessment = attempt.get("assessment") or {
        "submission_score": float(attempt.get("submission_score") or 0),
        "is_correct": bool(attempt.get("submission_correct")),
        "mode": str(attempt.get("assessment_mode") or "diagnostic"),
        "feedback": "Сохранённый результат предыдущей попытки.",
        "criterion_results": [],
    }
    next_activity = attempt.get("next_activity") or {
        "title": "Результат сохранённой попытки",
        "instructions": "Для этой ранней попытки персональный следующий шаг ещё не сохранялся.",
        "why": "Можно изучить разбор ответов выше или перезапустить задание.",
        "explanation": "",
        "worked_example": "",
        "practice_task": "",
        "success_criteria": "Новая попытка завершена с подтверждением понимания.",
    }
    return {
        "assessment_version": ASSESSMENT_VERSION,
        "student_id": student["id"],
        "assignment_id": assignment["id"],
        "artifact": attempt["artifact"],
        "assessment": assessment,
        "findings": [],
        "questions": [],
        "current": 0,
        "evidence": [Evidence(**item) for item in attempt["evidence"]],
        "mastery": attempt["mastery"],
        "completed": True,
        "traces": attempt.get("traces") or [],
        "next_activity": next_activity,
        "branch": (
            "transfer"
            if attempt.get("submission_correct") and attempt.get("overall_score", 0) >= 0.75
            else "remediation"
        ),
        "teacher_recommendation": attempt.get("teacher_recommendation") or {},
        "regraded_legacy": bool(attempt.get("regraded_legacy")),
    }


def restart_marker_key(student_id: str, assignment_id: int) -> str:
    return f"restart-after-{student_id}-{assignment_id}"


def begin_new_attempt(student: dict, assignment: dict, history: list[dict]) -> None:
    if history:
        st.session_state[restart_marker_key(student["id"], assignment["id"])] = history[0]["id"]
    reset_flow()


def render_student_dashboard(student: dict, assignments: list[dict]) -> None:
    assignments_by_id = {item["id"]: item for item in assignments}
    history = [
        regrade_legacy_attempt(item, assignments_by_id[item["assignment_id"]])
        for item in student_history(student["id"])
    ]
    latest_by_assignment = {}
    for attempt in history:
        latest_by_assignment.setdefault(attempt["assignment_id"], attempt)
    completed_count = len(latest_by_assignment)
    mean_submission = (
        sum(float(item.get("submission_score") or 0) for item in latest_by_assignment.values())
        / completed_count
        if completed_count
        else 0
    )
    valid_viva_count = sum(not item.get("regraded_legacy") for item in history)
    mission_rows = mission_history(student_id=student["id"])
    completed_missions = {
        item["mission_id"]: item for item in mission_rows if item["status"] == "completed"
    }
    error_counts: Counter[str] = Counter()
    for attempt in history:
        assessment = attempt.get("assessment") or {}
        objective = assessment.get("objective_check") or {}
        if objective:
            wrong_slots = sum(not bool(slot.get("correct")) for slot in objective.get("slots", []))
            if wrong_slots:
                rule_id = next(
                    (
                        skill_id
                        for skill_id in (attempt.get("mastery") or {})
                        if skill_id in RULEBOOK
                    ),
                    "eng_articles",
                )
                error_counts[rule_id] += wrong_slots
        for evidence in attempt["evidence"]:
            if float(evidence.get("score", 0)) < 0.75:
                error_counts[str(evidence.get("rule_id") or evidence.get("skill_id"))] += 1

    with st.expander("Мой учебный кабинет", expanded=True):
        columns = st.columns(4)
        with columns[0]:
            metric_card("Выполнено заданий", f"{completed_count} из {len(assignments)}", "уникальные задания")
        with columns[1]:
            metric_card("Проверок знаний", str(valid_viva_count), "валидные завершённые циклы")
        with columns[2]:
            metric_card("Средний результат", f"{mean_submission:.0%}", "по последним попыткам")
        with columns[3]:
            metric_card("Практические миссии", str(len(completed_missions)), "завершено сценариев")
        st.markdown("#### Что повторить")
        if not error_counts:
            st.success("Пока устойчивых пробелов не зафиксировано.")
        else:
            for rule_id, count in error_counts.most_common(3):
                rule = RULEBOOK.get(rule_id, {})
                skill = CURRICULUM.skill_by_id.get(rule_id)
                title = rule.get("title") or (skill.name if skill else rule_id)
                st.markdown(f"- **{title}** · ошибок или слабых ответов: {count}")
        if latest_by_assignment:
            with st.expander("Выполненные задания"):
                rows = [
                    {
                        "Задание": item.get("assignment_title", "—"),
                        "Тема": item.get("topic", "—"),
                        "Задание, %": round(float(item.get("submission_score") or 0) * 100),
                        "Viva": (
                            "пересдать"
                            if item.get("regraded_legacy")
                            else f'{round(float(item.get("overall_score") or 0) * 100)}%'
                        ),
                    }
                    for item in latest_by_assignment.values()
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        review_records = [
            {
                "topic_key": item.get("topic_key"),
                "topic": item.get("topic"),
                "score": (
                    float(item.get("overall_score") or 0)
                    if not item.get("regraded_legacy")
                    else float(item.get("submission_score") or 0)
                ),
                "completed_at": item.get("completed_at"),
            }
            for item in latest_by_assignment.values()
        ] + [
            {
                "topic_key": item["topic_key"],
                "topic": item["title"],
                "score": item["score"],
                "completed_at": item["completed_at"],
            }
            for item in completed_missions.values()
        ]
        review_plan = build_review_plan(review_records)
        if review_plan:
            with st.expander("План интервального повторения"):
                st.caption(
                    "Дата следующего извлечения знания зависит от результата: слабые темы возвращаются раньше."
                )
                for item in review_plan[:4]:
                    when = (
                        "повторить сегодня"
                        if item["days_left"] == 0
                        else "повторить сейчас"
                        if item["days_left"] < 0
                        else f'через {item["days_left"]} дн.'
                    )
                    st.markdown(
                        f'<div class="mission-signal"><b>{escape(item["title"])}</b>'
                        f'<span style="float:right"><b>{when}</b></span><br>'
                        f'<span class="muted">Последнее подтверждение: {item["score"]:.0%} · '
                        f'интервал {item["interval_days"]} дн.</span></div>',
                        unsafe_allow_html=True,
                    )


def render_mission_mode(student: dict, assignments: list[dict]) -> None:
    topic_assignments: dict[str, dict] = {}
    for assignment in assignments:
        topic_key = assignment_topic_key(assignment)
        if topic_key in MISSIONS_BY_TOPIC:
            topic_assignments.setdefault(topic_key, assignment)
    topic_keys = list(topic_assignments)
    if not topic_keys:
        st.info("Практические миссии появятся после добавления тем английского B2.")
        return

    def mission_label(topic_key: str) -> str:
        assignment = topic_assignments[topic_key]
        mission = MISSIONS_BY_TOPIC[topic_key]
        return f'{assignment["topic"]} · {mission["title"]}'

    topic_key = st.selectbox(
        "Сценарий",
        topic_keys,
        format_func=mission_label,
        key=f'mission-topic-{student["id"]}',
    )
    mission = MISSIONS_BY_TOPIC[topic_key]
    attempt = latest_mission_attempt(student["id"], mission["id"])
    st.markdown(
        f'<div class="mission-hero"><div class="mission-objective">ПРАКТИКА В РЕАЛЬНОЙ СИТУАЦИИ</div>'
        f'<h2>{escape(mission["title"])}</h2><p>{escape(mission["student_brief"])}</p>'
        f'<p><b>Ваша цель:</b> {escape(mission["objective"])}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "".join(
            f'<span class="chip">{escape(item["label"])}</span>'
            for item in mission["required_features"]
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f'{mission["npc_name"]} · {mission["npc_role"]} · до {mission["max_turns"]} коротких реплик'
    )

    if attempt is None:
        with st.expander("Можно начать так"):
            for suggestion in mission.get("suggestions") or []:
                st.markdown(f"- {suggestion}")
        if st.button("Начать миссию", type="primary", width="stretch"):
            start_mission_attempt(student["id"], mission)
            st.rerun()
        return

    messages = list(attempt["messages"])
    for message in messages:
        role = "user" if message.get("role") == "student" else "assistant"
        with st.chat_message(role):
            st.write(message.get("content") or "")

    state = dict(attempt.get("state") or {})
    student_messages = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "student"
    ]
    signal = state.get("signal") or detect_mission_features(mission, student_messages)
    if signal.get("features"):
        st.markdown("#### Прогресс цели")
        columns = st.columns(len(signal["features"]))
        for column, feature in zip(columns, signal["features"], strict=True):
            with column:
                mark = "✓ подтверждено" if feature["found"] else "○ ещё нужно"
                metric_card(feature["label"], mark, " · ".join(feature.get("evidence") or []))

    if attempt["turn_count"]:
        feedback_columns = st.columns(3)
        with feedback_columns[0]:
            metric_card("Результат миссии", f'{attempt["score"]:.0%}', "гибридная оценка")
        with feedback_columns[1]:
            metric_card(
                "Грамматическая точность",
                f'{float(state.get("grammar_score") or 0):.0%}',
                "по целевому правилу",
            )
        with feedback_columns[2]:
            metric_card(
                "Коммуникативная задача",
                f'{float(state.get("communicative_score") or 0):.0%}',
                f'{attempt["turn_count"]} из {mission["max_turns"]} реплик',
            )
        if float(signal.get("coverage") or 0) >= 0.99:
            st.success("Все целевые конструкции найдены в ваших репликах.")
        elif signal.get("found"):
            st.success(f'Уже подтверждено: {", ".join(signal["found"])}.')
        for error in state.get("errors") or []:
            st.markdown(
                f'<div class="finding"><b>Фрагмент:</b> «{escape(error["fragment"])}»<br>'
                f'<b>Что изменить:</b> {escape(error["explanation"] if re.search(r"[а-яё]", error["explanation"], re.IGNORECASE) else "Форма требует исправления по правилу темы.")}<br>'
                f'<b>Исправление:</b> {escape(error["correction"])}</div>',
                unsafe_allow_html=True,
            )
        if attempt["status"] == "active":
            if signal.get("missing"):
                st.info(f'В следующей реплике добавьте: {", ".join(signal["missing"])}.')
            else:
                st.info("Продвиньте ситуацию к цели ещё одной короткой репликой.")

    if attempt["status"] == "completed":
        result_summary = str(state.get("state_summary") or "")
        if not re.search(r"[а-яё]", result_summary, re.IGNORECASE):
            result_summary = (
                "Коммуникативная цель достигнута, обязательные конструкции использованы корректно."
            )
        st.markdown(
            f'<div class="mission-result"><h3>Миссия выполнена</h3>'
            f'<p>{escape(result_summary)}</p>'
            f'<b>Подтверждённое освоение: {attempt["score"]:.0%}</b></div>',
            unsafe_allow_html=True,
        )
    elif attempt["status"] == "needs_retry":
        st.warning(
            "Цель пока не подтверждена. Разбор сохранён: можно начать новую попытку с учётом подсказок."
        )
    else:
        with st.form(f'mission-turn-{attempt["id"]}', clear_on_submit=True):
            answer = st.text_area(
                "Ваша реплика на английском",
                height=120,
                placeholder="Ответьте персонажу и продвиньте ситуацию к цели.",
            )
            submitted = st.form_submit_button("Отправить реплику", width="stretch")
        if submitted:
            if len(answer.strip().split()) < 3:
                st.warning("Напишите содержательную реплику минимум из трёх слов.")
            else:
                try:
                    with st.spinner("Персонаж отвечает и проверяет ход…"):
                        new_student_messages = [*student_messages, answer.strip()]
                        new_signal = detect_mission_features(mission, new_student_messages)
                        next_turn = int(attempt["turn_count"]) + 1
                        result, trace = LLM.advance_mission(
                            mission, messages, answer.strip(), new_signal, next_turn
                        )
                        new_messages = [
                            *messages,
                            {"role": "student", "content": answer.strip()},
                            {"role": "npc", "content": result["npc_reply"]},
                        ]
                        new_state = {
                            **result,
                            "mastery_applied": bool(state.get("mastery_applied")),
                        }
                        if result["status"] == "completed" and not new_state["mastery_applied"]:
                            previous = get_mastery(
                                student["id"], [mission["skill_id"]]
                            )[mission["skill_id"]]
                            mastery = combine_mastery_evidence(previous, result["score"], [])
                            update_mastery(student["id"], {mission["skill_id"]: mastery})
                            new_state["mastery_applied"] = True
                            new_state["mastery_after"] = mastery
                        save_mission_turn(
                            attempt["id"],
                            new_messages,
                            new_state,
                            result["score"],
                            result["status"],
                            [*attempt.get("traces", []), asdict(trace)],
                        )
                    st.rerun()
                except LocalLLMError as error:
                    st.error(str(error))

    if attempt["status"] in {"completed", "needs_retry"}:
        if st.button("Новая попытка этой миссии", width="stretch"):
            start_mission_attempt(student["id"], mission)
            st.rerun()
    with st.expander("Технический аудит миссии"):
        st.write(
            "Наличие обязательных форм проверяет код; локальная LLM ведёт персонажа и оценивает "
            "уместность. Замечание модели показывается только вместе с буквальной цитатой ответа."
        )
        for trace in attempt.get("traces", []):
            trace_card(trace)


def render_voice_mode(student: dict, assignments: list[dict]) -> None:
    english_assignments = [
        item for item in assignments if str(item.get("subject") or "").startswith("Английский")
    ]
    topics: dict[str, list[dict]] = {}
    for item in english_assignments:
        topics.setdefault(assignment_topic_key(item), []).append(item)
    if not topics:
        st.info("Для голосовой практики пока нет заданий по английскому языку.")
        return

    topic_key = st.selectbox(
        "Тема голосовой практики",
        list(topics),
        format_func=lambda key: f'{topics[key][0]["subject"]} · {topics[key][0]["topic"]}',
        key=f'voice-topic-{student["id"]}',
    )
    topic_assignments = sorted(
        topics[topic_key], key=lambda item: (int(item.get("variant") or 1), item["id"])
    )
    assignment_ids = [item["id"] for item in topic_assignments]
    assignment_by_id = {item["id"]: item for item in topic_assignments}
    assignment_id = st.selectbox(
        "Опора для разговора",
        assignment_ids,
        format_func=lambda item_id: (
            f'{difficulty_label(assignment_by_id[item_id])} · '
            f'{assignment_by_id[item_id]["title"]}'
        ),
        key=f'voice-assignment-{student["id"]}-{topic_key}',
    )
    assignment = assignment_by_id[assignment_id]
    hero(
        "Голосовая Viva",
        "Короткий разговор по изученной теме: локальный бот слушает, отвечает голосом и сохраняет доказательства speaking.",
        f'СТУДЕНТ · {student["name"].upper()} · {assignment["topic"].upper()}',
    )
    st.markdown(
        '<span class="chip">микрофон всегда активен</span>'
        '<span class="chip">можно перебить бота</span>'
        '<span class="chip">без API и облака</span>',
        unsafe_allow_html=True,
    )
    runtime = ensure_voice_server()
    if not runtime.get("ready"):
        missing = ", ".join(runtime.get("missing") or [])
        st.error(
            "Голосовые модели не установлены. Запустите scripts\\setup_local_voice.ps1. "
            f"Не найдено: {missing}"
        )
        return
    if not runtime.get("server_ready"):
        st.error("Локальный голосовой сервер не запустился. Проверьте logs/voice-server.stderr.log.")
        return

    session_key = f'voice-session-{student["id"]}-{assignment_id}'
    if session_key not in st.session_state:
        st.session_state[session_key] = str(uuid4())
    control_left, control_right = st.columns([4, 1])
    with control_left:
        st.caption(
            f'ASR: {runtime["asr"]} · VAD: {runtime["vad"]} · '
            f'грамматика: {runtime["grammar"]} · TTS: {runtime["tts"]} · '
            'Qwen работает через локальный llama.cpp.'
        )
    with control_right:
        if st.button("Новая сессия", key=f'new-{session_key}', width="stretch"):
            st.session_state[session_key] = str(uuid4())
            st.rerun()
    config = {
        "session_id": st.session_state[session_key],
        "student_id": student["id"],
        "assignment_id": assignment_id,
        "topic": assignment["topic"],
        "port": int(runtime["port"]),
        "websocket_url": f'ws://127.0.0.1:{int(runtime["port"])}',
    }
    components.html(voice_component_html(config), height=690, scrolling=False)

    sessions = student_voice_sessions(student["id"])
    topic_sessions = [item for item in sessions if item.get("topic_key") == topic_key]
    if topic_sessions:
        latest = topic_sessions[0]
        st.caption(
            f'Последняя сохранённая голосовая сессия: {latest["turn_count"]} реплик · '
            f'общая оценка {latest["average_score"]:.0%} · '
            f'беглость {latest["average_fluency"]:.0%}.'
        )
    with st.expander("Как устроена проверка и что именно она оценивает"):
        st.markdown(
            "Микрофон передаёт PCM-аудио по постоянному WebSocket. Энергетический streaming-gate "
            "быстро определяет начало реплики и barge-in, затем Silero VAD и Whisper локально получают "
            "транскрипт. LanguageTool и предметные правила независимо проверяют структуру; Qwen оценивает "
            "смысл, формирует диалог и перепроверяет спорные случаи. Код считает темп, паузы и "
            "слова-паразиты. Произношение и акцент намеренно не оцениваются без фонемного alignment."
        )


def render_student(student: dict) -> None:
    assignments = list_assignments(active_only=True)
    render_student_dashboard(student, assignments)
    mode = st.radio(
        "Формат занятия",
        ["Тренажер", "Практическая миссия", "Голосовая Viva"],
        horizontal=True,
        key=f'learning-mode-{student["id"]}',
    )
    if mode == "Практическая миссия":
        render_mission_mode(student, assignments)
        return
    if mode == "Голосовая Viva":
        render_voice_mode(student, assignments)
        return
    progress = student_progress(student["id"])
    topics: dict[str, list[dict]] = {}
    for item in assignments:
        topics.setdefault(assignment_topic_key(item), []).append(item)
    def topic_label(key: str) -> str:
        items = topics[key]
        completed = sum(item["id"] in progress for item in items)
        return f'{items[0]["subject"]} · {items[0]["topic"]} · {completed}/{len(items)}'

    topic_key = st.selectbox(
        "Тема",
        list(topics),
        format_func=topic_label,
        key=f'student-topic-{student["id"]}',
    )
    topic_assignments = sorted(
        topics[topic_key], key=lambda item: (int(item.get("variant") or 1), item["id"])
    )
    offset = (int(student["id"][-2:]) - 1) % len(topic_assignments)
    personal_order = topic_assignments[offset:] + topic_assignments[:offset]
    assignments_by_id = {item["id"]: item for item in personal_order}

    def task_label(assignment_id: int) -> str:
        item = assignments_by_id[assignment_id]
        completed = item["id"] in progress
        mark = "🟢 Выполнено" if completed else "○ Следующее"
        return f'{mark} · {difficulty_label(item)} · вариант {item.get("variant", 1)} · {item["title"]}'

    first_unfinished = next(
        (index for index, item in enumerate(personal_order) if item["id"] not in progress), 0
    )
    assignment_id = st.selectbox(
        "Вариант задания",
        list(assignments_by_id),
        index=first_unfinished,
        format_func=task_label,
        key=f'student-assignment-{student["id"]}-{topic_key}',
    )
    assignment = assignments_by_id[assignment_id]
    context = (student["id"], assignment["id"])
    if st.session_state.get("learning_context") != context:
        st.session_state.learning_context = context
        reset_flow()

    hero(
        assignment["title"],
        assignment["instructions"],
        f'СТУДЕНТ · {student["name"].upper()} · {assignment["subject"].upper()}',
    )
    names = [CURRICULUM.skill_by_id[item].name for item in assignment["skill_ids"]]
    st.markdown("".join(f'<span class="chip">{name}</span>' for name in names), unsafe_allow_html=True)
    completed_in_topic = sum(item["id"] in progress for item in topic_assignments)
    st.caption(
        f'{difficulty_label(assignment)} уровень · персональный вариант {assignment.get("variant", 1)} · '
        f'прогресс по теме: {completed_in_topic} из {len(topic_assignments)}'
    )
    history = [
        regrade_legacy_attempt(item, assignment)
        for item in student_attempts(student["id"], assignment["id"])
    ]
    if history:
        st.caption(f'Завершённых попыток: {len(history)} · последняя оценка задания: {(history[0].get("submission_score") or 0):.0%}')

    flow = st.session_state.get("learning_flow")
    if flow is not None and flow.get("assessment_version") != ASSESSMENT_VERSION:
        reset_flow()
        st.rerun()
    latest_attempt_id = history[0]["id"] if history else None
    restart_marker = st.session_state.get(
        restart_marker_key(student["id"], assignment["id"])
    )
    if flow is None and history and restart_marker != latest_attempt_id:
        flow = completed_flow_from_attempt(student, assignment, history[0])
        st.session_state.learning_flow = flow
    if flow is None:
        st.subheader("1. Выполните задание")
        artifact = render_assignment_answer(student, assignment)
        if not LLM.identity()["ready"]:
            st.error(r"Сервис проверки сейчас недоступен. Запустите scripts\setup_local_llm.ps1 один раз.")
        if st.button("Отправить на проверку", width="stretch", disabled=not LLM.identity()["ready"]):
            numbered_answers = parse_numbered_items(artifact)
            if not artifact.strip() or (numbered_answers and not any(numbered_answers)):
                st.warning("Введите ответ перед проверкой.")
            else:
                try:
                    with st.spinner("Проверяем решение и готовим следующий этап…"):
                        start_assessment(student, assignment, artifact)
                    st.rerun()
                except LocalLLMError as error:
                    st.error(str(error))
        return

    if flow["student_id"] != student["id"] or flow["assignment_id"] != assignment["id"]:
        reset_flow()
        st.rerun()

    assessment = flow["assessment"]
    if not flow["completed"]:
        correct = bool(assessment["is_correct"])
        title = "Решение принято → подтверждаем понимание" if correct else "В решении есть ошибки → находим точный пробел"
        st.subheader(f'2. {title}')
        columns = st.columns([1, 1])
        with columns[0]:
            metric_card("Оценка самого задания", f'{assessment["submission_score"]:.0%}', "отдельно от накопленного освоения")
            css = "finding-ok" if correct else "finding"
            st.markdown(f'<div class="{css}"><b>{assessment["feedback"]}</b></div>', unsafe_allow_html=True)
        with columns[1]:
            branch_name = "Короткая проверка понимания" if correct else "Уточняющие вопросы и помощь"
            branch_hint = (
                f'{len(flow["questions"])} вопроса по правилам из задания'
                if correct
                else "сначала определим точный пробел"
            )
            metric_card("Следующий этап", branch_name, branch_hint)

        render_criterion_results(assessment)

        question = flow["questions"][flow["current"]]
        st.progress(flow["current"] / len(flow["questions"]))
        st.markdown(
            f'<div class="question"><div class="question-number">ВОПРОС {flow["current"] + 1} ИЗ {len(flow["questions"])}</div>'
            f'<h3>{question.text}</h3><div class="muted">{question.purpose}</div></div>',
            unsafe_allow_html=True,
        )
        answer = st.text_area(
            "Ответ своими словами",
            key=f'answer-{question.id}-{flow["current"]}',
            height=145,
            placeholder="Сформулируйте ответ своими словами и при необходимости приведите пример.",
        )
        left, right = st.columns([1, 3])
        with left:
            if st.button("Изменить решение"):
                begin_new_attempt(student, assignment, history)
                st.rerun()
        with right:
            if st.button("Ответить и продолжить", width="stretch"):
                if not answer.strip():
                    st.warning("Введите ответ перед продолжением.")
                else:
                    try:
                        with st.spinner("Проверяем ответ…"):
                            evidence, trace = LLM.evaluate_answer(assignment, question, answer)
                            skill_submission = next(
                                (
                                    float(item["score"])
                                    for item in assessment["skill_results"]
                                    if item["skill_id"] == question.skill_id
                                ),
                                float(assessment["submission_score"]),
                            )
                            skill_viva_scores = [
                                item.score
                                for item in [*flow["evidence"], evidence]
                                if item.skill_id == question.skill_id
                            ]
                            flow["mastery"][question.skill_id] = combine_mastery_evidence(
                                flow.get("mastery_before", {}).get(question.skill_id, 0.35),
                                skill_submission,
                                skill_viva_scores,
                            )
                            flow["evidence"].append(evidence)
                            flow["traces"].append(asdict(trace))
                            flow["current"] += 1
                            if flow["current"] >= len(flow["questions"]):
                                complete_attempt(student, assignment, flow)
                        st.session_state.learning_flow = flow
                        st.rerun()
                    except LocalLLMError as error:
                        st.error(str(error))
        return

    st.subheader("Обучающий цикл завершён")
    if flow.get("regraded_legacy"):
        st.warning(
            "Задание пересчитано новым предметным ключом. Старая проверка понимания скрыта как "
            "недостоверная — перезапустите задание, чтобы пройти новую Viva."
        )
    else:
        st.success("Результат сохранён и уже доступен преподавателю.")
    viva_average = (
        sum(item.score for item in flow["evidence"]) / len(flow["evidence"])
        if flow["evidence"]
        else None
    )
    result_columns = st.columns(3)
    with result_columns[0]:
        st.metric("Исходное задание", f'{assessment["submission_score"]:.0%}')
    with result_columns[1]:
        st.metric("Проверка понимания", f"{viva_average:.0%}" if viva_average is not None else "—")
    with result_columns[2]:
        combined_mastery = sum(flow["mastery"].values()) / max(len(flow["mastery"]), 1)
        st.metric("Освоение после всего цикла", f"{combined_mastery:.0%}")
    st.caption("Освоение пересчитывается последовательно: прошлые попытки → исходное задание → оба ответа Viva.")
    render_criterion_results(assessment)
    for evidence in flow["evidence"]:
        skill = CURRICULUM.skill_by_id[evidence.skill_id]
        mastery = flow["mastery"][evidence.skill_id]
        if evidence.verdict == "correct":
            review_details = (
                f'<p><b>Что подтверждено:</b> {escape(evidence.what_was_correct or "Ответ принят.")}</p>'
            )
        else:
            review_details = (
                f'<p><b>Что уже верно:</b> {escape(evidence.what_was_correct or "—")}</p>'
                f'<p><b>Что исправить:</b> {escape(evidence.what_needs_improvement or "—")}</p>'
                f'<p><b>Ожидаемый ответ:</b> {escape(evidence.correct_answer or "—")}</p>'
            )
        st.markdown(
            f'<div class="evidence"><b>{skill.name}</b><br>'
            f'<span class="muted">Исходный вопрос:</span> {escape(evidence.question_text)}<br>'
            f'<span class="muted">Ваш ответ:</span> «{escape(evidence.quote)}»<br>'
            f'<b>Оценка текущего ответа: {evidence.score:.0%}</b> · '
            f'<span class="muted">накопленное освоение с учётом прошлых попыток: {mastery:.0%}</span><br>'
            f'{review_details}</div>',
            unsafe_allow_html=True,
        )
        if evidence.confidence < 0.6:
            st.warning(
                "Автопроверка не уверена в этом выводе. Ответ отмечен для просмотра преподавателем."
            )
        rule = rule_for_evidence(evidence)
        if rule:
            with st.expander(f'Правило: {rule["title"]}'):
                st.write(rule["summary"])
                for principle in rule["principles"]:
                    st.markdown(f"- {principle}")
                st.link_button(f'Открыть источник · {rule["source_title"]}', rule["source_url"])

    reference_answer = (assignment.get("rubric") or {}).get("reference_answer")
    if reference_answer:
        with st.expander("Эталон исходного задания"):
            st.write(reference_answer)
    activity = flow["next_activity"]
    branch_label = (
        "Задание на перенос знания"
        if flow["branch"] == "transfer"
        else "Диагноз LLM + проверенная база правил"
    )
    help_blocks = ""
    if activity.get("explanation"):
        help_blocks += f'<p><b>Короткое объяснение:</b> {activity["explanation"]}</p>'
    if activity.get("worked_example"):
        help_blocks += f'<p><b>Разобранный пример:</b> {activity["worked_example"]}</p>'
    if activity.get("practice_task"):
        help_blocks += f'<p><b>Повторная практика:</b> {activity["practice_task"]}</p>'
    st.markdown(
        f'<div class="decision"><strong>{branch_label}</strong><h3>{activity["title"]}</h3>'
        f'<p>{activity["instructions"]}</p>{help_blocks}<p><b>Цель:</b> {activity["why"]}<br>'
        f'<b>Критерий успеха:</b> {activity["success_criteria"]}</p></div>',
        unsafe_allow_html=True,
    )
    failed_rules = {
        evidence.rule_id or evidence.skill_id: rule_for_evidence(evidence)
        for evidence in flow["evidence"]
        if evidence.score < 0.75 and rule_for_evidence(evidence)
    }
    if failed_rules:
        st.markdown("#### Материалы по выявленным пробелам")
        for rule in failed_rules.values():
            st.link_button(
                f'Изучить правило · {rule["title"]}',
                rule["source_url"],
                width="stretch",
            )
    with st.expander("Технический журнал проверки"):
        for trace in flow["traces"]:
            trace_card(trace)
    if st.button("Перезапустить это задание"):
        begin_new_attempt(student, assignment, history)
        st.rerun()


def attempts_to_states(attempts: list[dict]) -> list[StudentState]:
    return [
        StudentState(
            student_id=item["student_id"],
            name=item["student_name"],
            mastery=item["mastery"],
            evidence=[Evidence(**entry) for entry in item["evidence"]],
        )
        for item in attempts
    ]


def grounded_group_gaps(attempts: list[dict]) -> dict[str, dict]:
    gaps: dict[str, dict] = {}
    for attempt in attempts:
        attempt_objective_gaps: dict[str, str] = {}
        assessment = attempt.get("assessment") or {}
        objective = assessment.get("objective_check") or {}
        objective_rule_id = next(
            (
                str(item.get("rule_id") or item.get("skill_id"))
                for item in attempt.get("evidence", [])
                if item.get("rule_id") or item.get("skill_id")
            ),
            str(objective.get("rule_id") or ""),
        )
        for slot in objective.get("slots", []):
            if slot.get("correct") or not objective_rule_id:
                continue
            diagnostic = dict(slot.get("diagnostic") or {})
            component_code = str(diagnostic.get("code") or "objective_form")
            gap_key = f"{objective_rule_id}:{component_code}"
            attempt_objective_gaps.setdefault(objective_rule_id, gap_key)
            gap = gaps.setdefault(
                gap_key,
                {
                    "students": set(),
                    "observations": [],
                    "expected_forms": [],
                    "focus_counts": Counter(),
                    "component_counts": Counter(),
                    "viva_failures": 0,
                    "rule": RULEBOOK.get(objective_rule_id, {}),
                },
            )
            gap["students"].add(attempt["student_name"])
            component_label = str(diagnostic.get("label") or "предметная форма")
            gap["component_counts"][component_label] += 1
            rule_focus = str(diagnostic.get("rule_focus") or "").strip()
            if rule_focus:
                gap["focus_counts"][rule_focus] += 1
            observation = (
                f'позиция {slot.get("position")} · {component_label}: '
                f'«{slot.get("student_evidence") or "пропуск"}» '
                f'→ «{slot.get("correction") or slot.get("expected_phrase") or "эталон"}»'
            )
            if observation not in gap["observations"]:
                gap["observations"].append(observation)
            expected_form = str(slot.get("correction") or slot.get("expected_phrase") or "")
            if expected_form and expected_form not in gap["expected_forms"]:
                gap["expected_forms"].append(expected_form)
        for entry in attempt.get("evidence", []):
            if float(entry.get("score", 0)) >= 0.75:
                continue
            rule_id = str(entry.get("rule_id") or entry.get("skill_id") or "")
            if not rule_id:
                continue
            gap_key = attempt_objective_gaps.get(rule_id, f"{rule_id}:viva")
            gap = gaps.setdefault(
                gap_key,
                {
                    "students": set(),
                    "observations": [],
                    "expected_forms": [],
                    "focus_counts": Counter(),
                    "component_counts": Counter(),
                    "viva_failures": 0,
                    "rule": RULEBOOK.get(rule_id, {}),
                },
            )
            gap["students"].add(attempt["student_name"])
            gap["viva_failures"] += 1
    return gaps


def grounded_gap_focus(gap: dict) -> str:
    focus_counts = gap.get("focus_counts") or Counter()
    if focus_counts:
        return str(focus_counts.most_common(1)[0][0]).rstrip(".")
    rule = gap.get("rule") or {}
    principles = list(rule.get("principles") or [])
    expected_focus = " ".join(gap.get("expected_forms") or [])
    if principles and expected_focus:
        focus_tokens = set(re.findall(r"[a-z]+(?:'[a-z]+)?", expected_focus.lower()))
        return max(
            principles,
            key=lambda item: len(
                focus_tokens & set(re.findall(r"[a-z]+(?:'[a-z]+)?", item.lower()))
            ),
        ).rstrip(".")
    return str(
        principles[0] if principles else rule.get("summary") or "проверяемый навык"
    ).rstrip(".")


def lesson_plan_matches_gap(candidate: str, gap: dict) -> bool:
    focus = grounded_gap_focus(gap).lower()
    stop_words = {
        "relative",
        "clause",
        "правило",
        "форма",
        "формы",
        "нужно",
        "речь",
        "идёт",
        "каком",
    }
    focus_tokens = {
        token
        for token in re.findall(r"[a-zа-яё]+", focus)
        if len(token) >= 5 and token not in stop_words
    }
    candidate_tokens = set(re.findall(r"[a-zа-яё]+", candidate.lower()))
    return not focus_tokens or bool(focus_tokens & candidate_tokens)


def grounded_teacher_summary(attempts: list[dict]) -> dict[str, str] | None:
    gaps = grounded_group_gaps(attempts)
    if not gaps:
        return None
    _, gap = max(
        gaps.items(),
        key=lambda item: (
            len(item[1]["students"]),
            bool(item[1]["observations"]),
            item[1]["viva_failures"],
        ),
    )
    rule = gap["rule"]
    title = str(rule.get("title") or "Проверяемый навык")
    student_count = len(gap["students"])
    student_word = (
        "студент"
        if student_count % 10 == 1 and student_count % 100 != 11
        else "студента"
        if student_count % 10 in {2, 3, 4} and student_count % 100 not in {12, 13, 14}
        else "студентов"
    )
    student_genitive = "студента" if student_count == 1 else "студентов"
    facts = "; ".join(gap["observations"][:2])
    if facts:
        reason = f"{student_count} {student_word}: проверяемые ошибки — {facts}."
    else:
        reason = (
            f'У {student_count} {student_genitive} — ответов Viva ниже 75%: {gap["viva_failures"]}. '
            f'Навык: «{title}».'
        )
    newest = max(attempts, key=lambda item: item["completed_at"])
    candidate = str(
        (newest.get("teacher_recommendation") or {}).get("lesson_plan") or ""
    ).strip()
    unsafe_fragments = ("[{", "}]", "```", "null", "undefined")
    llm_plan_is_valid = (
        45 <= len(candidate) <= 320
        and not any(fragment in candidate.lower() for fragment in unsafe_fragments)
        and candidate.endswith((".", "!", "?"))
        and all(marker in candidate for marker in ("1)", "2)", "3)"))
        and "правильно:" not in candidate.lower()
        and lesson_plan_matches_gap(candidate, gap)
    )
    focus = grounded_gap_focus(gap)
    if any(token in focus.lower() for token in ("defining", "запят", "relative clause")):
        fallback_plan = (
            f"1) Разберите контраст двух предложений по правилу: {focus}. "
            "2) Попросите студентов изменить структуру и объяснить, как запятые меняют статус информации. "
            "3) Завершите новым exit-ticket на тот же компонент."
        )
    else:
        fallback_plan = (
            f"1) Сопоставьте ошибочную и правильную конструкцию по правилу: {focus}. "
            "2) Дайте два новых контрастных примера. 3) Завершите exit-ticket на тот же компонент."
        )
    return {
        "focus_topic": title,
        "reason": reason,
        "lesson_plan": candidate if llm_plan_is_valid else fallback_plan,
        "source": "llm_grounded" if llm_plan_is_valid else "grounded_fallback",
    }


def topic_key_for_new_topic(topic: str, skill_id: str) -> str:
    digest = hashlib.sha1(topic.strip().lower().encode("utf-8")).hexdigest()[:10]
    return f"custom_{skill_id}_{digest}"


def lines_from_text(value: str) -> list[str]:
    return [line.strip(" -•\t") for line in value.splitlines() if line.strip(" -•\t")]


@st.dialog("Добавить задание или тему", width="large")
def render_quick_assignment_dialog(assignments: list[dict]) -> None:
    creation_type = st.segmented_control(
        "Что добавить?",
        ["Задание к существующей теме", "Новую тему"],
        default="Задание к существующей теме",
        width="stretch",
    )
    if creation_type is None:
        return

    topics: dict[str, list[dict]] = {}
    for item in assignments:
        topics.setdefault(assignment_topic_key(item), []).append(item)
    labels = {
        f'{items[0]["subject"]} · {items[0]["topic"]}': key
        for key, items in topics.items()
    }
    skill_labels = {skill.name: skill.id for skill in CURRICULUM.skills}

    if creation_type == "Задание к существующей теме":
        selected_label = st.selectbox("Тема", list(labels), key="quick-existing-topic")
        selected_key = labels[selected_label]
        selected_items = topics[selected_key]
        base = selected_items[0]
        subject = str(base["subject"])
        topic = str(base["topic"])
        topic_key = selected_key
        skill_ids = list(
            dict.fromkeys(
                skill_id for item in selected_items for skill_id in item["skill_ids"]
            )
        )
        variant = max(int(item.get("variant") or 1) for item in selected_items) + 1
        st.caption(
            f"Предмет: {subject} · вариант {variant} · навыки подставлены автоматически."
        )
    else:
        st.info("Новая тема появится после создания её первого задания.")
        subject = st.text_input(
            "Предмет и уровень",
            value="Английский язык · B2",
            placeholder="Например: Английский язык · B2",
        )
        topic = st.text_input(
            "Название новой темы",
            placeholder="Например: Phrasal verbs в академической речи",
        )
        skill_names = list(skill_labels)
        preferred_skill_index = next(
            (
                index
                for index, name in enumerate(skill_names)
                if skill_labels[name].startswith("eng_")
            ),
            0,
        )
        selected_skill = st.selectbox(
            "Какой основной навык проверяем?",
            skill_names,
            index=preferred_skill_index,
            help="Навык связывает ответы студентов с картой освоения группы.",
        )
        skill_ids = [skill_labels[selected_skill]]
        topic_key = topic_key_for_new_topic(topic, skill_ids[0]) if topic.strip() else ""
        variant = 1

    with st.form(f"quick-create-{creation_type}"):
        title = st.text_input(
            "Название задания",
            placeholder="Короткое название, которое увидит студент",
        )
        difficulty = st.select_slider(
            "Сложность",
            options=list(DIFFICULTY_LABELS),
            value=1,
            format_func=lambda value: DIFFICULTY_LABELS[value],
        )
        instructions = st.text_area(
            "Условие задания",
            height=150,
            placeholder="Напишите точное условие и все данные, необходимые студенту.",
        )
        starter = st.text_area(
            "Шаблон ответа — необязательно",
            height=80,
            placeholder="Например: 1) ...  2) ...",
        )
        st.markdown("#### Как проверять")
        reference_answer = st.text_area(
            "Эталонный ответ",
            height=110,
            placeholder="Правильное решение или возможный образец ответа",
        )
        criteria_text = st.text_area(
            "Критерии — каждый с новой строки",
            height=120,
            placeholder="Правильно применено правило\nОтвет соответствует контексту",
        )
        common_errors_text = st.text_area(
            "Типичные ошибки — необязательно, каждая с новой строки",
            height=90,
            placeholder="Неверная форма\nПравило применено без учёта контекста",
        )
        submitted = st.form_submit_button(
            "Создать и показать студентам", type="primary", width="stretch"
        )

    if not submitted:
        return
    criteria = lines_from_text(criteria_text)
    common_errors = lines_from_text(common_errors_text)
    if not all(
        [
            subject.strip(),
            topic.strip(),
            title.strip(),
            instructions.strip(),
            reference_answer.strip(),
            criteria,
            skill_ids,
        ]
    ):
        st.error("Заполните название, условие, эталон и хотя бы один критерий.")
        return
    rubric = {
        "reference_answer": reference_answer.strip(),
        "criteria": criteria,
        "common_errors": common_errors,
    }
    create_assignment(
        title,
        topic,
        instructions,
        starter,
        skill_ids,
        subject,
        rubric,
        topic_key,
        difficulty,
        variant,
    )
    st.success("Готово: задание создано и уже доступно студентам.")
    st.rerun()


def render_teacher_missions(topic_key: str) -> None:
    mission = MISSIONS_BY_TOPIC.get(topic_key)
    if not mission:
        return
    history = mission_history(topic_key=topic_key)
    latest_by_student: dict[str, dict] = {}
    for attempt in history:
        latest_by_student.setdefault(attempt["student_id"], attempt)
    with st.expander("Практическая миссия группы", expanded=bool(latest_by_student)):
        st.caption(
            f'{mission["title"]}: короткий ролевой перенос правила в реальную коммуникативную задачу.'
        )
        if not latest_by_student:
            st.info("Студенты ещё не запускали эту миссию.")
            return
        rows = list(latest_by_student.values())
        completed = [item for item in rows if item["status"] == "completed"]
        mean_score = sum(item["score"] for item in rows) / len(rows)
        reviewed = sum(bool((item.get("state") or {}).get("requires_review")) for item in rows)
        columns = st.columns(3)
        with columns[0]:
            metric_card("Участвовали", str(len(rows)), "уникальные студенты")
        with columns[1]:
            metric_card("Достигли цели", f"{len(completed)} из {len(rows)}", "по проверенному порогу")
        with columns[2]:
            metric_card(
                "Средний результат",
                f"{mean_score:.0%}",
                f"требуют внимания: {reviewed}",
            )
        table_rows = []
        for item in rows:
            signal = (item.get("state") or {}).get("signal") or {}
            table_rows.append(
                {
                    "Студент": item["student_name"],
                    "Статус": (
                        "цель достигнута"
                        if item["status"] == "completed"
                        else "нужна новая попытка"
                        if item["status"] == "needs_retry"
                        else "в процессе"
                    ),
                    "Реплик": item["turn_count"],
                    "Результат": round(item["score"] * 100),
                    "Обязательные формы": f'{len(signal.get("found") or [])}/{len(signal.get("features") or mission["required_features"])}',
                }
            )
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
        grounded_errors = [
            {
                "student": item["student_name"],
                **error,
            }
            for item in rows
            for error in (item.get("state") or {}).get("errors") or []
        ]
        if grounded_errors:
            st.markdown("#### Подтверждённые фрагменты для разбора")
            for error in grounded_errors[:6]:
                st.markdown(
                    f'<div class="gap-card"><h4>{escape(error["student"])}</h4>'
                    f'<p><b>Фрагмент:</b> «{escape(error["fragment"])}»</p>'
                    f'<p><b>Разбор:</b> {escape(error["explanation"])}</p>'
                    f'<p><b>Исправление:</b> {escape(error["correction"])}</p></div>',
                    unsafe_allow_html=True,
                )


def render_teacher_voice_sessions(topic_key: str) -> None:
    rows = latest_voice_topic_sessions(topic_key)
    with st.expander("Голосовая Viva группы", expanded=bool(rows)):
        if not rows:
            st.caption(
                "После первой голосовой сессии здесь появятся транскрипты и проверяемые speaking-метрики."
            )
            return
        mean_score = sum(float(item["average_score"]) for item in rows) / len(rows)
        mean_fluency = sum(float(item["average_fluency"]) for item in rows) / len(rows)
        columns = st.columns(3)
        with columns[0]:
            metric_card("Участвовали", str(len(rows)), "уникальные студенты")
        with columns[1]:
            metric_card("Speaking", f"{mean_score:.0%}", "грамматика + словарь + смысл + беглость")
        with columns[2]:
            metric_card("Беглость", f"{mean_fluency:.0%}", "темп, паузы и fillers")
        table = []
        for item in rows:
            metrics = item.get("latest_metrics") or {}
            assessment = item.get("latest_assessment") or {}
            table.append(
                {
                    "Студент": item["student_name"],
                    "Реплик": int(item["turn_count"]),
                    "Speaking, %": round(float(item["average_score"]) * 100),
                    "Слов/мин": round(float(metrics.get("words_per_minute") or 0)),
                    "Паузы, %": round(float(metrics.get("pause_ratio") or 0) * 100),
                    "Грамматика, %": round(float(assessment.get("grammar_score") or 0) * 100),
                    "Словарь, %": round(float(assessment.get("vocabulary_score") or 0) * 100),
                }
            )
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        weakest = min(rows, key=lambda item: float(item["average_score"]))
        assessment = weakest.get("latest_assessment") or {}
        st.markdown(
            f'<div class="gap-card"><h4>{escape(weakest["student_name"])}</h4>'
            f'<p><b>Последняя реплика:</b> «{escape(weakest.get("latest_student_text") or "—")}»</p>'
            f'<p><b>Обратная связь:</b> {escape(str(assessment.get("feedback_ru") or "—"))}</p>'
            f'<p><b>Следующий фокус:</b> {escape(str(assessment.get("next_goal_ru") or "—"))}</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Произношение не включено в балл: для него потребуется отдельное выравнивание аудио по фонемам."
        )


def render_teacher() -> None:
    assignments = list_assignments(active_only=False)
    topics: dict[str, list[dict]] = {}
    for item in assignments:
        topics.setdefault(assignment_topic_key(item), []).append(item)
    by_label = {
        f'{items[0]["subject"]} · {items[0]["topic"]} · {len(items)} вариантов': key
        for key, items in topics.items()
    }
    selector_column, action_column = st.columns([3, 1])
    with selector_column:
        topic_key = by_label[st.selectbox("Результаты по теме", list(by_label))]
    with action_column:
        st.write("")
        st.write("")
        if st.button("＋ Добавить", type="primary", width="stretch"):
            render_quick_assignment_dialog(assignments)
    topic_assignments = topics[topic_key]
    assignment = dict(topic_assignments[0])
    assignment["skill_ids"] = list(
        dict.fromkeys(
            skill_id for item in topic_assignments for skill_id in item["skill_ids"]
        )
    )
    topic_assignments_by_id = {item["id"]: item for item in topic_assignments}
    attempts = [
        regrade_legacy_attempt(item, topic_assignments_by_id[item["assignment_id"]])
        for item in latest_topic_attempts(topic_key)
    ]
    students = list_students()
    hero(
        "Пульс учебной группы",
        "Результаты разных персональных вариантов объединены по общему навыку и показывают конкретные пробелы группы.",
        f'ПРЕПОДАВАТЕЛЬ · {assignment["subject"].upper()} · {assignment["topic"].upper()}',
    )
    render_teacher_voice_sessions(topic_key)
    render_teacher_missions(topic_key)
    if not attempts:
        st.markdown(
            f'<div class="empty"><h2>Пока нет завершённых попыток</h2><p>Пульс пуст: предзаполненных результатов нет.</p><b>Пройдено: 0 из {len(students)}</b></div>',
            unsafe_allow_html=True,
        )
        render_assignment_management(topic_assignments[0])
        return

    states = attempts_to_states(attempts)
    cohort = assignment_curriculum(assignment)
    frame = mastery_frame(cohort, states)
    skill_ids = [skill.id for skill in cohort.skills]
    mean_submission = sum((item.get("submission_score") or 0) for item in attempts) / len(attempts)
    valid_viva_attempts = [item for item in attempts if not item.get("regraded_legacy")]
    mean_viva = (
        sum(item["overall_score"] for item in valid_viva_attempts)
        / len(valid_viva_attempts)
        if valid_viva_attempts
        else None
    )
    needs_help = sum(
        1
        for item in attempts
        if not item.get("submission_correct")
        or (not item.get("regraded_legacy") and item["overall_score"] < 0.55)
        or any(float(entry.get("confidence") or 0) < 0.6 for entry in item["evidence"])
    )
    columns = st.columns(4)
    with columns[0]:
        metric_card("Завершили", f"{len(attempts)} из {len(students)}", "последние попытки")
    with columns[1]:
        metric_card("Задание", f"{mean_submission:.0%}", "средняя предметная оценка")
    with columns[2]:
        metric_card(
            "Viva",
            f"{mean_viva:.0%}" if mean_viva is not None else "—",
            "среднее понимание" if mean_viva is not None else "нужна новая проверка",
        )
    with columns[3]:
        metric_card("Нужна помощь", str(needs_help), "ошибка в задании или viva")

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.subheader("Карта накопленного освоения")
        heatmap = frame.set_index("student")[skill_ids]
        heatmap.columns = [skill.name for skill in cohort.skills]
        figure = px.imshow(
            heatmap,
            zmin=0,
            zmax=1,
            aspect="auto",
            color_continuous_scale=[[0, COLORS["coral"]], [0.5, COLORS["amber"]], [0.8, COLORS["lime"]], [1, COLORS["green"]]],
            labels={"color": "Освоение"},
            text_auto=".0%",
        )
        figure.update_layout(height=max(280, 110 + 70 * len(attempts)), margin=dict(l=5, r=5, t=10, b=5), coloraxis_colorbar_tickformat=".0%")
        figure.update_xaxes(side="top")
        figure.update_yaxes(title_text="Студент")
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Тема следующего занятия")
        newest = max(attempts, key=lambda item: item["completed_at"])
        recommendation = grounded_teacher_summary(attempts) or {}
        if recommendation:
            source_label = (
                "План локальной LLM, факты проверены по данным группы"
                if recommendation["source"] == "llm_grounded"
                else "Безопасный план по проверенным данным группы"
            )
            st.markdown(
                f'<div class="decision"><strong>{source_label}</strong>'
                f'<h3>{recommendation["focus_topic"]}</h3><p>{recommendation["reason"]}</p>'
                f'<p><b>План:</b> {recommendation["lesson_plan"]}</p></div>',
                unsafe_allow_html=True,
            )
            if recommendation["source"] == "llm_grounded" and newest.get("traces"):
                trace_card(newest["traces"][-1], "план следующей пары")
        else:
            st.warning(
                "Новая рекомендация появится после попытки в обновлённом формате проверки."
            )

    st.divider()
    detail_left, detail_right = st.columns([1, 1.2], gap="large")
    with detail_left:
        st.subheader("Выявленные пробелы")
        gaps = grounded_group_gaps(attempts)
        if not gaps:
            st.info("В новых попытках явные повторяющиеся заблуждения не зафиксированы.")
        else:
            for gap in sorted(gaps.values(), key=lambda item: len(item["students"]), reverse=True):
                rule = gap["rule"]
                title = rule.get("title", "Проверяемый навык")
                students_text = ", ".join(sorted(gap["students"]))
                observations = " · ".join(gap["observations"][:3])
                if gap["viva_failures"]:
                    viva_fact = f'ответов Viva ниже 75%: {gap["viva_failures"]}'
                    observations = f"{observations} · {viva_fact}" if observations else viva_fact
                corrections = grounded_gap_focus(gap)
                st.markdown(
                    f'<div class="gap-card"><span class="count">{len(gap["students"])} чел.</span>'
                    f'<h4>{escape(title)}</h4><p><b>У кого:</b> {escape(students_text)}</p>'
                    f'<p><b>Что не получается:</b> {escape(observations)}</p>'
                    f'<p><b>Ориентир:</b> {escape(corrections)}</p></div>',
                    unsafe_allow_html=True,
                )
    with detail_right:
        st.subheader("Последние результаты")
        rows = [
            {
                "Студент": item["student_name"],
                "Вариант": item.get("assignment_title", "—"),
                "Задание": round((item.get("submission_score") or 0) * 100),
                "Viva": (
                    "нужна повторная проверка"
                    if item.get("regraded_legacy")
                    else f'{round(item["overall_score"] * 100)}%'
                ),
                "Маршрут": "Проверка понимания" if item.get("submission_correct") else "Диагностика пробела",
                "Контроль": (
                    "проверить"
                    if any(
                        float(entry.get("confidence") or 0) < 0.6
                        for entry in item["evidence"]
                    )
                    else "не требуется"
                ),
            }
            for item in attempts
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Ответы и LLM-аудит конкретного студента"):
        by_name = {item["student_name"]: item for item in attempts}
        detail = by_name[st.selectbox("Студент", list(by_name), key="teacher-detail")]
        st.markdown(f'**Оценка задания: {(detail.get("submission_score") or 0):.0%}**')
        st.caption(f'Вариант: {detail.get("assignment_title", "—")}')
        st.code(detail["artifact"], language="text")
        if detail.get("regraded_legacy"):
            st.warning(
                "Оценка задания пересчитана новым предметным ключом. Старая Viva скрыта как "
                "недостоверная; студенту нужно пройти проверку понимания заново."
            )
        for entry in detail["evidence"]:
            name = CURRICULUM.skill_by_id[entry["skill_id"]].name
            st.markdown(
                f'**{name} · текущий ответ {entry["score"]:.0%}**  \n'
                f'**Вопрос:** {entry.get("question_text", "—")}  \n'
                f'> {entry["quote"]}  \n'
                f'**Пробел:** {entry.get("what_needs_improvement") or entry.get("misconception") or "—"}  \n'
                f'**Корректный ответ:** {entry.get("correct_answer", "—")}'
            )
        for trace in detail.get("traces", []):
            trace_card(trace)
    render_assignment_management(topic_assignments[0])


def render_assignment_management(selected_assignment: dict) -> None:
    with st.expander("Расширенное редактирование задания"):
        mode = st.radio("Действие", ["Изменить текущее", "Создать новое"], horizontal=True)
        defaults = selected_assignment if mode == "Изменить текущее" else {
            "title": "", "subject": "", "topic": "", "instructions": "", "starter_code": "",
            "skill_ids": [], "rubric": {}, "topic_key": "", "difficulty": 1, "variant": 1,
        }
        labels = {skill.name: skill.id for skill in CURRICULUM.skills}
        with st.form(f'assignment-{mode}-{selected_assignment["id"]}'):
            subject = st.text_input("Предмет и уровень", value=defaults.get("subject", ""))
            title = st.text_input("Название", value=defaults["title"])
            topic = st.text_input("Тема", value=defaults["topic"])
            topic_key = st.text_input(
                "Ключ темы",
                value=defaults.get("topic_key") or (defaults.get("skill_ids") or [""])[0],
            )
            difficulty = st.selectbox(
                "Сложность",
                list(DIFFICULTY_LABELS),
                index=max(0, int(defaults.get("difficulty") or 1) - 1),
                format_func=lambda value: DIFFICULTY_LABELS[value],
            )
            variant = st.number_input(
                "Номер варианта", min_value=1, value=int(defaults.get("variant") or 1)
            )
            instructions = st.text_area("Условие", value=defaults["instructions"], height=130)
            starter = st.text_area("Шаблон ответа", value=defaults["starter_code"], height=170)
            chosen = st.multiselect(
                "Проверяемые навыки",
                list(labels),
                default=[CURRICULUM.skill_by_id[item].name for item in defaults["skill_ids"]],
            )
            rubric_text = st.text_area(
                "Рубрика для локальной LLM (JSON)",
                value=json.dumps(defaults.get("rubric") or {}, ensure_ascii=False, indent=2),
                height=220,
            )
            submitted = st.form_submit_button("Сохранить" if mode == "Изменить текущее" else "Создать", width="stretch")
        if submitted:
            try:
                rubric = json.loads(rubric_text)
                skill_ids = [labels[name] for name in chosen]
                if not all([subject.strip(), title.strip(), topic.strip(), instructions.strip(), skill_ids, rubric]):
                    raise ValueError("Заполните все поля, навыки и рубрику.")
                if mode == "Изменить текущее":
                    update_assignment(
                        selected_assignment["id"], title, topic, instructions, starter,
                        skill_ids, subject, rubric, topic_key, difficulty, variant,
                    )
                else:
                    create_assignment(
                        title, topic, instructions, starter, skill_ids, subject, rubric,
                        topic_key, difficulty, variant,
                    )
                st.success("Задание сохранено.")
                st.rerun()
            except (json.JSONDecodeError, ValueError) as error:
                st.error(str(error))
        st.markdown("---")
        confirm = st.checkbox("Удалить все результаты трёх демонстрационных студентов")
        if st.button("Очистить результаты", disabled=not confirm):
            reset_learning_data()
            reset_flow()
            st.rerun()


init_database()
role, selected_student = render_sidebar()
if role == "Студент" and selected_student:
    render_student(selected_student)
else:
    render_teacher()
