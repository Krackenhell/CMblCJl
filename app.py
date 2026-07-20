from __future__ import annotations

import json
from dataclasses import asdict, replace
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vivatrace.bkt import BKTModel, BKTParameters
from vivatrace.cohort import mastery_frame
from vivatrace.curriculum import load_curriculum
from vivatrace.database import (
    create_assignment,
    get_mastery,
    init_database,
    latest_topic_attempts,
    list_assignments,
    list_students,
    reset_learning_data,
    save_attempt,
    student_attempts,
    student_progress,
    update_assignment,
)
from vivatrace.demo import DATA_DIR
from vivatrace.local_llm import LLMTrace, LocalLLM, LocalLLMError
from vivatrace.models import ArtifactFinding, Curriculum, Evidence, StudentState
from vivatrace.rulebook import load_rulebook


ROOT = Path(__file__).resolve().parent
CURRICULUM = load_curriculum(DATA_DIR / "curriculum.json")
LLM = LocalLLM()
RULEBOOK = load_rulebook()
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


def rule_for_evidence(evidence: Evidence | dict) -> dict:
    rule_id = evidence.rule_id if isinstance(evidence, Evidence) else evidence.get("rule_id")
    return RULEBOOK.get(str(rule_id or ""), {})


def render_criterion_results(assessment: dict) -> None:
    items = assessment.get("criterion_results") or []
    if not items:
        return
    with st.expander("Разбор исходного решения", expanded=not assessment["is_correct"]):
        for item in items:
            status = item["status"]
            css = "ok" if status == "correct" else "partial" if status == "partial" else "bad"
            label = "Верно" if status == "correct" else "Частично" if status == "partial" else "Нужно исправить"
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
    st.session_state.learning_flow = {
        "student_id": student["id"],
        "assignment_id": assignment["id"],
        "artifact": artifact,
        "assessment": assessment,
        "findings": findings,
        "questions": assessment["questions"],
        "current": 0,
        "evidence": [],
        "mastery": get_mastery(student["id"], assignment["skill_ids"]),
        "completed": False,
        "traces": [asdict(trace) for trace in traces],
    }


def cohort_context_for_llm(assignment: dict, current_student: dict, flow: dict) -> list[dict]:
    rows = [
        {
            "student_id": row["student_id"],
            "assignment_title": row.get("assignment_title"),
            "submission_score": row.get("submission_score"),
            "viva_score": row["overall_score"],
            "errors": [
                {
                    "rule_id": item.get("rule_id"),
                    "gap": item.get("what_needs_improvement") or item.get("misconception"),
                    "correct_answer": item.get("correct_answer"),
                }
                for item in row["evidence"]
                if item.get("score", 0) < 0.75
            ],
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
            "errors": [
                {
                    "rule_id": item.rule_id,
                    "gap": item.what_needs_improvement or item.misconception,
                    "correct_answer": item.correct_answer,
                }
                for item in flow["evidence"]
                if item.score < 0.75
            ],
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
    )
    flow["completed"] = True


def render_student(student: dict) -> None:
    assignments = list_assignments(active_only=True)
    progress = student_progress(student["id"])
    topics: dict[str, list[dict]] = {}
    for item in assignments:
        topics.setdefault(assignment_topic_key(item), []).append(item)
    topic_labels = {
        f'{items[0]["subject"]} · {items[0]["topic"]} · {sum(i["id"] in progress for i in items)}/{len(items)}': key
        for key, items in topics.items()
    }
    selected_topic_label = st.selectbox("Тема", list(topic_labels))
    topic_key = topic_labels[selected_topic_label]
    topic_assignments = sorted(
        topics[topic_key], key=lambda item: (int(item.get("variant") or 1), item["id"])
    )
    offset = (int(student["id"][-2:]) - 1) % len(topic_assignments)
    personal_order = topic_assignments[offset:] + topic_assignments[:offset]
    task_labels = {}
    for item in personal_order:
        completed = item["id"] in progress
        mark = "🟢 Выполнено" if completed else "○ Следующее"
        task_labels[
            f'{mark} · {difficulty_label(item)} · вариант {item.get("variant", 1)} · {item["title"]}'
        ] = item
    first_unfinished = next(
        (index for index, item in enumerate(personal_order) if item["id"] not in progress), 0
    )
    selected = st.selectbox("Вариант задания", list(task_labels), index=first_unfinished)
    assignment = task_labels[selected]
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
    history = student_attempts(student["id"], assignment["id"])
    if history:
        st.caption(f'Завершённых попыток: {len(history)} · последняя оценка задания: {(history[0].get("submission_score") or 0):.0%}')

    flow = st.session_state.get("learning_flow")
    if flow is None:
        st.subheader("1. Выполните задание")
        artifact = st.text_area(
            "Ваш ответ",
            value=assignment["starter_code"],
            height=330,
            key=f'artifact-{student["id"]}-{assignment["id"]}-{st.session_state.get("attempt_nonce", 0)}',
        )
        if not LLM.identity()["ready"]:
            st.error(r"Сервис проверки сейчас недоступен. Запустите scripts\setup_local_llm.ps1 один раз.")
        if st.button("Отправить на проверку", width="stretch", disabled=not LLM.identity()["ready"]):
            if len(artifact.strip()) < 2:
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
            branch_hint = "три вопроса по правилам из задания" if correct else "сначала определим точный пробел"
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
                reset_flow()
                st.rerun()
        with right:
            if st.button("Ответить и продолжить", width="stretch"):
                if not answer.strip():
                    st.warning("Введите ответ перед продолжением.")
                else:
                    try:
                        with st.spinner("Проверяем ответ…"):
                            evidence, trace = LLM.evaluate_answer(assignment, question, answer)
                            model = BKTModel(BKTParameters())
                            flow["mastery"][question.skill_id] = model.update(
                                flow["mastery"].get(question.skill_id, model.params.prior),
                                evidence.score,
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
    st.success("Результат сохранён и уже доступен преподавателю.")
    st.markdown(f'**Оценка исходного задания: {assessment["submission_score"]:.0%}**')
    render_criterion_results(assessment)
    for evidence in flow["evidence"]:
        skill = CURRICULUM.skill_by_id[evidence.skill_id]
        mastery = flow["mastery"][evidence.skill_id]
        st.markdown(
            f'<div class="evidence"><b>{skill.name}</b><br>'
            f'<span class="muted">Исходный вопрос:</span> {escape(evidence.question_text)}<br>'
            f'<span class="muted">Ваш ответ:</span> «{escape(evidence.quote)}»<br>'
            f'<b>Оценка текущего ответа: {evidence.score:.0%}</b> · '
            f'<span class="muted">накопленное освоение с учётом прошлых попыток: {mastery:.0%}</span><br>'
            f'<p><b>Что верно:</b> {escape(evidence.what_was_correct or "—")}</p>'
            f'<p><b>Что улучшить:</b> {escape(evidence.what_needs_improvement or "—")}</p>'
            f'<p><b>Корректный ответ:</b> {escape(evidence.correct_answer or "—")}</p>'
            f'<span class="muted">{escape(evidence.rationale)}</span></div>',
            unsafe_allow_html=True,
        )
        if evidence.typo_handling and not evidence.typo_handling.lower().startswith("нет"):
            st.caption(f"Опечатки: {evidence.typo_handling}")
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
    if st.button("Начать новую попытку"):
        reset_flow()
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


def render_teacher() -> None:
    assignments = list_assignments(active_only=False)
    topics: dict[str, list[dict]] = {}
    for item in assignments:
        topics.setdefault(assignment_topic_key(item), []).append(item)
    by_label = {
        f'{items[0]["subject"]} · {items[0]["topic"]} · {len(items)} вариантов': key
        for key, items in topics.items()
    }
    topic_key = by_label[st.selectbox("Результаты по теме", list(by_label))]
    topic_assignments = topics[topic_key]
    assignment = dict(topic_assignments[0])
    assignment["skill_ids"] = list(
        dict.fromkeys(
            skill_id for item in topic_assignments for skill_id in item["skill_ids"]
        )
    )
    attempts = latest_topic_attempts(topic_key)
    students = list_students()
    hero(
        "Пульс учебной группы",
        "Результаты разных персональных вариантов объединены по общему навыку и показывают конкретные пробелы группы.",
        f'ПРЕПОДАВАТЕЛЬ · {assignment["subject"].upper()} · {assignment["topic"].upper()}',
    )
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
    mean_viva = sum(item["overall_score"] for item in attempts) / len(attempts)
    needs_help = sum(
        1 for item in attempts if not item.get("submission_correct") or item["overall_score"] < 0.55
    )
    columns = st.columns(4)
    with columns[0]:
        metric_card("Завершили", f"{len(attempts)} из {len(students)}", "последние попытки")
    with columns[1]:
        metric_card("Задание", f"{mean_submission:.0%}", "средняя LLM-оценка")
    with columns[2]:
        metric_card("Viva", f"{mean_viva:.0%}", "среднее понимание")
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
        newest_is_current = any(
            entry.get("question_text")
            and entry.get("what_needs_improvement")
            and entry.get("correct_answer")
            for entry in newest["evidence"]
        )
        recommendation = (
            newest.get("teacher_recommendation") or {} if newest_is_current else {}
        )
        if recommendation:
            st.markdown(
                f'<div class="decision"><strong>Сформировано локальной LLM по группе</strong>'
                f'<h3>{recommendation["focus_topic"]}</h3><p>{recommendation["reason"]}</p>'
                f'<p><b>План:</b> {recommendation["lesson_plan"]}</p></div>',
                unsafe_allow_html=True,
            )
            if newest.get("traces"):
                trace_card(newest["traces"][-1], "план следующей пары")
        else:
            st.warning(
                "Новая рекомендация появится после попытки в обновлённом формате проверки."
            )

    st.divider()
    detail_left, detail_right = st.columns([1, 1.2], gap="large")
    with detail_left:
        st.subheader("Выявленные пробелы")
        gaps: dict[str, dict] = {}
        legacy_attempts = 0
        for attempt in attempts:
            has_structured_evidence = any(
                entry.get("question_text")
                and entry.get("what_needs_improvement")
                and entry.get("correct_answer")
                for entry in attempt["evidence"]
            )
            if not has_structured_evidence:
                legacy_attempts += 1
                continue
            for entry in attempt["evidence"]:
                if float(entry.get("score", 0)) >= 0.75:
                    continue
                key = str(entry.get("rule_id") or entry.get("skill_id"))
                gap = gaps.setdefault(
                    key,
                    {
                        "students": set(),
                        "observations": [],
                        "correct_answers": [],
                        "rule": RULEBOOK.get(key, {}),
                    },
                )
                gap["students"].add(attempt["student_name"])
                observation = entry.get("what_needs_improvement") or entry.get("misconception")
                if observation and observation not in gap["observations"]:
                    gap["observations"].append(observation)
                correction = entry.get("correct_answer")
                if correction and correction not in gap["correct_answers"]:
                    gap["correct_answers"].append(correction)
        if not gaps:
            st.info("В новых попытках явные повторяющиеся заблуждения не зафиксированы.")
        else:
            for gap in sorted(gaps.values(), key=lambda item: len(item["students"]), reverse=True):
                rule = gap["rule"]
                title = rule.get("title", "Проверяемый навык")
                students_text = ", ".join(sorted(gap["students"]))
                observations = " · ".join(gap["observations"][:2]) or "Нужен дополнительный разбор."
                corrections = " · ".join(gap["correct_answers"][:2]) or "См. карточку правила."
                st.markdown(
                    f'<div class="gap-card"><span class="count">{len(gap["students"])} чел.</span>'
                    f'<h4>{escape(title)}</h4><p><b>У кого:</b> {escape(students_text)}</p>'
                    f'<p><b>Что не получается:</b> {escape(observations)}</p>'
                    f'<p><b>Ориентир:</b> {escape(corrections)}</p></div>',
                    unsafe_allow_html=True,
                )
        if legacy_attempts:
            st.caption(
                f"{legacy_attempts} старых попыток не включены в точную диагностику: "
                "они были выполнены до обновления формата проверки."
            )
    with detail_right:
        st.subheader("Последние результаты")
        rows = [
            {
                "Студент": item["student_name"],
                "Вариант": item.get("assignment_title", "—"),
                "Задание": round((item.get("submission_score") or 0) * 100),
                "Viva": round(item["overall_score"] * 100),
                "Маршрут": "Проверка понимания" if item.get("submission_correct") else "Диагностика пробела",
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
    with st.expander("Управление заданиями"):
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
