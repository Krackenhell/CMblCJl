from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from vivatrace.artifact import inspect_ml_artifact
from vivatrace.bkt import BKTModel, BKTParameters
from vivatrace.cohort import (
    cohort_skill_summary,
    intervention_for_gap,
    mastery_frame,
    misconception_summary,
)
from vivatrace.curriculum import load_curriculum
from vivatrace.database import (
    create_assignment,
    get_mastery,
    init_database,
    latest_attempts,
    list_assignments,
    list_students,
    reset_learning_data,
    save_attempt,
    student_attempts,
    update_assignment,
)
from vivatrace.demo import DATA_DIR
from vivatrace.evaluator import get_evaluator
from vivatrace.models import Curriculum, Evidence, StudentState
from vivatrace.routing import choose_route
from vivatrace.viva import follow_up_question, select_questions


ROOT = Path(__file__).resolve().parent
CURRICULUM = load_curriculum(DATA_DIR / "curriculum.json")
COLORS = {
    "ink": "#18231F",
    "muted": "#64726C",
    "green": "#176B50",
    "green_dark": "#103E30",
    "mint": "#DEF3E8",
    "lime": "#C9F26B",
    "amber": "#F4B860",
    "coral": "#EC7565",
    "paper": "#F5F6F1",
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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
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
.hero {
  border-radius: 24px; padding: 28px 32px; color: white; margin-bottom: 22px;
  background: radial-gradient(circle at 88% 18%, rgba(201,242,107,.32), transparent 25%),
              linear-gradient(120deg, #123D2F, #1A7053);
  box-shadow: 0 14px 40px rgba(23,61,48,.13);
}
.hero .eyebrow { color: #C9F26B; font-weight: 800; font-size: .76rem; letter-spacing: .11em; }
.hero h1 { color: white; margin: 7px 0 7px; font-size: 2.2rem; }
.hero p { color: #E7F1EC; max-width: 880px; margin: 0; line-height: 1.55; }
.card { background: white; border: 1px solid #E2E7E2; border-radius: 18px; padding: 18px 20px; margin-bottom: 13px; }
.card-title { font-weight: 800; font-size: 1rem; margin-bottom: 5px; }
.muted { color: #64726C; font-size: .82rem; line-height: 1.45; }
.metric-card { background: white; border: 1px solid #E2E7E2; border-radius: 18px; padding: 18px; min-height: 118px; }
.metric-card .label { color: #64726C; font-size: .73rem; font-weight: 800; text-transform: uppercase; letter-spacing: .055em; }
.metric-card .value { color: #18231F; font-size: 1.9rem; font-weight: 800; margin-top: 8px; }
.metric-card .hint { color: #64726C; font-size: .76rem; margin-top: 4px; }
.finding { border-left: 4px solid #EC7565; background: #FFF4F1; border-radius: 10px; padding: 12px 14px; margin: 9px 0; }
.finding-ok { border-left: 4px solid #176B50; background: #EFF8F3; border-radius: 10px; padding: 12px 14px; margin: 9px 0; }
.question { background: white; border: 1px solid #DDE5DF; border-radius: 20px; padding: 22px 24px; margin: 14px 0; }
.question-number { color: #176B50; font-size: .75rem; font-weight: 800; letter-spacing: .08em; }
.evidence { border-left: 4px solid #176B50; background: white; border-radius: 12px; padding: 14px 16px; margin: 10px 0; border-top: 1px solid #E2E7E2; border-right: 1px solid #E2E7E2; border-bottom: 1px solid #E2E7E2; }
.decision { background: #18362C; color: white; border-radius: 18px; padding: 20px 22px; }
.decision strong { color: #C9F26B; }
.decision h3 { color: white; margin: .35rem 0; }
.decision p { color: #E4EEE9; margin-bottom: 0; }
.chip { display: inline-block; background: #DEF3E8; color: #16533E; border-radius: 999px; padding: 5px 10px; margin: 2px 4px 2px 0; font-size: .74rem; font-weight: 700; }
.empty { background: white; border: 1px dashed #BCC9C2; border-radius: 22px; padding: 42px; text-align: center; }
.empty .icon { font-size: 2.2rem; color: #176B50; }
.progress-label { display:flex; justify-content:space-between; font-size:.8rem; color:#64726C; margin:6px 0; }
.stButton > button { border-radius: 12px; font-weight: 750; border: 0; background: #176B50; color: white; }
.stButton > button:hover { background: #124D3B; color: white; }
.stFormSubmitButton > button { border-radius: 12px; background: #176B50; color: white; font-weight: 750; }
[data-testid="stExpander"] { background: white; border: 1px solid #E2E7E2; border-radius: 16px; }
</style>
""",
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""<div class="metric-card"><div class="label">{label}</div>
        <div class="value">{value}</div><div class="hint">{hint}</div></div>""",
        unsafe_allow_html=True,
    )


def assignment_curriculum(assignment: dict) -> Curriculum:
    allowed = set(assignment["skill_ids"])
    return replace(CURRICULUM, skills=tuple(s for s in CURRICULUM.skills if s.id in allowed))


def reset_viva_flow() -> None:
    st.session_state.pop("viva_flow", None)
    st.session_state["attempt_nonce"] = st.session_state.get("attempt_nonce", 0) + 1


def render_sidebar() -> tuple[str, dict | None]:
    students = list_students()
    with st.sidebar:
        st.markdown('<div class="brand">◉ VivaTrace</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">Адаптивная обратная связь для учебной группы</div>', unsafe_allow_html=True)
        role = st.selectbox("Роль", ["Студент", "Преподаватель"])
        selected_student = None
        if role == "Студент":
            student_by_name = {student["name"]: student for student in students}
            selected_name = st.selectbox("Аккаунт студента", list(student_by_name))
            selected_student = student_by_name[selected_name]
            st.caption("Демонстрационное переключение аккаунтов без авторизации")
        else:
            st.markdown("**Кабинет преподавателя**")
            st.caption("Результаты обновляются после каждой завершённой защиты")
        st.markdown("---")
        st.caption("Курс")
        st.markdown(f"**{CURRICULUM.course_name}**")
    return role, selected_student


def start_viva(student: dict, assignment: dict, artifact: str) -> None:
    findings = inspect_ml_artifact(artifact)
    mastery = get_mastery(student["id"], assignment["skill_ids"])
    questions = select_questions(
        findings,
        mastery,
        limit=min(3, len(assignment["skill_ids"])),
        allowed_skills=assignment["skill_ids"],
        seed_key=f"{student['id']}:{assignment['id']}:{artifact}",
    )
    st.session_state.viva_flow = {
        "student_id": student["id"],
        "assignment_id": assignment["id"],
        "artifact": artifact,
        "findings": findings,
        "questions": questions,
        "current": 0,
        "evidence": [],
        "mastery": mastery,
        "completed": False,
        "followed_skills": set(),
    }


def render_findings(findings: list) -> None:
    if not findings:
        st.markdown(
            """<div class="finding-ok"><b>Явных технических сигналов не найдено</b><br>
            <span class="muted">Защита проверит ключевые понятия задания и способность обосновать решение.</span></div>""",
            unsafe_allow_html=True,
        )
        return
    for finding in findings:
        skill = CURRICULUM.skill_by_id[finding.skill_id]
        st.markdown(
            f"""<div class="finding"><b>{skill.name}</b><br>{finding.evidence}<br>
            <span class="muted">Что нужно уточнить: {finding.hypothesis}</span></div>""",
            unsafe_allow_html=True,
        )


def render_student(student: dict) -> None:
    assignments = list_assignments(active_only=True)
    if not assignments:
        hero("Пока нет активных заданий", "Преподаватель ещё не опубликовал задание.", "КАБИНЕТ СТУДЕНТА")
        return

    assignment_by_label = {
        f"{item['topic']} · {item['title']}": item for item in assignments
    }
    selected_label = st.selectbox("Задание", list(assignment_by_label), label_visibility="collapsed")
    assignment = assignment_by_label[selected_label]
    context = (student["id"], assignment["id"])
    if st.session_state.get("viva_context") != context:
        st.session_state.viva_context = context
        reset_viva_flow()

    hero(
        assignment["title"],
        assignment["instructions"],
        f"СТУДЕНТ · {student['name'].upper()} · {assignment['topic'].upper()}",
    )

    skill_names = [CURRICULUM.skill_by_id[item].name for item in assignment["skill_ids"]]
    st.markdown("".join(f'<span class="chip">{name}</span>' for name in skill_names), unsafe_allow_html=True)
    history = student_attempts(student["id"], assignment["id"])
    if history:
        st.caption(f"Завершённых попыток: {len(history)} · последняя оценка понимания: {history[0]['overall_score']:.0%}")
    else:
        st.caption("Это первая попытка по заданию")

    flow = st.session_state.get("viva_flow")
    if flow is None:
        st.subheader("1. Отправьте решение")
        artifact_key = f"artifact-{student['id']}-{assignment['id']}-{st.session_state.get('attempt_nonce', 0)}"
        artifact = st.text_area(
            "Код или развёрнутый ответ",
            value=assignment["starter_code"],
            height=390,
            key=artifact_key,
            help="Измените решение: вопросы защиты будут сформированы заново по его содержанию.",
        )
        if st.button("Проанализировать решение и начать защиту", width="stretch"):
            if len(artifact.strip()) < 30:
                st.warning("Добавьте решение или развёрнутый ответ перед началом защиты.")
            else:
                start_viva(student, assignment, artifact)
                st.rerun()
        return

    if flow["student_id"] != student["id"] or flow["assignment_id"] != assignment["id"]:
        reset_viva_flow()
        st.rerun()

    if not flow["completed"]:
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.subheader("2. Что обнаружено в решении")
            render_findings(flow["findings"])
            with st.expander("Показать отправленное решение"):
                st.code(flow["artifact"], language="python")
        with right:
            st.subheader("Ход защиты")
            total = len(flow["questions"])
            current = flow["current"]
            st.progress(current / max(total, 1))
            st.caption(f"Получено ответов: {current} из {total}")

        question = flow["questions"][flow["current"]]
        st.markdown(
            f"""<div class="question"><div class="question-number">ВОПРОС {flow['current'] + 1} ИЗ {len(flow['questions'])}</div>
            <h3>{question.text}</h3><div class="muted">{question.purpose}</div></div>""",
            unsafe_allow_html=True,
        )
        answer_key = f"answer-{student['id']}-{assignment['id']}-{question.id}-{flow['current']}"
        answer = st.text_area(
            "Ответ своими словами",
            key=answer_key,
            height=155,
            placeholder="Объясните ход рассуждений. Стиль речи и формулировки не оцениваются.",
        )
        buttons = st.columns([1, 3])
        with buttons[0]:
            if st.button("Изменить решение"):
                reset_viva_flow()
                st.rerun()
        with buttons[1]:
            if st.button("Ответить и продолжить", width="stretch"):
                if len(answer.strip()) < 12:
                    st.warning("Дайте короткое, но содержательное объяснение.")
                else:
                    evaluator, _ = get_evaluator()
                    evidence = evaluator.evaluate(question, answer)
                    model = BKTModel(BKTParameters())
                    flow["mastery"][question.skill_id] = model.update(
                        flow["mastery"].get(question.skill_id, model.params.prior),
                        evidence.score,
                    )
                    flow["evidence"].append(evidence)

                    if (
                        evidence.score < 0.45
                        and question.skill_id not in flow["followed_skills"]
                        and len(flow["questions"]) < 4
                    ):
                        follow_up = follow_up_question(
                            question,
                            seed_key=f"{student['id']}:{flow['artifact']}",
                        )
                        if follow_up:
                            flow["questions"].insert(flow["current"] + 1, follow_up)
                            flow["followed_skills"].add(question.skill_id)

                    flow["current"] += 1
                    if flow["current"] >= len(flow["questions"]):
                        save_attempt(
                            student_id=student["id"],
                            assignment_id=assignment["id"],
                            artifact=flow["artifact"],
                            findings=flow["findings"],
                            evidence=flow["evidence"],
                            mastery=flow["mastery"],
                        )
                        flow["completed"] = True
                    st.session_state.viva_flow = flow
                    st.rerun()
        return

    st.subheader("Защита завершена")
    st.success("Результат сохранён. Он уже появился в пульсе группы преподавателя.")
    evidence_by_skill = {item.skill_id: item for item in flow["evidence"]}
    for skill_id, mastery_value in flow["mastery"].items():
        skill = CURRICULUM.skill_by_id[skill_id]
        evidence = evidence_by_skill.get(skill_id)
        if not evidence:
            continue
        route = choose_route(skill, mastery_value, evidence)
        route_names = {
            "repair": "восстановить пробел",
            "practice": "закрепить навык",
            "transfer": "перейти к усложнению",
            "human_review": "проверка преподавателем",
        }
        st.markdown(
            f"""<div class="evidence"><b>{skill.name} · освоение {mastery_value:.0%}</b><br>
            «{evidence.quote}»<br><span class="muted">{evidence.rationale}</span><br>
            <b>Следующий шаг:</b> {route_names[route.route.value]} · {route.duration_minutes} минут</div>""",
            unsafe_allow_html=True,
        )

    if st.button("Начать новую попытку"):
        reset_viva_flow()
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
    assignment_by_label = {
        f"{item['topic']} · {item['title']}": item for item in assignments
    }
    selected_label = st.selectbox("Результаты по заданию", list(assignment_by_label))
    assignment = assignment_by_label[selected_label]
    attempts = latest_attempts(assignment["id"])
    students = list_students()

    hero(
        "Пульс учебной группы",
        "Данные появляются только после реальных попыток студентов. Здесь видно, что повторить всей группе, кому нужна адресная помощь и кто готов к усложнению.",
        f"ПРЕПОДАВАТЕЛЬ · {assignment['topic'].upper()}",
    )

    if not attempts:
        st.markdown(
            f"""<div class="empty"><div class="icon">◎</div><h2>Пока нет завершённых защит</h2>
            <p class="muted">Переключитесь на аккаунты студентов и завершите хотя бы одну защиту.<br>
            Пульс группы будет рассчитан из полученных ответов, без предзаполненных результатов.</p>
            <b>Пройдено: 0 из {len(students)}</b></div>""",
            unsafe_allow_html=True,
        )
    else:
        cohort = assignment_curriculum(assignment)
        states = attempts_to_states(attempts)
        summary = cohort_skill_summary(cohort, states)
        frame = mastery_frame(cohort, states)
        skill_ids = [skill.id for skill in cohort.skills]
        mean_mastery = frame[skill_ids].to_numpy().mean()
        student_means = frame[skill_ids].mean(axis=1)
        at_risk = int((student_means < 0.50).sum())
        advanced = int((student_means >= 0.78).sum())
        top_gap = summary.iloc[0]

        metric_cols = st.columns(4)
        with metric_cols[0]:
            metric_card("Завершили защиту", f"{len(attempts)} из {len(students)}", "учитываются последние попытки")
        with metric_cols[1]:
            metric_card("Среднее освоение", f"{mean_mastery:.0%}", "по навыкам задания")
        with metric_cols[2]:
            metric_card("Нужна помощь", str(at_risk), "студентов ниже порога 50%")
        with metric_cols[3]:
            metric_card("Главный пробел", f"{top_gap['Доля с пробелом']:.0%}", str(top_gap["Навык"]))

        st.write("")
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            st.subheader("Карта освоения навыков")
            heatmap = frame.set_index("student")[skill_ids]
            heatmap.columns = [skill.name for skill in cohort.skills]
            figure = px.imshow(
                heatmap,
                zmin=0,
                zmax=1,
                aspect="auto",
                color_continuous_scale=[
                    [0, COLORS["coral"]],
                    [0.45, COLORS["amber"]],
                    [0.8, COLORS["lime"]],
                    [1, COLORS["green"]],
                ],
                labels={"color": "Освоение"},
                text_auto=".0%",
            )
            figure.update_layout(
                height=max(280, 105 + 72 * len(attempts)),
                margin=dict(l=5, r=5, t=10, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar_tickformat=".0%",
            )
            figure.update_xaxes(side="top")
            figure.update_yaxes(title_text="Студент")
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        with right:
            st.subheader("Решение для следующей пары")
            decision = intervention_for_gap(float(top_gap["Доля с пробелом"]), str(top_gap["Навык"]))
            st.markdown(
                f"""<div class="decision"><strong>{decision['level']}</strong>
                <h3>{decision['decision']}</h3><p>{decision['format']}</p></div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown("**Распределение по траекториям**")
            st.write(f"Восстановление пробелов: **{at_risk}**")
            st.write(f"Закрепление: **{len(attempts) - at_risk - advanced}**")
            st.write(f"Усложнение: **{advanced}**")

        st.divider()
        lower_left, lower_right = st.columns([1, 1.25], gap="large")
        with lower_left:
            st.subheader("Повторяющиеся заблуждения")
            misconceptions = misconception_summary(states)
            if misconceptions.empty:
                st.info("Повторяющихся заблуждений пока не обнаружено.")
            else:
                st.dataframe(misconceptions, hide_index=True, width="stretch")
        with lower_right:
            st.subheader("Последние результаты")
            rows = [
                {
                    "Студент": item["student_name"],
                    "Понимание": round(item["overall_score"] * 100),
                    "Ошибок в работе": len(item["findings"]),
                    "Вопросов": len(item["evidence"]),
                }
                for item in attempts
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Понимание": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                    )
                },
            )

        with st.expander("Посмотреть ответы конкретного студента"):
            attempt_by_name = {item["student_name"]: item for item in attempts}
            detail_name = st.selectbox("Студент", list(attempt_by_name), key="teacher-detail")
            detail = attempt_by_name[detail_name]
            st.code(detail["artifact"], language="python")
            for entry in detail["evidence"]:
                skill_name = CURRICULUM.skill_by_id[entry["skill_id"]].name
                st.markdown(
                    f"**{skill_name} · ответ {entry['score']:.0%}**  \n"
                    f"> {entry['quote']}  \n{entry['rationale']}"
                )

    render_assignment_management(assignment)


def render_assignment_management(selected_assignment: dict) -> None:
    with st.expander("Управление заданиями"):
        mode = st.radio("Действие", ["Изменить текущее", "Создать новое"], horizontal=True)
        selected_skills = [skill.id for skill in CURRICULUM.skills]
        skill_labels = {skill.name: skill.id for skill in CURRICULUM.skills}
        if mode == "Изменить текущее":
            defaults = selected_assignment
            button_label = "Сохранить изменения"
        else:
            defaults = {
                "title": "",
                "topic": "",
                "instructions": "",
                "starter_code": "",
                "skill_ids": selected_skills,
            }
            button_label = "Создать задание"

        with st.form(f"assignment-form-{mode}-{selected_assignment['id']}"):
            title = st.text_input("Название", value=defaults["title"])
            topic = st.text_input("Тема", value=defaults["topic"])
            instructions = st.text_area("Условие задания", value=defaults["instructions"], height=130)
            starter_code = st.text_area("Начальный код или шаблон ответа", value=defaults["starter_code"], height=220)
            default_names = [
                CURRICULUM.skill_by_id[item].name
                for item in defaults["skill_ids"]
                if item in CURRICULUM.skill_by_id
            ]
            chosen_names = st.multiselect("Проверяемые навыки", list(skill_labels), default=default_names)
            submitted = st.form_submit_button(button_label, width="stretch")
        if submitted:
            if not title.strip() or not topic.strip() or not instructions.strip() or not chosen_names:
                st.error("Заполните название, тему, условие и выберите хотя бы один навык.")
            else:
                chosen_ids = [skill_labels[name] for name in chosen_names]
                if mode == "Изменить текущее":
                    update_assignment(
                        selected_assignment["id"], title, topic, instructions, starter_code, chosen_ids
                    )
                    st.success("Задание обновлено. Новые попытки будут использовать новые параметры.")
                else:
                    create_assignment(title, topic, instructions, starter_code, chosen_ids)
                    st.success("Новое задание создано и доступно студентам.")
                st.rerun()

        st.markdown("---")
        st.caption("Демонстрационные данные")
        confirm_reset = st.checkbox("Я понимаю, что все результаты трёх студентов будут удалены")
        if st.button("Очистить результаты", disabled=not confirm_reset):
            reset_learning_data()
            reset_viva_flow()
            st.success("Результаты очищены. Пульс группы снова пуст.")
            st.rerun()


init_database()
role, selected_student = render_sidebar()
if role == "Студент" and selected_student:
    render_student(selected_student)
else:
    render_teacher()
