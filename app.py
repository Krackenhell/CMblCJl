from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from vivatrace.demo import DATA_DIR, DEMO_ASSIGNMENT, DEMO_SUBMISSION, load_demo_students
from vivatrace.evaluator import get_evaluator
from vivatrace.models import Evidence, StudentState
from vivatrace.routing import choose_route
from vivatrace.viva import select_questions


ROOT = Path(__file__).resolve().parent
CURRICULUM = load_curriculum(DATA_DIR / "curriculum.json")
COLORS = {
    "ink": "#17231F",
    "muted": "#65736D",
    "green": "#1D6B4F",
    "mint": "#DDF4E8",
    "lime": "#C8F169",
    "amber": "#F4B860",
    "coral": "#EF7A67",
    "paper": "#F7F7F2",
}


st.set_page_config(
    page_title="VivaTrace Classroom",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: #F7F7F2; color: #17231F; }
[data-testid="stSidebar"] { background: #12271F; }
[data-testid="stSidebar"] * { color: #F7F7F2 !important; }
[data-testid="stSidebar"] .stRadio label { padding: .55rem .6rem; border-radius: 10px; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3 { letter-spacing: -0.035em; color: #17231F; }
.hero {
  border-radius: 24px; padding: 30px 34px; color: white; margin-bottom: 22px;
  background: radial-gradient(circle at 82% 20%, rgba(200,241,105,.35), transparent 26%),
              linear-gradient(120deg, #173D30, #1D6B4F);
  box-shadow: 0 16px 45px rgba(23,61,48,.15);
}
.hero .eyebrow { color: #C8F169; font-weight: 800; font-size: .78rem; letter-spacing: .12em; }
.hero h1 { color: white; margin: 8px 0 8px; font-size: 2.35rem; }
.hero p { color: #E8F2ED; max-width: 850px; margin: 0; font-size: 1rem; line-height: 1.55; }
.metric-card { background: white; border: 1px solid #E4E8E3; border-radius: 18px; padding: 18px; min-height: 118px; }
.metric-card .label { color: #65736D; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.metric-card .value { color: #17231F; font-size: 2rem; line-height: 1.15; font-weight: 800; margin-top: 9px; }
.metric-card .hint { color: #65736D; font-size: .76rem; margin-top: 5px; }
.panel { background: white; border: 1px solid #E4E8E3; border-radius: 20px; padding: 20px 22px; margin-bottom: 15px; }
.finding { border-left: 4px solid #EF7A67; background: #FFF5F2; border-radius: 9px; padding: 12px 14px; margin: 9px 0; }
.evidence { border-left: 4px solid #1D6B4F; background: #F0F8F4; border-radius: 9px; padding: 12px 14px; margin: 9px 0; }
.decision { background: #17231F; color: white; border-radius: 18px; padding: 20px 22px; }
.decision strong { color: #C8F169; }
.decision p { color: #E6EEE9; margin-bottom: 0; }
.chip { display: inline-block; background: #DDF4E8; color: #17543E; border-radius: 999px; padding: 5px 10px; margin: 2px 4px 2px 0; font-size: .76rem; font-weight: 700; }
.small-note { color: #65736D; font-size: .82rem; }
.stButton > button { border-radius: 12px; font-weight: 700; border: 0; background: #1D6B4F; color: white; }
.stButton > button:hover { background: #164E3B; color: white; }
[data-testid="stMetric"] { background: white; border: 1px solid #E4E8E3; border-radius: 16px; padding: 14px; }
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


def init_state() -> None:
    if "students" not in st.session_state:
        st.session_state.students = load_demo_students()
    if "live_student" not in st.session_state:
        base = st.session_state.students[0]
        st.session_state.live_student = StudentState(
            student_id=base.student_id,
            name=base.name,
            mastery=dict(base.mastery),
            evidence=list(base.evidence),
        )
    if "viva_result" not in st.session_state:
        st.session_state.viva_result = None


def effective_students() -> list[StudentState]:
    return [
        st.session_state.live_student if student.student_id == "s01" else student
        for student in st.session_state.students
    ]


def render_teacher_dashboard() -> None:
    hero(
        "Пульс группы после темы",
        "Не список оценок, а карта того, что группа действительно поняла — и решение для следующей пары.",
        "VIVATRACE · ПРЕПОДАВАТЕЛЬ",
    )
    students = effective_students()
    summary = cohort_skill_summary(CURRICULUM, students)
    frame = mastery_frame(CURRICULUM, students)
    mean_mastery = frame[[skill.id for skill in CURRICULUM.skills]].to_numpy().mean()
    at_risk = int((frame[[skill.id for skill in CURRICULUM.skills]].mean(axis=1) < 0.5).sum())
    advanced = int((frame[[skill.id for skill in CURRICULUM.skills]].mean(axis=1) >= 0.78).sum())
    top_gap = summary.iloc[0]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Среднее освоение", f"{mean_mastery:.0%}", "по 5 навыкам темы")
    with cols[1]:
        metric_card("Нужна поддержка", str(at_risk), "студента с накопленным пробелом")
    with cols[2]:
        metric_card("Готовы к transfer", str(advanced), "студента опережают темп")
    with cols[3]:
        metric_card("Главный пробел", f"{top_gap['Доля с пробелом']:.0%}", str(top_gap["Навык"]))

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Карта знаний группы")
        heatmap = frame.set_index("student")[[skill.id for skill in CURRICULUM.skills]]
        heatmap.columns = [skill.name for skill in CURRICULUM.skills]
        fig = px.imshow(
            heatmap,
            zmin=0,
            zmax=1,
            aspect="auto",
            color_continuous_scale=[
                [0, "#EF7A67"],
                [0.45, "#F4B860"],
                [0.8, "#C8F169"],
                [1, "#1D6B4F"],
            ],
            labels={"color": "Освоение"},
        )
        fig.update_layout(
            height=445,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar_tickformat=".0%",
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, width="stretch")
        st.caption("Красный — repair, жёлтый — закрепление, зелёный — готовность к усложнению.")

    with right:
        st.subheader("Приоритеты интервенции")
        gap_fig = px.bar(
            summary.sort_values("Доля с пробелом"),
            x="Доля с пробелом",
            y="Навык",
            orientation="h",
            text_auto=".0%",
            color="Доля с пробелом",
            color_continuous_scale=["#C8F169", "#F4B860", "#EF7A67"],
        )
        gap_fig.update_layout(
            height=330,
            margin=dict(l=8, r=8, t=8, b=8),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_tickformat=".0%",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gap_fig, width="stretch")

        decision = intervention_for_gap(float(top_gap["Доля с пробелом"]), str(top_gap["Навык"]))
        st.markdown(
            f"""<div class="decision"><strong>{decision['level']}</strong>
            <h3 style="color:white;margin:.35rem 0">{decision['decision']}</h3>
            <p>{decision['format']}</p></div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    lower_left, lower_right = st.columns([1, 1.2], gap="large")
    with lower_left:
        st.subheader("Повторяющиеся misconceptions")
        misconceptions = misconception_summary(students)
        if misconceptions.empty:
            st.info("После новых viva здесь появятся повторяющиеся заблуждения.")
        else:
            st.dataframe(misconceptions, hide_index=True, width="stretch")
        st.caption("Агрегируются концептуальные ошибки, а не формулировки или стиль речи.")
    with lower_right:
        st.subheader("Группы на следующий учебный шаг")
        route_rows = []
        for _, row in frame.iterrows():
            mean = row[[skill.id for skill in CURRICULUM.skills]].mean()
            route = "Repair" if mean < 0.5 else "Practice" if mean < 0.78 else "Transfer"
            route_rows.append({"Студент": row["student"], "Среднее": mean, "Маршрут": route})
        routes = pd.DataFrame(route_rows)
        st.dataframe(
            routes.sort_values(["Маршрут", "Среднее"]),
            hide_index=True,
            width="stretch",
            column_config={"Среднее": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0%%")},
        )


def render_student_viva() -> None:
    hero(
        "Защити решение — не пересдавай работу",
        "Три коротких вопроса проверят логику твоего решения. VivaTrace не оценивает красноречие и показывает, на каких свидетельствах основан вывод.",
        "VIVATRACE · СТУДЕНТ",
    )
    student = st.session_state.live_student
    evaluator, evaluator_name = get_evaluator()
    st.caption(f"Режим оценивания: {evaluator_name}")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Задание и работа")
        st.markdown(f"<div class='panel'><b>{DEMO_ASSIGNMENT}</b></div>", unsafe_allow_html=True)
        submission = st.text_area(
            "Отправленное решение",
            value=DEMO_SUBMISSION,
            height=330,
            label_visibility="collapsed",
        )
    with right:
        st.subheader("Что заметила система")
        findings = inspect_ml_artifact(submission)
        if findings:
            for finding in findings:
                skill_name = CURRICULUM.skill_by_id[finding.skill_id].name
                st.markdown(
                    f"""<div class="finding"><b>{skill_name}</b><br>{finding.evidence}<br>
                    <span class="small-note">Гипотеза для viva: {finding.hypothesis} · уверенность {finding.confidence:.0%}</span></div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success("Явных сигналов не найдено. Вопросы будут выбраны по карте знаний.")
        st.info("Это гипотезы для уточнения, а не обвинение и не финальная оценка.")

    questions = select_questions(findings, student.mastery, limit=3)
    st.divider()
    st.subheader("Micro-viva · около 4 минут")
    st.markdown(
        "<span class='chip'>1 вопрос за раз</span><span class='chip'>можно отвечать своими словами</span>"
        "<span class='chip'>есть письменный режим</span>",
        unsafe_allow_html=True,
    )

    example_answers = {
        "leakage-scaler": (
            "Scaler считает среднее и стандартное отклонение. Если обучить его до split, "
            "статистики test попадут в preprocessing и оценка станет оптимистичной. "
            "Нужно сначала разделить данные и fit scaler только на train, лучше через Pipeline."
        ),
        "metrics-imbalance": (
            "При дисбалансе модель может предсказывать только частый класс и получить высокую accuracy. "
            "Нужно смотреть precision, recall и F1, а выбор зависит от цены false positive и false negative."
        ),
        "reproducibility-seed": (
            "Изменится случайное разбиение и, возможно, параметры модели. Нужно задать random_state, "
            "зафиксировать версии данных и библиотек и сохранить конфигурацию эксперимента."
        ),
        "split-purpose": (
            "Train обучает параметры, validation выбирает гиперпараметры, а test используется один раз "
            "для независимой финальной оценки. Иначе мы подгоним решение под test."
        ),
    }
    demo_fill = st.toggle("Подставить сильные демонстрационные ответы", value=False)
    with st.form("viva-form"):
        answers: dict[str, str] = {}
        for index, question in enumerate(questions, start=1):
            st.markdown(f"**{index}. {question.text}**")
            st.caption(question.purpose)
            answers[question.id] = st.text_area(
                f"Ответ {index}",
                value=example_answers.get(question.id, "") if demo_fill else "",
                key=f"answer-{question.id}-{demo_fill}",
                height=105,
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Завершить micro-viva", width="stretch")

    if submitted:
        if any(not answer.strip() for answer in answers.values()):
            st.warning("Ответь на каждый вопрос — короткого объяснения достаточно.")
        else:
            model = BKTModel(BKTParameters())
            evidence_list: list[Evidence] = []
            mastery = dict(student.mastery)
            for question in questions:
                evidence = evaluator.evaluate(question, answers[question.id])
                evidence_list.append(evidence)
                mastery[question.skill_id] = model.update(
                    mastery.get(question.skill_id, model.params.prior), evidence.score
                )
            st.session_state.live_student = StudentState(
                student_id=student.student_id,
                name=student.name,
                mastery=mastery,
                evidence=student.evidence + evidence_list,
            )
            st.session_state.viva_result = evidence_list
            st.rerun()

    if st.session_state.viva_result:
        st.divider()
        st.subheader("Результат с доказательствами")
        results: list[Evidence] = st.session_state.viva_result
        for evidence in results:
            skill = CURRICULUM.skill_by_id[evidence.skill_id]
            mastery = st.session_state.live_student.mastery[evidence.skill_id]
            route = choose_route(skill, mastery, evidence)
            st.markdown(
                f"""<div class="evidence"><b>{skill.name}: {mastery:.0%}</b><br>
                «{evidence.quote}»<br><span class="small-note">{evidence.rationale}</span><br>
                <b>Следующий шаг:</b> {route.title} · {route.duration_minutes} мин</div>""",
                unsafe_allow_html=True,
            )

        mastery_data = pd.DataFrame(
            {
                "Навык": [skill.name for skill in CURRICULUM.skills],
                "Освоение": [
                    st.session_state.live_student.mastery.get(skill.id, 0.35)
                    for skill in CURRICULUM.skills
                ],
                "Цель": [skill.target_mastery for skill in CURRICULUM.skills],
            }
        )
        fig = go.Figure()
        fig.add_bar(
            x=mastery_data["Навык"],
            y=mastery_data["Освоение"],
            name="Текущее освоение",
            marker_color=COLORS["green"],
        )
        fig.add_scatter(
            x=mastery_data["Навык"],
            y=mastery_data["Цель"],
            name="Цель темы",
            mode="markers",
            marker=dict(color=COLORS["coral"], size=11, symbol="diamond"),
        )
        fig.update_layout(
            height=340,
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=15, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        if st.button("Пройти viva заново"):
            st.session_state.viva_result = None
            st.rerun()


def render_research() -> None:
    hero(
        "Доказательство ценности",
        "Сравниваем оценку только по сданной работе с гибридом «работа + micro-viva». Эксперимент воспроизводится одной командой.",
        "VIVATRACE · DATA SCIENCE",
    )
    metrics_path = ROOT / "artifacts" / "experiment_metrics.json"
    if not metrics_path.exists():
        st.warning("Эксперимент ещё не запущен. Выполни: python scripts/run_experiment.py")
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = payload["models"]["assignment_only"]
    hybrid = payload["models"]["assignment_plus_viva"]
    cols = st.columns(3)
    with cols[0]:
        metric_card("Balanced accuracy", f"{hybrid['balanced_accuracy']:.1%}", f"baseline {baseline['balanced_accuracy']:.1%}")
    with cols[1]:
        metric_card("F1 понимания", f"{hybrid['f1']:.1%}", f"baseline {baseline['f1']:.1%}")
    with cols[2]:
        gain = hybrid["roc_auc"] - baseline["roc_auc"]
        metric_card("Прирост ROC-AUC", f"+{gain:.2f}", "после добавления viva-сигнала")

    comparison = pd.DataFrame(
        [
            {"Модель": "Только работа", **baseline},
            {"Модель": "Работа + micro-viva", **hybrid},
        ]
    )
    chart_data = comparison.melt(
        id_vars="Модель",
        value_vars=["balanced_accuracy", "f1", "roc_auc"],
        var_name="Метрика",
        value_name="Значение",
    )
    fig = px.bar(
        chart_data,
        x="Метрика",
        y="Значение",
        color="Модель",
        barmode="group",
        text_auto=".1%",
        color_discrete_sequence=[COLORS["amber"], COLORS["green"]],
    )
    fig.update_layout(
        height=430,
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
    st.info(
        "Данные синтетические и используются для проверки пайплайна, а не как доказательство "
        "эффекта на реальных студентах. Следующий этап — пилот и экспертная разметка ответов."
    )
    with st.expander("Методика эксперимента"):
        st.json(payload["dataset"])


def render_architecture() -> None:
    hero(
        "Объяснимая AI-архитектура",
        "LLM не принимает непрозрачное финальное решение. Каждый модуль имеет ограниченную ответственность, а преподаватель сохраняет контроль.",
        "VIVATRACE · КАК ЭТО РАБОТАЕТ",
    )
    modules = [
        ("01", "Artifact Inspector", "Ищет в работе проверяемые сигналы: leakage, метрики, воспроизводимость."),
        ("02", "Viva Planner", "Выбирает 3 вопроса из curriculum graph и конкретной работы."),
        ("03", "Evidence Evaluator", "Оценивает по рубрике и сохраняет цитату-основание."),
        ("04", "BKT Learner Model", "Обновляет вероятность освоения каждого навыка."),
        ("05", "Routing Engine", "Назначает repair, practice, transfer или human review."),
        ("06", "Cohort Intelligence", "Агрегирует пробелы и предлагает интервенцию преподавателю."),
    ]
    cols = st.columns(3)
    for index, (number, name, description) in enumerate(modules):
        with cols[index % 3]:
            st.markdown(
                f"""<div class="panel"><span class="chip">{number}</span><h3>{name}</h3>
                <p>{description}</p></div>""",
                unsafe_allow_html=True,
            )
    st.subheader("Границы автоматизации")
    st.markdown(
        """
- Система проверяет понимание, но не объявляет факт списывания.
- Низкая уверенность маршрутизируется преподавателю без автоматического штрафа.
- Речь и стиль не входят в оценку предметного знания.
- Студент видит свидетельство, рубрику и может оспорить результат.
- Преподаватель подтверждает карту навыков и учебные материалы.
"""
    )


init_state()
with st.sidebar:
    st.markdown("## ◉ VivaTrace")
    st.caption("Adaptive feedback loop")
    st.write("")
    page = st.radio(
        "Режим",
        ["Пульс группы", "Micro-viva", "DS-эксперимент", "Архитектура"],
        label_visibility="collapsed",
    )
    st.write("")
    st.markdown("---")
    st.caption(CURRICULUM.course_name)
    st.markdown(f"**{CURRICULUM.topic_name}**")
    st.caption("Демо-группа · 12 студентов")

if page == "Пульс группы":
    render_teacher_dashboard()
elif page == "Micro-viva":
    render_student_viva()
elif page == "DS-эксперимент":
    render_research()
else:
    render_architecture()
