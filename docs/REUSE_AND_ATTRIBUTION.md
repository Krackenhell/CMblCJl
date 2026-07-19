# Переиспользование и атрибуция

VivaTrace — самостоятельная интеграционная и продуктовая реализация. Мы не форкали и не маскировали интерфейс существующего продукта. Вместо этого использованы открытые научные идеи, совместимые библиотеки и явно указанные архитектурные паттерны.

## OATutor

- Репозиторий: <https://github.com/CAHLR/OATutor>
- Лицензия кода: MIT.
- Что изучено: четырёхпараметрическая BKT-модель, связь problem/step со skill model, выбор следующего задания по низкому mastery.
- Что реализовано нами: Python-модуль BKT с поддержкой непрерывного evidence score, связь с micro-viva и cohort routing.
- Чужой исходный код в проект не копировался построчно.

## MathDial

- Репозиторий: <https://github.com/eth-nlped/mathdial>
- Лицензия данных: CC BY-SA 4.0.
- Что изучено: tutoring должен направлять и диагностировать confusion, а не сразу раскрывать решение.
- Датасет MathDial в MVP не распространяется.

## MRBench / Unifying AI Tutor Evaluation

- Репозиторий: <https://github.com/kaushal0494/UnifyingAITutorEvaluation>
- Лицензия данных: CC BY-SA 4.0.
- Что изучено: измерения mistake identification, mistake location, guidance, answer revealing, actionability и coherence.
- В VivaTrace использована собственная компактная рубрика evidence; данные MRBench не копируются.

## Библиотеки

- Streamlit — интерфейс MVP;
- Plotly — интерактивная аналитика;
- scikit-learn — воспроизводимый baseline-эксперимент;
- Pydantic/OpenAI client — опциональный structured LLM adapter.

Полные тексты лицензий библиотек доступны в их пакетах. При подготовке публичного релиза следует сформировать lock-файл и автоматический software bill of materials.

## Как говорить об этом на защите

Корректная формулировка:

> «Мы не изобретали Bayesian Knowledge Tracing. Мы взяли проверенную модель из intelligent tutoring systems, ориентируясь на OATutor, и расширили её для непрерывных свидетельств из открытой micro-viva. Наша разработка — замкнутый контур artifact → viva → evidence → mastery → individual route → cohort intervention».

Некорректная формулировка: «Мы сами изобрели BKT» или «весь проект написан без использования существующих решений».

