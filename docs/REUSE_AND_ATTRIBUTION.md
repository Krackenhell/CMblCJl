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

## Google DeepMind Concordia

- Репозиторий: <https://github.com/google-deepmind/concordia>.
- Лицензия: Apache 2.0.
- Что изучено: паттерн генеративной симуляции `Entities → Game Master → Engine`,
  где действия на естественном языке меняют состояние сценария.
- Что сделано нами: отдельный локальный smoke-test через собственный адаптер
  llama.cpp (`scripts/run_concordia_local.py`). Репозиторий не входит в основной
  runtime VivaTrace и не выдаётся за собственную разработку.
- Продуктовый вывод: для MVP выгоднее перенести паттерн короткой учебной миссии,
  чем подключать тяжёлую многоагентную симуляцию на каждом задании.

## Библиотеки

- Streamlit — интерфейс MVP;
- Plotly — интерактивная аналитика;
- scikit-learn — воспроизводимый baseline-эксперимент;
- Qwen2.5-7B-Instruct-GGUF — локальная instruction-модель, источник: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF>.
- Qwen2.5-3B-Instruct-GGUF — быстрая локальная instruction-модель, источник: <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF>.
- llama.cpp — локальный inference runtime и совместимый HTTP server, MIT: <https://github.com/ggml-org/llama.cpp>.
- whisper.cpp — локальный ASR runtime, MIT: <https://github.com/ggml-org/whisper.cpp>.
- Whisper base.en — ASR-модель OpenAI в формате ggml для whisper.cpp;
- Silero VAD — локальное обнаружение речи, MIT: <https://github.com/snakers4/silero-vad>.
- LanguageTool 6.6 — локальный grammar checker, LGPL-2.1-or-later:
  <https://github.com/languagetool-org/languagetool>.
- Eclipse Temurin JRE 17 — переносимый Java runtime для LanguageTool:
  <https://adoptium.net/temurin/>.
- WebSockets — двунаправленный транспорт между браузером и локальным voice server:
  <https://websockets.readthedocs.io/>.

Веса модели и бинарные файлы не маскируются под собственную разработку. Их точные
версии, источники и SHA-256 фиксируются в `scripts/setup_local_llm.ps1`,
`scripts/setup_local_voice.ps1` и локальных манифестах. Собственная часть проекта —
учебный workflow, предметные контракты и structural rules, полнодуплексный voice-
оркестратор, BKT, хранилище, аудит и интерфейсы студента/преподавателя.

Полные тексты лицензий библиотек доступны в их пакетах. При подготовке публичного релиза следует сформировать lock-файл и автоматический software bill of materials.

## Как говорить об этом на защите

Корректная формулировка:

> «Мы не изобретали Bayesian Knowledge Tracing. Мы взяли проверенную модель из intelligent tutoring systems, ориентируясь на OATutor, и расширили её для непрерывных свидетельств из открытой micro-viva. Наша разработка — замкнутый контур artifact → viva → evidence → mastery → individual route → cohort intervention».

Некорректная формулировка: «Мы сами изобрели BKT» или «весь проект написан без использования существующих решений».
