# Эксперимент с Google DeepMind Concordia

## Что проверено

22 июля 2026 года Concordia 2.4.0 была запущена на Windows через локальный
`llama.cpp` VivaTrace без облачного API. Собственный адаптер находится в
`scripts/run_concordia_local.py`; он реализует интерфейс `LanguageModel` и
использует быстрый сервер Qwen2.5-3B на `127.0.0.1:8081`.

Сценарий `Philosophy Student Exam Prep` создал две сущности (студент Jordan и
AI-помощник Sage), Game Master, общее состояние и первый осмысленный ход студента.

## Результат

Полный runtime слишком тяжёл для синхронного MVP: ограниченная четырьмя ходами
симуляция не завершилась за семь минут на локальном оборудовании. Причина — много
внутренних LLM-вызовов на память, наблюдения, выбор действия и разрешение события.
Подключение Concordia целиком ухудшило бы уже заметную задержку Viva.

## Решение для продукта

Заимствуется не runtime, а архитектурный паттерн:

1. `Student` выполняет языковое действие в короткой миссии;
2. `NPC` отвечает в заданной роли;
3. облегчённый `Game Master` одним LLM-вызовом определяет последствие;
4. предметный grader отдельно проверяет целевую грамматическую конструкцию;
5. состояние миссии и evidence сохраняются в SQLite.

Такой модуль можно назвать «Практическая миссия». Он добавляет игровой перенос
навыка, но сохраняет главное свойство VivaTrace: воспроизводимая предметная оценка
не передаётся генеративной модели.

## Повтор запуска

```powershell
git clone --depth 1 https://github.com/google-deepmind/concordia.git external/concordia
python -m venv external/concordia/.venv
external/concordia/.venv/Scripts/python.exe -m pip install --editable external/concordia
external/concordia/.venv/Scripts/python.exe scripts/run_concordia_local.py --steps 4
```

Код Concordia имеет лицензию Apache 2.0 и не включён в основной runtime проекта.
