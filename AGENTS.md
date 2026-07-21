# AGENTS.md

## Доктрина
- Допустима только одна активная нейросеть. Остальные используемые в процессе нейросети должны выгружаться в RAM или выгружаться насовсем

## Куда смотреть
- `README.md` — краткий старт для человека.
- `TODO.md` — актуальный ближайший roadmap.
- `TODO_legacy.md` — архив старого long-form плана.
- `doc/Multimodal GUI Design.md` — текущая продуктовая рамка мультимодального GUI.
- `run.ps1` и `build.ps1` — источник истины для локальной проверки и запуска.

## Code Map
- `gui/` — окно, маршрутизация режимов и worker orchestration.
- `core/` — pipeline, FFmpeg, STT, diarization и settings.
- - `external_queue.py` / `external_daemon.py` / `external_client.py` — single-orchestrator транскрипция: файловая очередь, резидентный демон-пул и тонкий клиент для `external-transcribe` (см. `doc/EXTERNAL_CLI_TRANSCRIBER.md`). По умолчанию `external-transcribe` идёт через демон; `--no-daemon` — legacy one-shot. `--stt-provider whisper|gigaam-ctc` (Whisper по умолчанию; GigaAM CTC — CPU-only optional extra `.[gigaam]`; mismatch живого демона → unavailable).
- - `stt_providers.py` / `gigaam_ctc_wrapper.py` — контракт провайдеров STT и lazy CPU wrapper для `ai-sage/GigaAM-Multilingual` revision `ctc`.
- `editing/` — text model и editing tools.
- `utils/` — exporters, logging и общие helpers.
- `tests/` — автоматические проверки и E2E сценарии.

## Границы
- Не добавляй секреты, токены и машинно-зависимые пути в код или документацию.
- Если есть неочевидное исключение или новое правило, зафиксируй его здесь или в `.cursor/rules/`.

## Контроль качества
- Основная команда проверки `. .venv/Scripts/Activate.ps1 && ./run.ps1 -SkipLaunch -Fast` обеспечивает быстрый чекап.
- - Используй `./run.ps1 -SkipLaunch` для запуска всех проверок.
- - Ты обязан запустить хотя бы одну из этих команд в конце своей работы для верификации качества кода.
- - Запускай проверки иными способами только если ранера недостаточно.
- Запускай терминал с `block_until_ms=600000` или больше.
- Если терминал не уложился в таймаут, итеративно решай убивать или ждать: для ожидания запускай паузу на 5 минут.
