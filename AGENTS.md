# AGENTS.md

## Доктрина
- Учитывать доступную RAM и VRAM при загрузке нейромоделей.

## Куда смотреть
- `README.md` — краткий старт для человека.
- `TODO.md` — актуальный ближайший roadmap.
- `TODO_legacy.md` — архив старого long-form плана.
- `doc/Multimodal GUI Design.md` — текущая продуктовая рамка мультимодального GUI.
- `run.ps1` и `build.ps1` — источник истины для локальной проверки и запуска.

## Code Map
- `gui/` — окно, маршрутизация режимов и worker orchestration.
- `core/` — pipeline, FFmpeg, STT, diarization и settings.
- - `external_queue.py` / `external_daemon.py` / `external_client.py` — single-orchestrator транскрипция: файловая очередь, резидентный демон-пул и тонкий клиент для `external-transcribe` (см. `doc/EXTERNAL_CLI_TRANSCRIBER.md`). По умолчанию `external-transcribe` идёт через демон; `--no-daemon` — legacy one-shot. `--stt-provider whisper|gigaam-ctc|gigaam-e2e-ctc` (Whisper по умолчанию; GigaAM в составе `.[ml]`; `gigaam-ctc` = Multilingual @ `ctc` без пунктуации, только CPU; `gigaam-e2e-ctc` = v3 @ `e2e_ctc` с пунктуацией, `--device auto|cpu|cuda` с CUDA→CPU fallback, Windows child-isolation и client CPU-recovery как у Whisper; `run.ps1`/`build.ps1` по умолчанию держат torch 2.10+CUDA; provider mismatch живого демона → unavailable, device mismatch → только предупреждение `ASKVLM_DAEMON_DEVICE_MISMATCH`).
- - `stt_providers.py` / `gigaam_device.py` / `gigaam_ctc_wrapper.py` / `gigaam_e2e_ctc_wrapper.py` — torch-free контракт провайдеров и проверка `--device`; политика CUDA для GigaAM (порог свободной VRAM для `auto`, распознавание CUDA OOM, освобождение кэша); lazy wrapper `ai-sage/GigaAM-Multilingual` revision `ctc` (только CPU) с общим device-механизмом (загрузка на CPU → перенос на CUDA, OOM → CPU с повтором окна, `empty_cache` при выгрузке); наследник для `ai-sage/GigaAM-v3` revision `e2e_ctc` (патчи transformers 5.x на загрузке, fp32, CUDA разрешена).
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
