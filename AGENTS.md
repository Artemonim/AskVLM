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
- - `stt_pool/` — STT-супервизор с пулом воркеров для `external-transcribe` (см. `doc/EXTERNAL_CLI_TRANSCRIBER.md`): `client.py` (тонкий клиент, ML-free: пачка входов, один дедлайн, `--priority`, `--max-parallel`), `queue.py`/`fsio.py` (файловая очередь v2 в `<root>/v2`, OS-lock супервизора, heartbeat), `config.py`/`keys.py` (лимиты из `stt_pool.json` → `ASKVLM_STT_POOL_CONFIG` → defaults; ключ воркера provider/model/compute/backend), `scheduler.py`/`estimates.py` (чистый планировщик: резервы слотов/потоков/commit/VRAM, spill `auto` на CPU по ETA, эвикция idle-воркеров других конфигураций, приоритеты с чередованием вызовов и старением; seed-оценки из бенчмарка + выученные), `supervisor.py` (цикл супервизора), `procs.py`/`winjob.py`/`ipc.py`/`sysprobe.py` (процесс-воркер в своём Job Object с лимитом памяти, JSON-lines протокол, commit/nvidia-smi пробы), `worker.py`/`engine_api.py`/`engines.py` (процесс воркера; CPU-слоты = потоки над одной моделью, GPU = батчинг между заданиями). `--no-daemon` — legacy one-shot. `--stt-provider whisper|gigaam-ctc|gigaam-e2e-ctc` (Whisper по умолчанию; GigaAM в `.[ml]`; `gigaam-ctc` только CPU; `run.ps1`/`build.ps1` по умолчанию держат torch 2.10+CUDA).
- - `stt_providers.py` / `gigaam_device.py` / `gigaam_ctc_wrapper.py` / `gigaam_e2e_ctc_wrapper.py` / `whisperx_wrapper.py` / `whisper_batch.py` / `stt_errors.py` — torch-free контракт провайдеров и проверка `--device`; политика CUDA для GigaAM (порог свободной VRAM для `auto`, распознавание CUDA OOM, освобождение кэша); lazy wrapper `ai-sage/GigaAM-Multilingual` revision `ctc` (только CPU) с общим device-механизмом (загрузка на CPU → перенос на CUDA, OOM → CPU с повтором окна, `empty_cache` при выгрузке); наследник для `ai-sage/GigaAM-v3` revision `e2e_ctc` (патчи transformers 5.x на загрузке, fp32, CUDA разрешена). Opt-in режимы для пула: `strict_device` (без CUDA→CPU fallback, типизированные `CudaOutOfMemoryError`/`DeviceUnavailableError`), число потоков, потокобезопасный путь GigaAM и батч окон, батч клипов Whisper.
- `editing/` — text model и editing tools.
- `utils/` — exporters, logging и общие helpers.
- `tests/` — автоматические проверки и E2E сценарии.

## Решения по STT-пулу (не поднимать повторно без новых данных)
- CPU-воркер — процесс с одной моделью и потоками-слотами; GPU-воркер — батчинг между заданиями; GigaAM на CUDA из нескольких потоков запрещён (порча качества по бенчмарку).
- В пуле устройство строгое: `cpu` никогда не на GPU, `cuda` на CPU только при подтверждённом отсутствии CUDA (с предупреждением), `auto` уходит на CPU по ETA (порог 0.8). Клиентского CPU-recovery нет.
- Имена моделей Whisper не меняют регистр (`auto` → `large-v3`); приоритет: класс со старением каждые `priority_aging_s`, внутри класса чередование вызовов; начатые задания не вытесняются.
- Тесты пула используют только фейковые движки и временные корни очередей; реальные модели — только в `heavy_ml` тестах по одной за раз.
- Число воркеров pytest-xdist по умолчанию 3 (`-n 3` в `pyproject.toml`, переопределение — `-TestWorkers N`). Прежний `-n auto` давал ~16 воркеров и 40–51 ГБ commit и однажды исчерпал лимит commit системы; измеренный пик дерева pytest на `-Fast` с 3 воркерами — ~6.4 ГБ commit. Полный (не-`-Fast`) прогон дополнительно гоняет `heavy_ml` с реальными моделями: перед ним проверяй свободный commit.

## Границы
- Не добавляй секреты, токены и машинно-зависимые пути в код или документацию.
- Если есть неочевидное исключение или новое правило, зафиксируй его здесь или в `.cursor/rules/`.

## Контроль качества
- Основная команда проверки `. .venv/Scripts/Activate.ps1 && ./run.ps1 -SkipLaunch -Fast` обеспечивает быстрый чекап.
- - Используй `./run.ps1 -SkipLaunch` для запуска всех проверок.
- - Ты обязан запустить хотя бы одну из этих команд в конце своей работы для верификации качества кода.
- - Запускай проверки иными способами только если ранера недостаточно.
- - `-TestWorkers N` — число воркеров pytest-xdist (по умолчанию 3; `N` — целое ≥ 0 или `auto`; `0` — прогон в одном процессе без воркеров, например под отладчиком). `auto` на этой машине даёт ~16 воркеров и 40–51 ГБ commit: не использовать без нужды.
- - `-Cli` — исключает тесты, которым нужен Qt GUI (`pytest -m "not gui"`); сочетается с `-Fast` в одно `-m`-выражение. Тесты, которым нужен Qt (`QApplication`, `MainWindow`, `qtbot`, код из `gui/`), обязаны иметь маркер `gui`; линтеры флаг не затрагивает. Флаг не гарантирует, что Qt вообще не импортируется: часть немаркированных тестов импортирует `PySide6` или создаёт `QApplication`, не используя его.
- - Тесты идут headless: `tests/conftest.py` ставит `QT_QPA_PLATFORM=offscreen` до создания `QApplication`, окна не показываются. Для отладки с видимыми окнами задай `ASKVLM_TESTS_SHOW_GUI=1` (заранее заданный `QT_QPA_PLATFORM` тоже сохраняется).
- Запускай терминал с `block_until_ms=600000` или больше.
- Если терминал не уложился в таймаут, итеративно решай убивать или ждать: для ожидания запускай паузу на 5 минут.
