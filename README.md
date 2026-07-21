 [🇷🇺 Русский](README.ru.md) | [🇬🇧 English](README.md)

# AskVLM

Десктопный AI-инструмент для транскрипции речи, генерации субтитров и мультимодального анализа видео (Video QA). Построен на локальных ML-пайплайнах с опциональной поддержкой облачных LLM.

## Обзор

AskVLM объединяет три рабочих процесса в одном PySide6-приложении:

- **Текстовый режим** — транскрипция аудио/видео в текст с опциональной диаризацией говорящих и LLM-форматированием.
- **Режим субтитров** — генерация субтитров (SRT/VTT) с правилами читаемости, предпросмотр и прожиг в видео через FFmpeg.
- **Режим Video QA** — задайте вопрос о видео на естественном языке; AskVLM разбивает видео на чанки, извлекает репрезентативные кадры, запускает ASR и обращается к VLM (LM Studio или OpenRouter) для формирования обоснованного ответа.

## Быстрый старт

### Требования

- **Python**: 3.11.x (обязательно; 3.12+ не поддерживается)
- **ОС**: Windows (основная), Linux (community)
- **RAM**: минимум 8 ГБ, рекомендуется 16 ГБ+
- **GPU**: NVIDIA с CUDA (опционально, есть fallback на CPU)
- **VRAM**: минимум 4 ГБ, рекомендуется 8 ГБ+ для крупных моделей
- **Диск**: 2 ГБ+ под модели и временные файлы

См. [PYTHON_SETUP.md](PYTHON_SETUP.md) для установки или выбора нужной версии Python.

### Установка

```bash
git clone <repository-url>
cd AskVLM

# Базовые зависимости
pip install -e .

# ML-зависимости (транскрипция, диаризация, LLM, GigaAM CTC)
pip install -e .[ml]
# GPU: run.ps1 / build.ps1 по умолчанию чинят torch до 2.10+CUDA (cu128/cu126)
.\run.ps1 -SkipLaunch -Fast

# Инструменты разработки (линтеры, тесты)
pip install -e .[dev]
```

### Запуск

```bash
# Проверки качества кода + запуск GUI (рекомендуется)
pwsh -File run.ps1

# Запуск GUI напрямую (без проверок)
pwsh -File run.ps1 -FastLaunch
# или
python -m gui.main_window
```

При первом запуске приложение создаёт настройки по умолчанию и автоматически скачивает нужные модели. Если ML-зависимости не установлены, выполните `pip install -e .[ml]`.

## GUI

Приложение открывается с выбором режима (Text + Subtitles / Video QA) и запоминает последний выбор между сессиями.

### Text + Subtitles

- Выбор отдельного файла или целой папки для пакетной обработки.
- Включение/отключение диаризации и LLM-форматирования диалогов.
- Экспорт в TXT, SRT, VTT или JSON.
- Предпросмотр субтитров и прожиг в видео.
- Прогресс в реальном времени с возможностью отмены; результат в `transcriptions/`.

### Video QA

- Укажите локальный видеофайл (или YouTube-ссылку, экспериментально).
- Введите вопрос на естественном языке и опционально прикрепите контекстные файлы (txt, md, код, изображения).
- Просмотрите preflight-сводку (источник, число чанков, примерный бюджет токенов).
- Запустите анализ: AskVLM разбивает видео на чанки, извлекает кадры, выполняет ASR через WhisperX и отправляет каждый чанк в VLM.
- Получите обоснованный Markdown-ответ и лог доказательств.
- Поддержка LM Studio (локально) и OpenRouter (облако) в качестве LLM-бэкендов.

![Video QA GUI](doc/media/VideoQA%20GUI.png)

## CLI

AskVLM предоставляет три CLI-команды через [Typer](https://typer.tiangolo.com/):

### `transcribe` — пакетная транскрипция

```bash
python cli.py transcribe PATH -o output_dir --whisper-model large-v3 --export txt
```

Основные опции: `--whisper-model`, `--diarization/--no-diarization`, `--dialog-blocks`, `--export` (txt/srt/vtt/json), `--device` (auto/cuda/cpu), `--language`, `--engine` (whisper/whisperx/auto), `--recursive`, `--compute-type`.

### `subtitle` — генерация субтитров с прожигом

```bash
python cli.py subtitle PATH -o output_dir --burn-in --whisper-model large-v3
```

Генерирует SRT с настраиваемыми правилами читаемости (макс. CPS, длина строки, лимиты длительности кью) и опционально прожигает субтитры в видео через FFmpeg.

Основные опции: `--max-cps`, `--max-line-chars`, `--max-lines`, `--min-duration`, `--max-duration`, `--burn-in/--no-burn-in`, `--save-srt/--no-save-srt`, `--diarization`.

### `external-transcribe` — транскрипция для интеграций

```bash
python cli.py external-transcribe PATH_TO_MEDIA
```

Выводит текст транскрипции в stdout. Предназначен как машинно-читаемый endpoint для внешних приложений.

- Провайдер по умолчанию: `--stt-provider whisper` (модель `small`).
- Опционально: `--stt-provider gigaam-ctc` (CPU-only; входит в `.[ml]`; ~2.5 ГБ RAM, без VRAM).
- JIT-загрузка модели: Whisper/GigaAM загружается при старте транскрипции и выгружается перед выходом (в `--no-daemon`; в демоне — резидентно).
- Безопасность CUDA (Whisper): на Windows, если дочерний GPU-процесс падает (OOM), AskVLM автоматически повторяет на CPU в изолированном подпроцессе. GigaAM в CUDA/fallback/Windows GPU-isolation не участвует.
- Живой демон с другим `--stt-provider` → unavailable/mismatch (singleton не подменяется молча; нужен рестарт демона).
- Опциональный файловый вывод: `--output-file transcript.txt`.
- Диаризация отключена по умолчанию (экономия VRAM).

Подробности — [doc/EXTERNAL_CLI_TRANSCRIBER.md](doc/EXTERNAL_CLI_TRANSCRIBER.md).

### Переменные окружения

| Переменная | Назначение |
| --- | --- |
| `HF_TOKEN` | Токен Hugging Face для моделей диаризации PyAnnote (опционально) |
| `LLM_GGUF_PATH` | Путь к локальному GGUF-файлу для LLM-форматирования текста (опционально) |
| `OPENROUTER_API_KEY` | API-ключ OpenRouter для облачной VLM в режиме Video QA (опционально) |

## LLM-бэкенды

AskVLM использует LLM в двух независимых контекстах. Оба опциональны — приложение работает без настроенных LLM, но часть функций деградирует (форматирование не применяется) или становится недоступной (Video QA).

### Форматирование текста (dialog blocks)

Когда включён `--dialog-blocks` (CLI) или соответствующий тогл (GUI), сырой вывод ASR отправляется в локальную LLM для восстановления пунктуации, регистра и разбивки на абзацы. Форматтер использует **llama-cpp-python** с GGUF-моделью.

Настройка:

1. Установите ML-зависимости: `pip install -e .[ml]` (включает `llama-cpp-python`).
2. Задайте переменную окружения `LLM_GGUF_PATH`, указывающую на ваш `.gguf`-файл.
3. Если доступна CUDA GPU, форматтер автоматически выгружает слои на GPU; иначе работает на CPU.

Если путь к GGUF не задан или `llama-cpp-python` отсутствует, форматтер молча деградирует — транскрипция продолжается без форматирования.

### Video QA (мультимодальный анализ)

В режиме Video QA каждый чанк видео (репрезентативные кадры + фрагмент транскрипции) отправляется в Vision-Language Model для обоснованного анализа. Поддерживаются два бэкенда:

**LM Studio (локально)**

- Запустите [LM Studio](https://lmstudio.ai/) и загрузите VLM (например, вариант Qwen-VL или LLaVA).
- AskVLM подключается через OpenAI-совместимый endpoint `http://127.0.0.1:1234/v1`.
- Приложение управляет жизненным циклом моделей через LM Studio Developer REST API: может перечислять, загружать и выгружать инстансы моделей, чтобы делить один GPU между Whisper и VLM.

**OpenRouter (облако)**

- Задайте `OPENROUTER_API_KEY` в `.env` или как переменную окружения.
- Выберите мультимодальную модель в GUI (например, `qwen/qwen3.6-plus:free`).
- Поддержка параметра `reasoning` OpenRouter с настраиваемым уровнем усилий (`none`, `low`, `medium`, `high`).
- Подробности — [doc/OPENROUTER_INTEGRATION.md](doc/OPENROUTER_INTEGRATION.md).

Оба бэкенда используют единый промпт-контракт (`core/llm_prompts.py`): структурированный JSON-анализ по чанкам с последующим финальным синтезом, формирующим обоснованный ответ с доказательствами и маркерами неопределённости.

### Доктрина GPU-памяти

В VRAM одновременно находится только одна тяжёлая нейросеть. При переходе между стадиями пайплайна (например, Whisper → VLM) предыдущая модель выгружается в RAM или освобождается полностью перед загрузкой следующей. Это позволяет выполнять полный пайплайн Video QA на одном GPU с 8 ГБ.

## Архитектура

```
core/           Ядро обработки
  ffmpeg.py             Конвертация аудио/видео через FFmpeg
  whisper_wrapper.py    Бэкенд OpenAI Whisper
  whisperx_wrapper.py   Бэкенд WhisperX (пословные таймстемпы)
  gigaam_ctc_wrapper.py Опциональный CPU-only GigaAM Multilingual CTC
  stt_providers.py      Идентификаторы STT-провайдеров (whisper / gigaam-ctc)
  diarization.py        Диаризация говорящих через PyAnnote
  llm_formatter.py      LLM-форматирование текста
  pipelines.py          Оркестрация LocalPipeline
  gpu_guard.py          Проверка VRAM и защита от OOM
  settings.py           Настройки приложения
  lm_studio_rest.py     REST-клиент LM Studio
  video_qa_*.py         Пайплайн Video QA (чанкинг, кадры, оркестрация, манифест, политики)
gui/            Интерфейс на PySide6
  main_window.py        Главное окно с маршрутизацией режимов
  video_qa.py           Экран Video QA
  wysiwyg_editor.py     WYSIWYG-редактор транскрипции
  subtitle_preview.py   Виджет предпросмотра субтитров
  speaker_sidebar.py    Боковая панель управления говорящими
  preferences_dialog.py Диалог настроек
  export_dialog.py      Диалог экспорта
editing/        Текстовая модель и операции редактирования
utils/          Экспортеры, логирование, загрузчик моделей, хелперы
tools/          Утилиты бенчмаркинга (бенчмарки STT, поиск порога OOM)
tests/          Тестовый набор Pytest (юнит, интеграционные, E2E)
doc/            Проектная документация, гайды по интеграции
```

![Архитектура мультимодального GUI](doc/media/Multimodal%20GUI%20Design%2001%20-%20%D0%90%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%BD%D0%B0%D1%8F%20%D1%81%D1%85%D0%B5%D0%BC%D0%B0.png)

## Качество кода

Все проверки запускаются через `run.ps1` → `build.py`. Отдельная настройка pre-commit не нужна.

```bash
# Полный пайплайн: авто-фикс → линт → тайпчек → тесты → аудит безопасности
pwsh -File run.ps1 -SkipLaunch

# Быстрый режим (без медленных тестов)
pwsh -File run.ps1 -SkipLaunch -Fast

# Один инструмент
pwsh -File run.ps1 -Tool ruff

# Только запуск GUI
pwsh -File run.ps1 -FastLaunch
```

**Тулчейн**: Ruff (формат + линт), MyPy (strict), Pyright, Bandit (безопасность), Pytest (с покрытием), pip-audit.

```bash
# Шорткаты Makefile
make setup-dev     # Полная настройка dev-окружения
make check-all     # Запуск всех проверок
make format        # Авто-форматирование
make clean         # Очистка сгенерированных файлов
```

## Устранение неполадок

### «No module named 'faster_whisper'»

Установите ML-зависимости: `pip install -e .[ml]`

### «CUDA out of memory»

Приложение автоматически переключается на CPU. Также можно попробовать меньшую модель Whisper (`base` или `small` вместо `large-v3`) или закрыть другие GPU-приложения.

### CUDA не обнаружена (PyTorch без GPU)

PyTorch установлен без CUDA-колёс. Переустановите с явным суффиксом CUDA:

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

Или через билд-скрипт (CUDA чинится по умолчанию): `.\run.ps1 -SkipLaunch -Fast`

Проверка:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Подробности — [doc/CUDA_SETUP.md](doc/CUDA_SETUP.md).

### GUI не запускается

```bash
python -m gui.main_window
```

Проверьте вывод консоли на наличие ошибок.

### Советы по производительности

- Используйте NVIDIA GPU с 8 ГБ+ VRAM для лучшей производительности.
- Начните с модели Whisper `base` или `small`; переключайтесь на `large-v3`, если качество недостаточно.
- Обрабатывайте несколько файлов пакетом.
- Для Video QA локальный LM Studio-инстанс устраняет задержки облачного API.

## Документация

| Документ | Описание |
| --- | --- |
| [PYTHON_SETUP.md](PYTHON_SETUP.md) | Гайд по установке Python 3.11 |
| [doc/CUDA_SETUP.md](doc/CUDA_SETUP.md) | Настройка CUDA и PyTorch для GPU |
| [doc/EXTERNAL_CLI_TRANSCRIBER.md](doc/EXTERNAL_CLI_TRANSCRIBER.md) | Гайд по интеграции `external-transcribe` |
| [doc/AutoSubtitles.md](doc/AutoSubtitles.md) | Дизайн пайплайна субтитров |
| [doc/Multimodal GUI Design.md](doc/Multimodal%20GUI%20Design.md) | Архитектура мультимодального GUI |
| [doc/OPENROUTER_INTEGRATION.md](doc/OPENROUTER_INTEGRATION.md) | Справка по интеграции с OpenRouter |
| [doc/Disfluency-Cleanup-Design.md](doc/Disfluency-Cleanup-Design.md) | Дизайн очистки дисфлюенций |
| [TODO.md](TODO.md) | Роадмап и текущий прогресс |
