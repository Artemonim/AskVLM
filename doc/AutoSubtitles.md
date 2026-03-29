### AskVLM — Автосубтитры: производственный пайплайн и интеграция с DaVinci Resolve

Дата: 2025‑10‑19

---

### Кратко

- **Цель**: надёжная автосубтитровка готового видео с упором на качество таймингов и производительность на Windows + CUDA 8 GB.
- **Движок**: WhisperX (forced alignment + word timestamps + опциональная диаризация) как основной; fallback на faster‑whisper‑only при ограничениях.
- **Артефакты по умолчанию**: всегда «вжигать» субтитры в видео + всегда сохранять sidecar `.srt` для QA/переэкспорта.
- **Интеграция с Resolve**: базовый флоу через импорт `.srt` (без плагина на первом этапе). Плагин/мост — позже.

---

### Аппаратные допущения (профиль по умолчанию)

- ОС: Windows
- GPU: CUDA, 8 GB VRAM
- Диски: NVMe для ускорения загрузки весов (полезно для OS file cache)
- Параллелизм: один «тяжёлый» GPU‑этап в единицу времени; CPU‑работы можно параллелить

---

### Архитектурные принципы

1) **WhisperX как основной слой синхронизации**: улучшение таймингов (align), word‑level timestamps; диаризация (pyannote) опциональна.
2) **Fallback‑стратегия**: если align/diar недоступны (VRAM/среда) — выполняем ASR (faster‑whisper) и продолжаем пайплайн.
3) **RAM↔VRAM оркестрация**: хранить PyTorch‑модели (Align/Diari) «горячими» в RAM; поднимать в VRAM строго по одному на время этапа; после этапа выгружать.
4) **Предзагрузка на простое**: ленивая CPU‑загрузка Align/Diari и «прогрев» OS file cache для ASR до старта.
5) **Чанкование**: длинные видео делить (VAD/хронология) для стабильной памяти и предсказуемой длительности вычислений.

---

### Оркестрация памяти (RAM↔VRAM) и производительность

- **ASR (faster‑whisper / CTranslate2)**:
  - На 8 GB: `compute_type=int8_float16` (или `float16`, если стабильно), `batch_size=8–16`.
  - Держать GPU‑инстанс только на время распознавания. Повторные загрузки ускоряются за счёт OS page cache.
- **Align (WhisperX align)**:
  - Держим в RAM, переносим в CUDA только на время шага; небольшой `batch_size`.
- **Diarization (pyannote)**:
  - По флагу; после align. При дефиците VRAM — fallback на CPU для диаризации.
- **Жёсткая последовательность**: одновременно в VRAM только одна тяжёлая модель. Между этапами `del model; torch.cuda.empty_cache()`.
- **Оценка VRAM**: перед переносом в CUDA — проверка свободной памяти; при нехватке — уменьшение `batch_size`/CPU‑fallback.

---

### Экспорт и прожиг субтитров

- Всегда сохраняем `.srt` (UTF‑8). По умолчанию также «вжариваем» в видео через `ffmpeg` (`subtitles=`/`libass`).
- Для сложного стиля — генерировать `.ass` и прожигать через `-vf subtitles=...`.

Примеры (для справки):

```bash
# Вжечь SRT (простая стилизация — убедиться, что ffmpeg собран с libass)
ffmpeg -i input.mp4 -vf "subtitles='subs.srt':force_style='Fontsize=42,Outline=2,Shadow=0'" -c:v libx264 -crf 18 -preset veryfast -c:a copy output.mp4

# Вжечь ASS (более точный контроль стиля)
ffmpeg -i input.mp4 -vf "subtitles='subs.ass'" -c:v libx264 -crf 18 -preset veryfast -c:a copy output.mp4
```

---

### Форматирование субтитров (правила читабельности)

- 1–2 строки, до ~42 символов в строке
- Длительность: ~1.2–6.0 секунд (короткие — сливать, слишком длинные — делить)
- Скорость чтения (CPS): целевой максимум ~17–18
- Разбиение по словам/знакам препинания с использованием word‑timestamps от WhisperX

Экспортируем SRT/WebVTT; опционально `.ass` для стилевого контроля.

---

### Интеграция с DaVinci Resolve (рекомендуемый флоу)

1) В Resolve собрать монтаж и выставить финальный таймлайн.
2) Экспортировать «мастер» (или быстрее: `audio‑only WAV 16 kHz`).
3) В AskVLM создать `.srt` (и при желании предварительный «вжаренный» файл).
4) Импортировать `.srt` в Resolve: File → Import → Subtitle → настроить Track Style в Inspector.
5) При необходимости подправить текст/тайминг в Resolve, затем финальный экспорт:
   - Burn into video — «вжечь» в рендере Resolve
   - Export Subtitle — выгрузить sidecar
6) После — при необходимости прогнать финал через видеоконвертер под целевые платформы.

Примечание: мост/плагин Resolve возможен позже; быстрый путь сейчас — «audio‑only экспорт → SRT → импорт» либо watch‑folder режим в AskVLM.

---

### CLI и GUI (минимальные требования)

- **CLI (`subtitle`)**: batch по файлам/папкам; ключи: `--burn-in` (по умолчанию true), `--save-srt` (always true), `--diarize`, `--device`, `--compute-type`, `--batch-size`, `--vad`, `--max-cps`, `--max-line-chars`, `--max-lines`, `--min-duration`, `--max-duration`, `--format srt|vtt|ass`.
- **GUI**: чекбокс «Вжечь субтитры», «Также сохранить .srt», «Диаризация», выбор стиля/формата.

---

### .env (плейсхолдеры — заменить своими значениями)

```
# Базовые пути/кеши
SK_MODELS_DIR=PATH_TO_MODELS
SK_CACHE_DIR=PATH_TO_CACHE
HF_HOME=PATH_TO_HF_CACHE
TRANSFORMERS_CACHE=PATH_TO_HF_CACHE
TORCH_HOME=PATH_TO_TORCH_CACHE
FFMPEG_PATH=PATH_TO_FFMPEG_BIN

# Производительность/девайс
CUDA_VISIBLE_DEVICES=0
SK_DEVICE=auto                 # auto|cuda|cpu
SK_COMPUTE_TYPE=int8_float16   # 8 GB: int8_float16 или float16
SK_BATCH_SIZE=8
SK_VAD=true
SK_DIARIZE=false               # включайте точечно
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:64
TOKENIZERS_PARALLELISM=false

# Prefetch/оркестрация
SK_PREFETCH_ON_IDLE=true       # ленивая предзагрузка в RAM
SK_RAM_RESIDENCY=true          # держать Align/Diari в CPU‑RAM

# Секреты/клиенты (по необходимости)
HF_TOKEN=YOUR_HF_TOKEN         # для pyannote (диаризация)
OPENAI_API_KEY=YOUR_OPENAI_KEY # если используете облачную LLM/ASR
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
YANDEX_SPEECHKIT_OAUTH=...
YANDEX_SPEECHKIT_FOLDER_ID=...
HTTPS_PROXY=...
HTTP_PROXY=...

# Поведение/логи
SK_OUTPUT_DIR=PATH_TO_OUTPUT
SK_LOG_LEVEL=INFO
SK_BURN_SUBS_DEFAULT=true
SK_SAVE_SRT_ALWAYS=true
```

Важно: `.env` и файлы с ключами не коммитить; добавить в `.gitignore`.

---

### SaaS‑аспекты (кратко)

- **RAM‑резиденция** Align/Diari и строгая последовательная миграция в VRAM экономичнее, чем постоянная «жизнь» в большой VRAM, при 1–2 одновременных задачах.
- **ASR (CT2)** полагается на NVMe+OS cache — RAM‑резиденция даёт меньший выигрыш, чем у PyTorch‑моделей.
- **LLM пост‑коррекция**: на 8 GB конкурирует с ASR; лучше выносить в отдельный сервис (vLLM) или в провайдерский API.
- **Масштабирование**: очередь задач; один GPU‑воркер — один тяжёлый шаг; CPU‑этапы можно распараллелить; кэширование весов на NVMe.

---

### Контроль качества (QA)

- Сэмплинг 0–10 минут для проверки текста/таймингов
- Валидация CPS (≤ ~18), длительностей (1.2–6.0 с), переносов
- Проверка имён/цифр/терминов; при многоголосии — метки спикеров
- Версионирование: `video_v1.srt`, `video_v2_resynced.srt`

---

### Дорожная карта (первый инкремент)

1) CLI `subtitle` (batch) и пайплайн: ASR → Align → (Diar opt) → форматирование → `.srt` → burn‑in.
2) `gpu_guard`/`ModelRegistry`: строго последовательные GPU‑этапы, предзагрузка Align/Diari в RAM.
3) Экспортёры: SRT/WebVTT/ASS с правилами (CPS/длительности/переносы).
4) FFmpeg‑обёртка: `burn_subtitles(video, srt_or_ass, style, out)`.
5) GUI минимум: «Вжечь субтитры», «Сохранить .srt», «Диаризация», формат.
6) Тесты: интеграция WAV → JSON → SRT; короткое видео → SRT → burn‑in; валидация CPS/формата.

---

### Открытые вопросы / отложенные решения

- Плагин/мост для DaVinci Resolve — этап 2 (после стабилизации пайплайна).
- Выделение `littletools_video` в отдельный проект — после фиксации UX/требований к кодекам/профилям.


