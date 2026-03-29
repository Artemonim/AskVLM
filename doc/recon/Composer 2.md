# Техразведка: мультимодальный GUI и Video QA (OSS)

Дата: 2026-03-29. Контекст: `TODO.md` (режимы Text / Subtitles / Video QA, оркестратор, эвристики кадров, preflight бюджета, chunk fallback, JSON-manifest, форматы ответа с таймкодами, тесты).

---

## Краткое резюме

Для **сегментации видео и «осмысленных» границ чанков** в экосистеме Python хорошо закрыты задачи **shot/scene detection** (PySceneDetect, фильтры FFmpeg, опционально TransNet V2). Для **равномерного или индексного сэмплирования кадров** — декодеры вроде **Decord** и **TorchCodec** (сторона PyTorch). Для **структурированного ответа** (цитаты, таймкоды) — паттерн **Pydantic + Instructor** или **JSON Schema / structured outputs** у облачных провайдеров; для **локального LM Studio** гарантии схемы слабее, это нужно явно заложить в TODO. **Готового одного OSS-пакета**, который одновременно даёт оркестратор «транскрипт → чанки → кадры → VLM → агрегация» под десктопное приложение, **в выдаче не обнаружено** — типичная картина: compose из FFmpeg, своих манифестов и HTTP-клиента к OpenAI-совместимому API. **Исследовательские** репозитории (LongVILA, LWM, Chapter-Llama, TimeLens и т.п.) полезны как **идеи и бенчмарки**, а не как прямое встраивание без сильной переработки.

---

## Найденные кандидаты и идеи

### 1. Режимы GUI, разделение Text / Subtitles / Video QA

| Кандидат | Суть | Примечание |
|----------|------|------------|
| Паттерн **single window + explicit mode state machine** | Один главный виджет, переключатель режима, разные панели результата | Это архитектурный паттерн, не библиотека; соответствует текущей доктрине «не смешивать editor и chat-like output». |
| **pytest-qt** | Pytest-плагин для PyQt5/6 и PySide: `qtbot`, клики, сигналы, headless CI | Репозиторий: [pytest-dev/pytest-qt](https://github.com/pytest-dev/pytest-qt), документация: [pytest-qt.readthedocs.io](https://pytest-qt.readthedocs.io/en/latest/tutorial.html). Подходит под пункт TODO «проверить router режимов GUI». |

**LangChain:** есть **VideoContentBlock** и развитие мультимодальных сообщений ([документация LangChain](https://reference.langchain.com/python/langchain-core/messages/content/VideoContentBlock)), для YouTube — **YoutubeLoader** с чанками по времени ([документация YouTube transcripts](https://docs.langchain.com/oss/python/integrations/document_loaders/youtube_transcript)). Для **локального файла** готового «загрузчика видео → чанки → кадры» как единого решения в выдаче **не видно**; тянуть LangChain ради режимов GUI **не обязательно**, если оркестрация уже в `core/`/`gui/`.

### 2. Representative frames и chunking (не «только keyframes кодека»)

| Кандидат | Внедрение | Ссылки |
|----------|-----------|--------|
| **PySceneDetect** (`scenedetect`) | **Почти сразу:** Python API и CLI, детекторы `AdaptiveDetector`, `ContentDetector`, разрез по сценам + опционально split через ffmpeg. Лицензия BSD-3. | [GitHub: Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect), [scenedetect.com](https://www.scenedetect.com/) |
| **FFmpeg `select` + `scene` / `scdet`** | **Почти сразу**, если FFmpeg уже в pipeline: метаданные смен сцен без новой Python-зависимости; пороги подбираются под контент. | Обсуждения и фильтры: [документация scdet (пример зеркала)](https://ayosec.github.io/ffmpeg-filters-docs/6.0/Filters/Video/scdet.html), [select filter](https://ayosec.github.io/ffmpeg-filters-docs/1.0/Filters/Video/select.html) |
| **TransNet V2** | **Будущая опция:** нейросеть для границ склеек, выше качество на сложном монтаже, но TF/PyTorch, веса, latency. | [GitHub: soCzech/TransNetV2](https://github.com/soCzech/TransNetV2), [PyPI: transnetv2-pytorch](https://pypi.org/project/transnetv2-pytorch/) |
| **Decord** | Удобное **случайное/индексное** чтение кадров для батчей (типичный DL-стек). | [GitHub: dmlc/decord](https://github.com/dmlc/decord) |
| **TorchCodec** | Равномерные клипы по индексам/таймстампам в экосистеме PyTorch (если когда-то понадобится унификация с torch). | [Блог PyTorch: TorchCodec](https://pytorch.org/blog/torchcodec/), [документация samplers](https://meta-pytorch.org/torchcodec/stable/api_ref_samplers.html) |

**Исследовательский ориентир (не drop-in):** **Chapter-Llama** — часовое видео, отбор кадров с опорой на речь ([arXiv:2504.00072](https://arxiv.org/html/2504.00072v1)). Идея **согласовать сегменты с транскриптом** совпадает с вашим subtitle-first путём, но это **метод из статьи**, не готовый модуль под продукт без адаптации.

### 3. Оркестратор, manifest чанков, fallback при overflow

Здесь **стандарта «один JSON-формат для всех» нет**; близкий практический подход:

- **Свой JSON-manifest** (версия схемы, `chunk_id`, `t_start`/`t_end`, пути к кадрам, хэши/размеры, статус инференса, ошибки) — позволяет **перезапускать один чанк** (пункт TODO).
- **NDJSON / JSON Lines** для потоковых логов шага — опционально, если нужен append-only trace.

Паттерн **map → reduce** по чанкам (локальный VLM на чанк, затем агрегирующий вызов) описан во многих long-video работах, но **как единая готовая библиотека под LM Studio** в поиске **не найден**.

### 4. Preflight бюджета: токены текста + изображения + ответ

| Инструмент | Комментарий |
|------------|-------------|
| **tiktoken** | Хорош для **текста** под совместимые с OpenAI токенизаторы; **картинки** так не посчитать точно. |
| **OpenAI: подсчёт input tokens** (API) | Точный preflight для мультимодальных входов в облаке — см. [руководство по подсчёту токенов](https://developers.openai.com/api/docs/guides/token-counting/). Для **чисто локального** LM Studio это **не переносится один в один**. |
| **Эвристики по размеру/тайлам** | Многие VLM считают визуальные токены от **разрешения и тайлинга** модели; без спецификации конкретной VLM preflight остаётся **оценкой с запасом** (что уже отражено в TODO: «запас под изображения»). |

**Вывод:** пункт TODO корректен; стоит **уточнить в спецификации**, для каких моделей/провайдеров нужна точность (облако vs локально) и документировать **консервативный множитель** для картинок.

### 5. Формат ответа: цитаты, таймкоды, ссылки на кадры

| Подход | Когда уместно |
|--------|----------------|
| **Pydantic-модель ответа + парсинг JSON** из ответа модели | Универсально; ручная валидация и retry при ошибке схемы. |
| **Instructor** ([GitHub: 567-labs/instructor](https://github.com/567-labs/instructor), [python.useinstructor.com](https://python.useinstructor.com/learning)) | Удобно, если один клиент и стабильный structured path; **проверить** поддержку вашей связки **LM Studio + конкретная VLM**. |
| **OpenAI Structured Outputs / json_schema** ([документация](https://developers.openai.com/docs/guides/structured-outputs), [cookbook](https://cookbook.openai.com/examples/structured_outputs_intro)) | Сильная гарантия схемы **для поддерживаемых облачных моделей**; для локального сервера может **отсутствовать** или вести себя иначе. |

**Исследовательский слой «как учить VLM давать таймкоды»:** работы по **temporal video grounding** (примеры: [TimeLens](https://timelens-arc-lab.github.io/), [TAR-TVG на arXiv](https://arxiv.org/abs/2508.07683)) — это **данные/обучение/оценка**, не замена продуктовому контракту JSON в AskVLM.

### 6. Video QA: бенчмарки и фреймворки (скорее оценка, чем продукт)

- **OVQA, OpenVQA, Just Ask** и др. — датасеты/эвал для Video QA ([пример: OVQA](https://github.com/mlvlab/OVQA), [OpenVQA](https://github.com/MILVLG/openvqa)). Полезны, если понадобится **регрессионный набор** или сравнение качества ответов, **не** как runtime-оркестратор.

### 7. Локальный инференс и совместимость с текущим стеком (LM Studio)

- **OpenAI-compatible API** и **vision / изображения**: [LM Studio — OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat), [Image input (Python)](https://lmstudio.ai/docs/python/llm-prediction/image-input). Это **прямая опора** для HTTP-клиента в оркестраторе без смены протокола.

### 8. Крупные OSS-модели «длинного видео» (фон, не быстрый merge)

- **LongVILA**, **Eagle**, **LWM** — масштаб контекста и инфраструктура GPU; для десктопного приложения с внешним VLM через API это **скорее ориентир по тому, как делают long-context**, а не библиотека «подключил и работает». Примеры: [LongVILA (paper page)](https://huggingface.co/papers/2408.10188), [NVlabs/Eagle](https://github.com/nvlabs/eagle), [LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM).

---

## Рекомендации по приоритету

**Ближе к «сразу» (низкий риск, высокая связность с TODO):**

1. Зафиксировать **контракт JSON-manifest чанков** (поля для ретрая, таймкоды, пути к артефактам) — без внешней зависимости.
2. Добавить **сценовую/контентную** эвристику границ: **PySceneDetect** и/или **FFmpeg scene/scdet**, с fallback на равномерную сетку по времени при «плоском» видео.
3. Для **тестов GUI режимов** — **pytest-qt** и сценарии переключения + изоляция виджетов ответа.
4. Для **формата ответа** — начать с **явной JSON-схемы** в промпте + Pydantic-валидация; при необходимости — **Instructor**, после проверки на целевой локальной модели.

**Отложить / опционально:**

- **TransNet V2** — если PySceneDetect/FFmpeg дают много ложных/пропущенных склеек на целевом контенте.
- **Decord/TorchCodec** — если текущий способ извлечения кадров узкое место по CPU/IO.
- **LangChain** — только если явно нужен экосистемный слой цепочек; иначе дублирует оркестратор.
- **Тяжёлые video-LLM репозитории** (LWM, LongVILA) — если только не планируется свой инференс на GPU внутри продукта.

---

## Риски, ограничения и что уточнить в TODO

| Риск / пробел | Детали |
|---------------|--------|
| **Structured output локально** | LM Studio и разные VLM могут не поддерживать жёсткий `json_schema` как в OpenAI; нужен **graceful fallback** (повтор с исправлением, упрощённая схема). |
| **Точный preflight картинок offline** | Без серверного token API остаётся **оценка**; в TODO стоит явно разделить «бюджет облако» и «бюджет локально». |
| **Сцены ≠ семантические главы** | Scene detection даёт **монтажные** границы; для вопросов «о смысле» может понадобиться **доп. объединение** чанков по транскрипту. |
| **Дублирование FFmpeg** | PySceneDetect часто вызывает ffmpeg; важно не размножить конфликтующие версии путей на Windows. |
| **Лицензии** | PySceneDetect, TransNet и сходные утилиты **часто** оказываются в пермиссивном OSS-сегменте, но в этом отчёте **файлы `LICENSE` по первоисточнику не сверялись** — перед интеграцией нужно прочитать лицензию в каждом конкретном репозитории. Для **весов, чекпойнтов и крупных model repos** условия использования и распространения проверяются **отдельно**. |

**Что уточнить в `TODO.md` (предложение формулировок, не правка файла в этой задаче):**

- Определение **«representative frame»**: середина сцены, кадр с макс. движением, первый после склейки, N кадров на чанк.
- Политика **chunk fallback**: уменьшение числа кадров, уменьшение разрешения, объединение соседних чанков для текста, повтор с другой эвристикой.
- Версия **manifest** (`schema_version`) и обязательные поля для **идемпотентного** рестарта.

---

## Что не удалось найти (без выдумок)

- **Единая OSS-библиотека** уровня «вставь в PyQt-приложение: полный Video QA pipeline с транскриптом, манифестом чанков и агрегацией под OpenAI-compatible localhost».
- **Стандартизированный отраслевой формат** JSON-manifest для VLM-чанков (де-факто каждый проект задаёт свой).
- **Надёжный офлайн-калькулятор** визуальных токенов, универсальный для всех VLM без привязки к конкретной модели/серверу (кроме грубых эвристик и облачных API).

---

## Что делать дальше

1. Прототип **манифеста** и одного **end-to-end** прохода: 2–3 чанка, один ретрай по `chunk_id`.
2. Сравнить на реальных роликах **FFmpeg scene** vs **PySceneDetect** по числу чанков и стабильности границ.
3. Зафиксировать **JSON-схему ответа QA** (поля: `answer`, `evidence[]` с `t0`/`t1`, `transcript_excerpt`, `frame_id` или путь) и проверить **на целевой VLM через LM Studio**.
4. Добавить **pytest-qt** тест: смена режима Text / Subtitles / Video QA не ломает существующие виджеты субтитров.
5. При необходимости облачного режима — оценить **официальный token counting** провайдера для точного preflight.

---

## Источники (сводно)

- PySceneDetect: [github.com/Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect), [www.scenedetect.com](https://www.scenedetect.com/)
- FFmpeg filters: [scdet](https://ayosec.github.io/ffmpeg-filters-docs/6.0/Filters/Video/scdet.html), [select](https://ayosec.github.io/ffmpeg-filters-docs/1.0/Filters/Video/select.html)
- TransNet V2: [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)
- Decord: [github.com/dmlc/decord](https://github.com/dmlc/decord)
- TorchCodec: [pytorch.org/blog/torchcodec](https://pytorch.org/blog/torchcodec/)
- pytest-qt: [github.com/pytest-dev/pytest-qt](https://github.com/pytest-dev/pytest-qt)
- Instructor: [github.com/567-labs/instructor](https://github.com/567-labs/instructor)
- OpenAI structured outputs: [developers.openai.com/docs/guides/structured-outputs](https://developers.openai.com/docs/guides/structured-outputs)
- OpenAI token counting: [developers.openai.com/api/docs/guides/token-counting](https://developers.openai.com/api/docs/guides/token-counting/)
- LM Studio OpenAI compat / images: [lmstudio.ai/docs/developer/openai-compat](https://lmstudio.ai/docs/developer/openai-compat), [lmstudio.ai/docs/python/llm-prediction/image-input](https://lmstudio.ai/docs/python/llm-prediction/image-input)
- LangChain VideoContentBlock: [reference.langchain.com](https://reference.langchain.com/python/langchain-core/messages/content/VideoContentBlock)
- Chapter-Llama (идея chaptering): [arxiv.org/html/2504.00072v1](https://arxiv.org/html/2504.00072v1)
