# TODO — AskVLM roadmap

Статус: базовая транскрипция, субтитры, preview, export и burn-in уже есть и считаются стабильной базой. Следующий этап: превратить приложение в мультимодальный desktop-инструмент, где пользователь подаёт видео, формулирует задачу и получает grounded-ответ от LLM без поломки subtitle-first workflow.

## Pipeline readiness

| Element | Status | Note |
| --- | --- | --- |
| GUI shell | ✅ ready | Mode routing and separate `Text + Subtitles` / `Video QA` surfaces exist. |
| Load media: local file | ✅ ready | Local file remains the stable base input. |
| Load media: network | 🟡 partial | YouTube URL import exists, but it is default-off and not the only missing release item. |
| Text mode | ✅ ready | Stable base workflow. |
| Subtitle mode | ✅ ready | Stable preview/export/burn-in workflow. |
| Whisper/WhisperX | ✅ ready for subtitle-first base, ✅ Video QA local run wired | Subtitle pipeline unchanged; Video QA uses ASR via `WhisperXWrapper` + `prepare_audio` without diarization/dialog formatting by default in the local run path. VRAM hand-off Whisper → LM phase not enforced yet (see §6). |
| GigaAM CTC | ✅ ready for `external-transcribe` | CPU-only `--stt-provider gigaam-ctc` (`ai-sage/GigaAM-Multilingual` revision `ctc`); deps in `.[ml]`; `run.ps1`/`build.ps1` default-ensure torch 2.10 CUDA; not used for CUDA / Whisper CUDA fallbacks / Windows GPU child isolation; daemon provider mismatch → unavailable. |
| Chunking | ✅ ready at planning level | Chunk planning exists in backend orchestration. |
| Representative frames | ✅ ready for local GUI run | `core/video_qa_local_run.VideoQAFFmpegFrameMaterializer` uses `extract_frame_to_file`. |
| LM Studio / VLM | 🟡 partial | GUI launch path runs chunk inferencer + LM Studio client; overflow vs live server behaviour still needs verification. |
| Transcript summary | ✅ ready for chunk prompts | `VideoQALMStudioChunkInferencer` builds per-chunk transcript excerpts; full QA path is wired from GUI. |
| Final answer | ✅ deterministic aggregate in local run | `VideoQADeterministicAnswerAggregator` builds `VideoQAAnswerBundle` without a second LM call; manifest + answer JSON save next to output. |
| Tests | 🟡 partial | CI passes, but manual regression and live overflow checks remain open. |

## Wave plan

- Wave 1: GUI shell + guardrails + minimal `LocalFile` provider. Subtitle-first `preview/export/burn-in` remains the stable base and is not mixed with `Video QA`.
- Wave 2: attachments/context + provider contract + URL import policy + early naming/legal slice.
- Wave 3: graph/manifest/schema/preparation only.
- Wave 4: real LLM passes + budget/model policy + LM Studio.
- Wave 5: outputs/tests/docs/release.
- Wave 1 stops before URL import, attachments/context, chunk planning, LLM orchestration, and budget/runtime policy in the backend; GUI surfacing of already-implemented backend capabilities is tracked separately in `## GUI tracking`.

## GUI tracking

**Активный workstream:** базовый `Video QA` surface в GUI (источник, вопрос, вложения, preflight, read-only ответ в Markdown и progress log на время прогона) развивается отдельно от стабильного `Text + Subtitles`. Полноценный запуск LLM из GUI (`Run Video QA`, worker в `MainWindow`, glue `core/video_qa_local_run.py`) уже есть; retry по чанкам из GUI и расширенный overflow-UX — следующие этапы (см. Wave 4+). Детализация сценариев по-прежнему в `## 2. MVP UX` и `## 10. Tests and verification`.

- [x] Ответ (Markdown) и progress log для `Video QA` (read-only; evidence после успешного run дописывается в log; контракт §5/§8).
- [x] Split layout: `Video QA` разделён на левую панель подготовки и правую для процесса/результатов с редактируемым бюджетом токенов.
- [x] Preflight summary перед запуском (структурированные поля и формы вместо текстового blob'а; стык с §6).
- [x] Attachments controls: список вложений расширяется по вертикали (resizable), include/exclude; переполнение budget отображается в тексте preflight.
- [ ] Stop-resume из GUI: backend и manifest уже поддерживают idempotent resume по `chunk_id` (§5), но отдельных кнопок в UI нет (ранний UX scaffold снят по тестам); добавить явные controls, когда будет решена связка с выбором чанка и output dir.
- [ ] Ручные regression checks по `Text + Subtitles`: preview, export, burn-in и переключение экранов не ломают subtitle-first path (дублирует дух §10, но как явный GUI-focused прогон).

## 1. Product guardrails

- [x] Разделить приложение на два рабочих экрана: текущий `Text + Subtitles` и отдельный `Video QA`.
- [x] Не ломать текущий subtitle preview / export / burn-in path при добавлении `Video QA`.
- [ ] Держать один главный сценарий UX: `video source -> task -> optional attachments -> grounded answer`.
- [x] Считать локальный файл базовым источником входа; URL-источники должны быть надстройкой, а не новым ядром pipeline.
- [ ] Считать это двумя сервисами в одном приложении: общий shell, но разные экраны, результаты и сценарии запуска.

## 2. MVP UX

- [x] Оставить текущий экран как workspace для `Text + Subtitles`, не смешивая его с `Video QA`.
- [x] Сделать отдельный экран `Video QA` со своим layout и своей зоной результата.
- [x] При старте приложения спрашивать, какой экран открыть, либо восстанавливать экран прошлой сессии.
- [x] Сохранять последний выбранный экран в settings и добавлять явный переключатель экрана внутри приложения.
- [x] Для `Video QA` добавить поле задания, отдельную область ответа (Markdown) и зону прогресса/лога пайплайна; evidence после run выводится в log (read-only).
- [x] Добавить секцию вложений к заданию: `txt`, `md`, кодовые файлы, `jpg`/`jpeg`, `png`, `webp` (через общий фильтр и нормализацию в `core/video_qa_context`).
- [x] Показать preflight перед запуском: источник, число чанков, примерный бюджет контекста, предупреждения (кнопка Refresh preflight).
- [x] Не смешивать subtitle editor и chat-like output; у каждого экрана должна быть своя зона результата.
- [ ] Автоматический повторный запуск по ошибочному чанку без повторной обработки всего видео (backend resume есть; GUI-кнопки и сценарий — позже, см. GUI tracking).

## 3. Input providers and source acquisition

- [x] Ввести контракт `input provider`: любой источник обязан резолвиться в локальный путь к медиа и метаданные.
- [x] Реализовать `LocalFile` как основной и самый надёжный provider для MVP.
- [x] Unit tests для `core/video_qa_policy.py` (default off, схемы/host, `file://`, описание temp policy).
- [x] Добавить опциональный URL import stage как отдельный provider, не вшивая загрузчик в orchestrator (`core/video_qa_url_import.py`: `VideoQAUrlImportProvider`).
- [x] Рассмотреть `yt-dlp`-класс инструментов как external optional dependency, а не как жёстко встроенную часть приложения (`VideoUrlDownloader` + `YtDlpCliDownloader`, без обязательной зависимости).
- [x] Поддержать YouTube URL как первый experimental provider после `LocalFile` (HTTP(S) хосты YouTube / `youtu.be`; остальные URL отклоняются на этом этапе).
- [ ] Отложить `VK Video` и `Rutube` в отдельный later/experimental слой после стабилизации YouTube path.
- [ ] Не планировать `Instagram`, `TikTok`, `X.com` в ближайший этап без устойчивого и юридически безопасного пути.
- [x] Зафиксировать политику временных файлов, кэша и очистки после URL import (`VideoQAUrlImportPolicy.temp_file_policy_description`, `UrlImportStagingHandle` / `cleanup_staging`).

## 4. Prompt context and attachments

- [x] Вынести вложения в отдельный слой `context providers`, чтобы не смешивать их с download pipeline видео.
- [x] Нормализовать текстовые вложения в единый внутренний формат с типом, именем файла и размером.
- [x] Для кодовых вложений сохранять язык/расширение, чтобы промпт мог корректно ссылаться на фрагменты.
- [x] Для изображений считать budget по эвристике с запасом, а не делать вид, что offline token count точен.
- [x] Добавить UI-флаги include/exclude для каждого вложения, если budget оказывается слишком большим (чекбоксы в таблице вложений; стратегия prompt — отдельно).
- [ ] Подготовить стратегию, как вложения попадают в prompt рядом с чанками видео и вопросом пользователя. Сейчас preflight при overflow просто блокирует запуск; отдельная future engineering task later — делить входной контекст на чанки и либо подавать чанки по очереди вместе с вопросом, либо суммаризировать чанки перед финальной агрегацией.

## 5. Video QA orchestration

- [x] Вынести orchestrator поверх текущего pipeline: `source resolve -> transcript reuse/build -> chunk plan -> representative frames -> LLM passes -> final aggregation` (`core/video_qa_executor.py`: `run_video_qa_executor`, инжектируемые transcript / frames / inference / aggregate).
- [x] Сохранять subtitle-first базу: транскрипт и субтитры остаются самостоятельным результатом, а не побочным артефактом QA (в planning-слое: `transcript_prepare` в `VIDEO_QA_SUBTITLE_FIRST_GRAPH_KINDS`, без слияния с QA chunk-планом).
- [x] Делать chunking по сценам/контенту с fallback на равномерную сетку по времени (`core/video_qa_orchestration.py`: `build_video_qa_chunk_plan`).
- [x] Зафиксировать политику `representative frame`: по умолчанию средний кадр сцены; альтернативы оставить как расширение (`VideoQARepresentativeFramePolicy`).
- [x] Явно описать overflow policy: сначала уменьшать число кадров, потом разрешение, потом дробить текст/чанк (`VideoQAOverflowPolicy`).
- [ ] Проверить фактическое поведение LM Studio при переполнении контекста: ошибка, partial output, `stopReason` или silent truncation.
- [ ] Строить budget control на своём preflight и fallback, а server-side overflow policy использовать только после отдельной верификации.
- [x] Ввести versioned JSON-manifest чанков: `schema_version`, `chunk_id`, `t_start`, `t_end`, кадры, артефакты, `status`, `attempts`, `error`.
- [x] Поддержать повторный запуск одного чанка и idempotent resume по manifest (`merge_planned_chunks_into_manifest` по `chunk_id`; executor пропускает `completed` и обновляет manifest без повторного inference).
- [x] Зафиксировать контракт финального ответа: `answer`, `evidence[]`, таймкоды, цитаты транскрипта, ссылки на кадры, признак неопределённости.

## 6. Budgeting and runtime scheduling

- [x] Для текста: абстракция счётчика токенов + консервативный fallback; точный tokenizer подключается при передаче счётчика в `build_video_qa_budget_estimate` (см. `TextTokenCounter`).
- [x] Для изображений использовать консервативную offline-эвристику с явным запасом (`VideoQAAttachment.budget_tokens` и attachment budget в `VideoQAContextBundle`).
- [x] Резервировать budget под финальный ответ и под служебные instructions, а не только под input (`VideoQABudgetPolicy` / `VideoQABudgetEstimate`).
- [x] Показать пользователю грубую оценку budget до старта и причину fallback/дробления при overflow (backend: `build_video_qa_preflight_report` / `format_video_qa_preflight_report_text` в `core/video_qa_orchestration.py`; отображение в GUI — см. §2).
- [x] Runtime scheduler (`VideoQARuntimeScheduler`): описана политика «одновременно не более одной тяжёлой нейросети» в `core/video_qa_runtime.py` (data-only: сводки для preflight/доков, **без** принудительной выгрузки моделей в executor).
- [ ] **Enforce: одна нейросеть в VRAM (процесс AskVLM) в каждый момент времени:** между `transcript_prepare` и `llm_pass` явно освобождать VRAM от локального Whisper (`WhisperXWrapper.unload`); сейчас Video QA / GUI path **не** вызывает `unload()` — модель остаётся загруженной на время HTTP к LM Studio. Учесть внешний VLM в LM Studio (уже в VRAM до старта сценария; приложение это не контролирует) — документация и при необходимости preflight-предупреждение или UX-hint.
- [x] Для конфигурации `8 GB VRAM / 64 GB RAM` зафиксирована политика `active -> offload to RAM -> unload` в `VideoQARuntimePolicy` / scheduler.
- [x] Сериализация model-heavy этапов и запрет параллельного inference по умолчанию (`allow_parallel_inference=False`, `serialize_model_heavy_steps=True`).
- [x] В `VideoQAModelProfile` задокументировано разделение model-dependent ограничений и эвристик приложения.

## 7. Model and LM Studio integration

- [x] Зафиксирован data-only профиль `VideoQAModelProfile` (`Qwen/Qwen3.5-35B-A3B`, LM Studio, multimodal, structured output best-effort); подробности по-прежнему в `doc/Qwen3.5-35B-A3B.md`.
- [x] Проверить связку `LM Studio + выбранная Qwen/VLM` на реальном мультимодальном запросе, а не только на тексте (локальный opt-in probe на `localhost:1234`).
- [x] Проверить, насколько локальный сервер поддерживает structured output / JSON contract, и предусмотреть graceful fallback (verified with the same local probe).
- [x] Добавить в репозиторий краткий internal reference по LM Studio: OpenAI-compatible API, multimodal payload, streaming, caveats structured output.

## 8. Outputs and artifacts

- [x] Оставить текущие `TXT`/`SRT`/`VTT`/`JSON` экспортёры как базовый путь.
- [x] Для `Video QA` добавить machine-readable export ответа и evidence-списка.
- [x] Сохранять итоговый answer bundle рядом с manifest, чтобы можно было разбирать run post factum.
- [x] Для ответа по видео дать формат с цитатами, таймкодами и ссылками на кадры.
- [ ] **Персистентный журнал / полный артефакт прогона:** на диск пишутся `{run_id}.manifest.json` и answer bundle; **во время прогона** GUI показывает in-session progress log (стадии пайплайна, `pipeline_log` в `run_local_video_qa`). Транскрипт ASR целиком в manifest не хранится, сырой ответ LM почти не логируется (кроме best-effort `logging` при сбое парсинга). Нет единого **файла** «run log» (стадии, тайминги, ошибки, опционально redacted payloads). Решить: что сохранять рядом с output dir (transcript.json / `.log`), уровень детализации и опциональный verbose-режим.

## 9. Naming, legal and release prep

- [x] Зафиксировать `AskVLM` как canonical product name в GUI, CLI, build/help-текстах, metadata и документации.
- [x] Найти и обновить legacy-упоминания старого бренда приложения и старые абсолютные пути в коде и документации.
- [x] Перевести `QSettings`, session keys и exporter metadata на `AskVLM` без legacy compatibility branches и fallback-парсинга.
- [ ] Подготовить MIT readiness checklist: лицензии зависимостей, бинарей, model weights и вспомогательных инструментов.
- [ ] Для URL import (backend `core/video_qa_url_import.py` уже есть; по умолчанию выключен в policy) перед релизом / enable-by-default проверить не только лицензии, но и ToS/redistributability для конкретных source adapters.
- [ ] Решить, какие external tools поставляются вместе с приложением, а какие пользователь устанавливает отдельно.
- [ ] Добавить user-facing дисклеймер про ответственность за источник контента при URL import.

## 10. Tests and verification

- [ ] Проверить стартовый выбор экрана, восстановление экрана прошлой сессии и ручное переключение между экранами.
- [ ] Проверить, что `Text` и `Subtitles` пути не деградируют после добавления `Video QA`.
- [ ] Проверить `input providers`: локальный файл, ошибки URL resolve, временные файлы, cleanup.
- [ ] Проверить сбор контекста из вложений и их budget trimming.
- [ ] Проверить chunk planning, manifest persistence и resume по `chunk_id` (частично: `tests/test_video_qa_executor.py` — executor, resume, run-level `failed` и сводка при ошибке чанка, смешанные исходы; полный E2E с GUI/диском — позже).
- [ ] Проверить overflow fallback order, реакцию на server-side context limit и объяснимость ошибок пользователю.
- [ ] Проверить scheduler и правило `one active model at a time`.
- [ ] Проверить формат ответа с таймкодами, цитатами и привязкой к кадрам.
- [ ] Прогонять локальную проверку через `./run.ps1 -SkipLaunch` перед заявлением о готовности.

## 11. Explicitly not now

- Не тянуть тяжёлые research-репозитории long-video / Video QA в ближайший этап.
- Не делать `LangChain` обязательным ядром orchestrator без отдельной необходимости.
- Не обещать точный offline token count для изображений, пока нет model-specific расчёта.

## 12. Расширение (post-MVP)

- **GigaAM CTC (сделано для external CLI):** опциональный CPU-only `--stt-provider gigaam-ctc` в `external-transcribe` / daemon / queue; Whisper остаётся default. Возможные follow-up: GUI picker, native longform вместо внешнего чанкинга, отдельный A/B quality gate на живых RU-записях.
- Блок задач по **загрузке видео из сети** (не только YouTube): унифицировать URL providers, политики temp/cleanup, UX импорта и ToS/лицензии в одном workstream, чтобы не размазывать сетевой ввод по разным §.
- **Чат с финальным ответом Video QA**, а не только read-only просмотр: возможность уточнять ответ в диалоге (контекст: evidence, manifest, вопрос), с отдельным контрактом от subtitle-first. Опциональное подключение Filesystem MCP.
- **Другие локальные LLM-провайдеры** помимо LM Studio: абстракция над OpenAI-compatible HTTP (base URL, auth, capabilities), плюс опциональные адаптеры (ollama, llama.cpp server, …) без смешения с subtitle pipeline.
- **LLM-via-API** (облачные ключи, rate limits, redaction): отдельный provider слой для Video QA / Text с явной политикой данных и opt-in.
