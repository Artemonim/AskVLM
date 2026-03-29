# TODO — AskVLM roadmap

Статус: базовая транскрипция, субтитры, preview, export и burn-in уже есть и считаются стабильной базой. Следующий этап: превратить приложение в мультимодальный desktop-инструмент, где пользователь подаёт видео, формулирует задачу и получает grounded-ответ от LLM без поломки subtitle-first workflow.

## Wave plan

- Wave 1: GUI shell + guardrails + minimal `LocalFile` provider. Subtitle-first `preview/export/burn-in` remains the stable base and is not mixed with `Video QA`.
- Wave 2: attachments/context + provider contract + URL import policy + early naming/legal slice.
- Wave 3: graph/manifest/schema/preparation only.
- Wave 4: real LLM passes + budget/model policy + LM Studio.
- Wave 5: outputs/tests/docs/release.
- Wave 1 stops before URL import, attachments/context, chunk planning, LLM orchestration, and budget/runtime policy.

## GUI tracking

GUI остаётся на уровне shell/stub: маршрутизация режимов и минимальные guardrails без полноценного Video QA surface. Более сложные controls и полная UI-поверхность для QA (ответ, evidence, preflight, вложения, retry) откладываются до готовности backend по **Wave 4** (реальные LLM passes, budget/model policy). Детализация сценариев по-прежнему в `## 2. MVP UX` и `## 10. Tests and verification`; этот блок — отдельный чеклист по GUI-работам, без реализации новых элементов «прямо сейчас».

- [ ] Ответ и evidence surface для `Video QA` (зона ответа, список evidence-блоков, согласованность с финальным контрактом из §5/§8).
- [ ] Preflight summary перед запуском (источник, чанки, грубый budget, предупреждения; стык с §6).
- [ ] Attachments controls: список вложений, include/exclude, UX при переполнении budget (связка с §4).
- [ ] Retry controls: повтор по чанку / resume без полной переобработки видео (связка с manifest/orchestration в §5).
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
- [ ] При старте приложения спрашивать, какой экран открыть, либо восстанавливать экран прошлой сессии.
- [x] Сохранять последний выбранный экран в settings и добавлять явный переключатель экрана внутри приложения.
- [ ] Для `Video QA` добавить поле задания, отдельную область ответа и список evidence-блоков.
- [ ] Добавить секцию вложений к заданию: `txt`, `md`, кодовые файлы, `jpg`/`jpeg`, `png`, `webp`.
- [ ] Показать preflight перед запуском: источник, число чанков, примерный бюджет контекста, предупреждения.
- [ ] Не смешивать subtitle editor и chat-like output; у каждого экрана должна быть своя зона результата.
- [ ] Подготовить UX для повторного запуска по ошибочному чанку без повторной обработки всего видео.

## 3. Input providers and source acquisition

- [x] Ввести контракт `input provider`: любой источник обязан резолвиться в локальный путь к медиа и метаданные.
- [x] Реализовать `LocalFile` как основной и самый надёжный provider для MVP.
- [x] Unit tests для `core/video_qa_policy.py` (default off, схемы/host, `file://`, описание temp policy).
- [ ] Добавить опциональный URL import stage как отдельный provider, не вшивая загрузчик в orchestrator.
- [ ] Рассмотреть `yt-dlp`-класс инструментов как external optional dependency, а не как жёстко встроенную часть приложения.
- [ ] Поддержать YouTube URL как первый experimental provider после `LocalFile`.
- [ ] Отложить `VK Video` и `Rutube` в отдельный later/experimental слой после стабилизации YouTube path.
- [ ] Не планировать `Instagram`, `TikTok`, `X.com` в ближайший этап без устойчивого и юридически безопасного пути.
- [ ] Зафиксировать политику временных файлов, кэша и очистки после URL import.

## 4. Prompt context and attachments

- [x] Вынести вложения в отдельный слой `context providers`, чтобы не смешивать их с download pipeline видео.
- [x] Нормализовать текстовые вложения в единый внутренний формат с типом, именем файла и размером.
- [x] Для кодовых вложений сохранять язык/расширение, чтобы промпт мог корректно ссылаться на фрагменты.
- [x] Для изображений считать budget по эвристике с запасом, а не делать вид, что offline token count точен.
- [ ] Добавить UI-флаги include/exclude для каждого вложения, если budget оказывается слишком большим.
- [ ] Подготовить стратегию, как вложения попадают в prompt рядом с чанками видео и вопросом пользователя.

## 5. Video QA orchestration

- [ ] Вынести orchestrator поверх текущего pipeline: `source resolve -> transcript reuse/build -> chunk plan -> representative frames -> LLM passes -> final aggregation`.
- [ ] Сохранять subtitle-first базу: транскрипт и субтитры остаются самостоятельным результатом, а не побочным артефактом QA.
- [ ] Делать chunking по сценам/контенту с fallback на равномерную сетку по времени.
- [ ] Зафиксировать политику `representative frame`: по умолчанию средний кадр сцены; альтернативы оставить как расширение.
- [ ] Явно описать overflow policy: сначала уменьшать число кадров, потом разрешение, потом дробить текст/чанк.
- [ ] Проверить фактическое поведение LM Studio при переполнении контекста: ошибка, partial output, `stopReason` или silent truncation.
- [ ] Строить budget control на своём preflight и fallback, а server-side overflow policy использовать только после отдельной верификации.
- [x] Ввести versioned JSON-manifest чанков: `schema_version`, `chunk_id`, `t_start`, `t_end`, кадры, артефакты, `status`, `attempts`, `error`.
- [ ] Поддержать повторный запуск одного чанка и idempotent resume по manifest.
- [ ] Зафиксировать контракт финального ответа: `answer`, `evidence[]`, таймкоды, цитаты транскрипта, ссылки на кадры, признак неопределённости.

## 6. Budgeting and runtime scheduling

- [ ] Для текста считать budget максимально точно через tokenizer/совместимый счётчик выбранной модели.
- [ ] Для изображений использовать консервативную offline-эвристику с явным запасом.
- [ ] Резервировать budget под финальный ответ и под служебные instructions, а не только под input.
- [ ] Показать пользователю грубую оценку budget до старта и причину fallback/дробления при overflow.
- [ ] Ввести runtime scheduler: одновременно активна только одна тяжёлая нейросеть.
- [ ] Для конфигурации `8 GB VRAM / 64 GB RAM` зафиксировать политику `active -> offload to RAM -> unload`, чтобы `Whisper`, `Qwen` и другие модели не конкурировали бесконтрольно.
- [ ] Сериализовать model-heavy этапы и не разрешать параллельный inference в GUI без явной очереди.
- [ ] Отдельно задокументировать, какие лимиты зависят от модели, а какие являются общими эвристиками приложения.

## 7. Model and LM Studio integration

- [ ] Зафиксировать целевой профиль локальной модели и её ограничения для `Video QA`.
- [ ] Проверить связку `LM Studio + выбранная Qwen/VLM` на реальном мультимодальном запросе, а не только на тексте.
- [ ] Проверить, насколько локальный сервер поддерживает structured output / JSON contract, и предусмотреть graceful fallback.
- [ ] Добавить в репозиторий краткий internal reference по LM Studio: OpenAI-compatible API, multimodal payload, streaming, caveats structured output.

## 8. Outputs and artifacts

- [ ] Оставить текущие `TXT`/`SRT`/`VTT`/`JSON` экспортёры как базовый путь.
- [ ] Для `Video QA` добавить machine-readable export ответа и evidence-списка.
- [ ] Сохранять итоговый answer bundle рядом с manifest, чтобы можно было разбирать run post factum.
- [ ] Для ответа по видео дать формат с цитатами, таймкодами и ссылками на кадры.

## 9. Naming, legal and release prep

- [x] Зафиксировать `AskVLM` как canonical product name в GUI, CLI, build/help-текстах, metadata и документации.
- [x] Найти и обновить legacy-упоминания старого бренда приложения и старые абсолютные пути в коде и документации.
- [x] Перевести `QSettings`, session keys и exporter metadata на `AskVLM` без legacy compatibility branches и fallback-парсинга.
- [ ] Подготовить MIT readiness checklist: лицензии зависимостей, бинарей, model weights и вспомогательных инструментов.
- [ ] До внедрения URL import проверить не только лицензии, но и ToS/redistributability для конкретных source adapters.
- [ ] Решить, какие external tools поставляются вместе с приложением, а какие пользователь устанавливает отдельно.
- [ ] Добавить user-facing дисклеймер про ответственность за источник контента при URL import.

## 10. Tests and verification

- [ ] Проверить стартовый выбор экрана, восстановление экрана прошлой сессии и ручное переключение между экранами.
- [ ] Проверить, что `Text` и `Subtitles` пути не деградируют после добавления `Video QA`.
- [ ] Проверить `input providers`: локальный файл, ошибки URL resolve, временные файлы, cleanup.
- [ ] Проверить сбор контекста из вложений и их budget trimming.
- [ ] Проверить chunk planning, manifest persistence и resume по `chunk_id`.
- [ ] Проверить overflow fallback order, реакцию на server-side context limit и объяснимость ошибок пользователю.
- [ ] Проверить scheduler и правило `one active model at a time`.
- [ ] Проверить формат ответа с таймкодами, цитатами и привязкой к кадрам.
- [ ] Прогонять локальную проверку через `./run.ps1 -SkipLaunch` перед заявлением о готовности.

## 11. Explicitly not now

- Не тянуть тяжёлые research-репозитории long-video / Video QA в ближайший этап.
- Не делать `LangChain` обязательным ядром orchestrator без отдельной необходимости.
- Не обещать точный offline token count для изображений, пока нет model-specific расчёта.
