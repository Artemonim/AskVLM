# TODO legacy — архив предыдущей дорожной карты

Дата архивации: 2026-03-29

Этот файл сохраняет прежний long-form roadmap проекта. Полный исторический чеклист остаётся доступен в git history, а здесь оставлено компактное резюме, чтобы старый scope было легче читать и не смешивать с новым мультимодальным планом.

## Что было в прежнем TODO

- Базовая локальная транскрипция: `FFmpeg -> Whisper/WhisperX -> diarization -> LLM formatting`.
- Quick Transcribe GUI: загрузка файла/папки, запуск в фоне, прогресс, предпросмотр, прожиг субтитров.
- AutoSubtitles и Resolve flow: SRT-first workflow, burn-in, sidecar export.
- Preview, autosave, metadata и status scanning.
- Queue parallelism, performance tuning и quality gates.
- Отложенные направления: disfluency cleanup, advanced editor, cloud integration, packaging, release automation.

## Почему файл архивирован

- Активный scope сместился от чистого транскрайбера к мультимодальному GUI.
- Новый `TODO.md` должен содержать только текущие ближайшие шаги.
- Длинный исторический checklist мешал быстро видеть, что делать следующим инкрементом.

## Как пользоваться

- Для текущих задач смотри `TODO.md`.
- Для архитектурного контекста смотри `doc/Multimodal GUI Design.md`.
- Для прежней подробной истории используй git history.

