# Независимый, более скептичный отчёт

Краткое резюме  
- Прямых drop-in оркестраторов «видео → чанки → кадры → LM Studio → ответ» нет. Придётся собирать из ffmpeg/scene detection + собственного манифеста + OpenAI-совместимого клиента.  
- Есть пару практичных OSS-утилит, уменьшающих объём самописного кода: лёгкие обёртки для keyframes/scene cuts, валидация структурированных ответов, и OpenAI-совместимый прокси с бюджетами.  
- Тяжёлые исследовательские репозитории по long-video/VideoQA остаются “на потом”: стоимость и интеграция выше, чем выигрыш в ближайшем релизе.

Топ-кандидаты (с пометками)  
- **PySceneDetect** (BSD-3, `scenedetect[opencv]`; GitHub Breakthrough/PySceneDetect) — «сразу»: надёжный shot/scene detection, умеет ключевые кадры и JSON/CSV отчёты; может заменить ручные ffmpeg-пайпы.  
- **scenecut-extractor** (PyPI, ffmpeg `select`/`scene`) — «сразу»: тонкая обёртка для получения JSON/CSV со списком сцен без OpenCV; хорошо для быстрых эвристик, если PySceneDetect избыточен.  
- **video-kf** (PyPI) — «после прототипа»: три метода keyframes (I-frames, color, flow), сам тянет ffmpeg; удобно для representative frame per shot без глубоких настроек.  
- **Guardrails AI** (MIT; guardrailsai.com, github.com/guardrails-ai/guardrails) — «после прототипа»: валидация/реструктурирование JSON-ответов по Pydantic/JSON Schema, может помочь с «структурированный ответ с таймкодами/цитатами» и автоповторами. Проверить совместимость с вашей VLM через LM Studio (json_schema и constrained decoding могут не поддерживаться).  
- **LiteLLM Proxy** (MIT; litellm.ai, github.com/BerriAI/litellm) — «сразу»: OpenAI-совместимый прокси с бюджетами, лимитами и логами; можно поставить перед LM Studio или смешанным стеком. Даёт preflight по цене/лимитам даже локально и единый клиент для GUI.  
- **Pydantic v2 + json_schema/instructor-like промпты** — «сразу»: базовый путь для структурированного ответа и ретраев, без тяжёлых зависимостей. Guardrails можно добавить, если понадобится жёсткая проверка.  

Что не стоит тащить сейчас  
- Тяжёлые research VideoQA/long-video модели (LongVILA, LWM, Eagle, Chapter-LLaMA и пр.): требуют GPU-инфры, не дают drop-in SDK под LM Studio.  
- LangChain ради GUI/маршрутизации: дублирует ваш оркестратор, выигрыша мало.  
- TorchCodec/Decord — только если упираетесь в производительность frame extraction; иначе PySceneDetect/ffmpeg достаточно.

Риски и ограничения  
- LM Studio + выбранная VLM могут не поддерживать строгий `json_schema`/constrained decoding: нужен fallback (например, Pydantic-валидация и повторный запрос с исправлением).  
- Preflight визуальных токенов офлайн остаётся эвристикой: точные числа есть только у облачных API; локально — закладывать запас и фиксировать модель-специфичные тайлы.  
- FFmpeg версии/бинарники под Windows: scenecut-extractor/video-kf притягивают или требуют ffmpeg; нужно зафиксировать путь и версию, чтобы не размножить конфликты.  
- Scene ≠ смысловой сегмент: придётся объединять/дополнять по транскрипту для ответов с таймкодами.  
- Guardrails даёт жёсткую валидацию, но увеличивает время отклика и может ломаться на нестабильных VLM-потоках.

3–5 следующих шагов  
- Зафиксировать минимальную схему `chunk_manifest` (schema_version, chunk_id, t_start/t_end, frames[], hash/size, status/error, retry_count) и проверить ручной рестарт одного чанка.  
- Прогнать на реальном ролике PySceneDetect vs scenecut-extractor (ffmpeg scene) и решить, что включать по умолчанию; держать ровную сетку как fallback.  
- Собрать прототип структурированного ответа: промпт с JSON Schema + Pydantic-валидация, один ретрай; затем опционально включить Guardrails, если модель устойчива.  
- Поднять LiteLLM Proxy перед LM Studio и измерить budgets/logging; это сразу даст preflight по лимитам и удобный единый endpoint для GUI.  
- Описать эвристику representative frame в TODO: «1 кадр на сцену (по умолчанию средний таймкод), опция — кадр с макс. движением или I-frame, лимит кадров/чанк».

Как ужать отдельные TODO-пункты  
- «Representative frames» — указать политику выбора (центр сцены | кадр с макс. движением | первый после склейки) и лимиты на кадры/чанк.  
- «Chunk fallback при overflow» — записать порядок: (1) уменьшить число кадров, (2) уменьшить разрешение, (3) дробить текст, (4) повторить с другой сценовой пороговой эвристикой.  
- «Preflight бюджета» — разделить на «облако (точный token API)» и «локально (эвристика с коэффициентом X для картинок)».  
- «JSON-manifest» — добавить `schema_version`, `status`, `error`, `attempts`, пути/хэши артефактов и повторное исполнение по `chunk_id`.

Источники  
- PySceneDetect — github.com/Breakthrough/PySceneDetect, scenedetect.com  
- scenecut-extractor — pypi.org/project/scenecut-extractor  
- video-kf — pypi.org/project/video-kf  
- Guardrails AI — guardrailsai.com/docs, github.com/guardrails-ai/guardrails  
- LiteLLM Proxy — litellm.ai, github.com/BerriAI/litellm  

