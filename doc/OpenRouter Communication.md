# AskVLM — общение с OpenRouter

Дата: 2026-04-02  
Обновлено: 2026-04-02 (разведка Models API + ссылки на официальные страницы)

Краткая внутренняя заметка по тому, как AskVLM должен общаться с OpenRouter в `Video QA` и какие ответы были подтверждены ручными прогонами.

## 1. Что проверено

- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Модель: `qwen/qwen3.6-plus:free`
- Клиент: `core/video_qa_lm_studio_client.py`
- Формирование prompt: тот же chunk-контракт, что использует `Video QA`, через `build_chunk_analysis_prompt(...)`
- Изображение: архитектурная схема `Multimodal GUI Design 01`
- Авторизация: `OPENROUTER_API_KEY`, загружаемый из `.env` при старте GUI

## 2. Что оказалось обязательным

OpenRouter принимает не строку `reasoning`, а объект конфигурации.

```json
{"effort": "none"}
```

Это соответствует режиму без reasoning.

```json
{"effort": "low"}
```

Это даёт reasoning-режим и возвращает ненулевые `reasoning_tokens`.

Если передать строку вроде `"off"` или `"on"`, OpenRouter отвечает ошибкой:

- `HTTP 400`
- `reasoning: Invalid input: expected object, received string`

## 3. Что возвращает OpenRouter

При успешном ответе в `choices[0].message` доступны поля:

- `content`
- `reasoning`
- `refusal`
- `role`

При включённом reasoning дополнительно может приходить:

- `reasoning_details`

В `usage.completion_tokens_details` доступны:

- `reasoning_tokens`

## 4. Практические наблюдения

На одном и том же изображении были подтверждены оба варианта:

- `reasoning={"effort":"none"}` -> `reasoning_tokens = 0`
- `reasoning={"effort":"low"}` -> `reasoning_tokens > 0`

Три параллельных запроса, запущенные с задержкой 2 секунды между стартами, все завершились успешно:

- `finish_reason = stop`
- structured JSON ответ был валиден
- содержание ответа корректно описывало архитектурную схему на картинке

## 5. Ошибки, которые нужно учитывать

- Слишком маленькое изображение может получить upstream-ошибку от модели:
  - пример: `InternalError.Algo.InvalidParameter`
  - причина: изображение меньше минимально допустимого размера по одной из сторон
- Если OpenRouter возвращает JSON с `error`, но без `choices`, клиентская ветка должна показывать именно тело ошибки, а не только общий сбой парсинга
- Для reasoning-контракта нельзя полагаться на локальный `on/off`; нужен объект `reasoning`

## 6. Что это значит для AskVLM

- `utils.env.load_env_file(...)` должен выполняться до создания GUI, чтобы `OPENROUTER_API_KEY` попадал в `os.environ`
- Для OpenRouter надо использовать `reasoning`-объект, а не строковый флаг
- Для видео-чанков полезно добавить проверку размеров кадров до отправки запроса
- При ошибках OpenRouter лучше показывать `error.message` из тела ответа, если оно есть

## 7. Рекомендуемый рабочий шаблон

- `base_url`: `https://openrouter.ai/api/v1`
- `model`: `qwen/qwen3.6-plus:free`
- `reasoning`: `{"effort": "low"}` для reasoning-режима или `{"effort": "none"}` для отключения reasoning
- `json_schema`: chunk JSON schema AskVLM
- `image_paths`: список кадров, закодированных клиентом в OpenAI-compatible формат

## 8. Информация о моделях

Официальная точка входа для каталога моделей: [List all models and their properties](https://openrouter.ai/docs/api-reference/models/get-models).

### 8.1. Существует ли модель на стороне OpenRouter

- Метод: `GET https://openrouter.ai/api/v1/models`
- Заголовок: `Authorization: Bearer <OPENROUTER_API_KEY>` (в OpenAPI помечен как обязательный)
- Ответ: JSON с полем `data: Model[]`
- Проверка: найти объект, у которого `id` совпадает с тем, что пользователь ввёл в GUI (например `qwen/qwen3.6-plus:free`)

Живой снимок на `2026-04-02` для `qwen/qwen3.6-plus:free`: модель **присутствует** в списке, `name`: «Qwen: Qwen3.6 Plus (free)», `context_length`: `1000000`.

### 8.2. Какие `effort` доступны для reasoning

Унифицированный контракт `reasoning` описан в [Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens): внутри объекта `reasoning` поле `effort` может быть одним из:

- `xhigh`, `high`, `medium`, `low`, `minimal`, `none`

Отдельно в [схеме chat completion](https://openrouter.ai/docs/api-reference/chat-completion) перечислен enum `ChatRequestReasoningEffort` с тем же набором значений.

Важно: это **глобальный** контракт API. На уровне конкретной модели смотрите `supported_parameters` в ответе Models API: если там есть строка `reasoning`, модель заявляет поддержку параметра `reasoning` (объекта). Для `qwen/qwen3.6-plus:free` на `2026-04-02` в `supported_parameters` есть `reasoning` и `include_reasoning`, строки `reasoning_effort` **нет** (это не значит, что провайдер никогда не примет `reasoning_effort`, но для этой карточки модели OpenRouter его не рекламирует).

### 8.3. Поддерживает ли модель image input

В объекте модели смотрите `architecture.input_modalities` и `architecture.output_modalities` (описание полей — в [Models overview](https://openrouter.ai/docs/guides/overview/models)).

Для `qwen/qwen3.6-plus:free` на `2026-04-02`:

- `input_modalities`: `text`, `image`, `video`
- `output_modalities`: `text`

То есть мультимодальный ввод с картинками для этой модели в каталоге **заявлен**.

### 8.4. Стоимость конкретной модели

В объекте модели поле `pricing` (см. [Models overview](https://openrouter.ai/docs/guides/overview/models)): значения в USD за токен или за единицу; `"0"` означает «бесплатно» для соответствующей метрики.

Для `qwen/qwen3.6-plus:free` на `2026-04-02`:

- `pricing.prompt`: `0`
- `pricing.completion`: `0`

Интерпретация: в каталоге эта модель отмечена как **free tier** по prompt/completion для текущей карточки. Фактический биллинг всё равно смотрите в ответе `usage` и в кабинете OpenRouter, если включены платные опции или другие провайдеры.

## 9. Результат ручной валидации (chat completions)

Проверка показала, что OpenRouter корректно принимает изображения и возвращает структурированный ответ по архитектурной схеме, если:

- загружен `OPENROUTER_API_KEY`
- reasoning передан объектом
- изображение имеет допустимый размер

