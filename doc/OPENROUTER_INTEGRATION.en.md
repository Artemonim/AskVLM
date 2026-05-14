# OpenRouter Integration Reference

Updated: 2026-04-02

Technical documentation for AskVLM integration with OpenRouter in `Video QA` mode. Describes API behavior confirmed by manual test runs.

## 1. What Was Verified

- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Model: `qwen/qwen3.6-plus:free`
- Client: `core/video_qa_lm_studio_client.py`
- Prompt building: the same chunk contract used by `Video QA`, via `build_chunk_analysis_prompt(...)`
- Image: architecture diagram from `Multimodal GUI Design 01`
- Authorization: `OPENROUTER_API_KEY`, loaded from `.env` at GUI startup

## 2. What Turned Out to Be Required

OpenRouter does not accept a string for `reasoning`, but a configuration object.

```json
{"effort": "none"}
```

This corresponds to no-reasoning mode.

```json
{"effort": "low"}
```

This enables reasoning mode and returns non-zero `reasoning_tokens`.

If you pass a string like `"off"` or `"on"`, OpenRouter returns an error:

- `HTTP 400`
- `reasoning: Invalid input: expected object, received string`

## 3. What OpenRouter Returns

On a successful response, `choices[0].message` contains the following fields:

- `content`
- `reasoning`
- `refusal`
- `role`

With reasoning enabled, additionally:

- `reasoning_details`

In `usage.completion_tokens_details`:

- `reasoning_tokens`

## 4. Practical Observations

Both variants were confirmed on the same image:

- `reasoning={"effort":"none"}` -> `reasoning_tokens = 0`
- `reasoning={"effort":"low"}` -> `reasoning_tokens > 0`

Three parallel requests launched with a 2-second delay between starts all completed successfully:

- `finish_reason = stop`
- structured JSON response was valid
- response content correctly described the architecture diagram in the image

## 5. Errors to Account For

- A too-small image may receive an upstream error from the model:
  - example: `InternalError.Algo.InvalidParameter`
  - cause: image is smaller than the minimum allowed size on one side
- If OpenRouter returns JSON with `error` but without `choices`, the client branch should show the error body, not just a generic parse failure
- For the reasoning contract you cannot rely on local `on/off`; a `reasoning` object is required

## 6. What This Means for AskVLM

- `utils.env.load_env_file(...)` must run before GUI creation so that `OPENROUTER_API_KEY` is in `os.environ`
- For OpenRouter, use a `reasoning` object, not a string flag
- For video chunks, it is useful to add frame size validation before sending the request
- On OpenRouter errors, prefer showing `error.message` from the response body if it is present

## 7. Recommended Working Template

- `base_url`: `https://openrouter.ai/api/v1`
- `model`: `qwen/qwen3.6-plus:free`
- `reasoning`: `{"effort": "low"}` for reasoning mode or `{"effort": "none"}` to disable reasoning
- `json_schema`: AskVLM chunk JSON schema
- `image_paths`: list of frames encoded by the client in OpenAI-compatible format

## 8. Model Information

Official entry point for the model catalog: [List all models and their properties](https://openrouter.ai/docs/api-reference/models/get-models).

### 8.1. Whether a Model Exists on OpenRouter

- Method: `GET https://openrouter.ai/api/v1/models`
- Header: `Authorization: Bearer <OPENROUTER_API_KEY>` (marked as required in OpenAPI)
- Response: JSON with field `data: Model[]`
- Check: find the object whose `id` matches what the user entered in the GUI (e.g. `qwen/qwen3.6-plus:free`)

Live snapshot on `2026-04-02` for `qwen/qwen3.6-plus:free`: model **is present** in the list, `name`: "Qwen: Qwen3.6 Plus (free)", `context_length`: `1000000`.

### 8.2. Which `effort` Values Are Available for Reasoning

The unified `reasoning` contract is described in [Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens): inside the `reasoning` object the `effort` field can be one of:

- `xhigh`, `high`, `medium`, `low`, `minimal`, `none`

Separately, the [chat completion schema](https://openrouter.ai/docs/api-reference/chat-completion) lists the `ChatRequestReasoningEffort` enum with the same set of values.

Important: this is a **global** API contract. At the individual model level, check `supported_parameters` in the Models API response: if `reasoning` is listed there, the model declares support for the `reasoning` parameter (object). For `qwen/qwen3.6-plus:free` on `2026-04-02`, `supported_parameters` includes `reasoning` and `include_reasoning`; the string `reasoning_effort` is **absent** (this does not mean the provider will never accept `reasoning_effort`, but for this model card OpenRouter does not advertise it).

### 8.3. Whether the Model Supports Image Input

In the model object, check `architecture.input_modalities` and `architecture.output_modalities` (field descriptions in [Models overview](https://openrouter.ai/docs/guides/overview/models)).

For `qwen/qwen3.6-plus:free` on `2026-04-02`:

- `input_modalities`: `text`, `image`, `video`
- `output_modalities`: `text`

So multimodal input with images for this model is **declared** in the catalog.

### 8.4. Cost of a Specific Model

In the model object, the `pricing` field (see [Models overview](https://openrouter.ai/docs/guides/overview/models)): values in USD per token or per unit; `"0"` means "free" for the corresponding metric.

For `qwen/qwen3.6-plus:free` on `2026-04-02`:

- `pricing.prompt`: `0`
- `pricing.completion`: `0`

Interpretation: in the catalog this model is marked as **free tier** for prompt/completion for the current card. Actual billing should still be verified in the `usage` response and in the OpenRouter dashboard if paid options or other providers are enabled.

## 9. Manual Validation Result (chat completions)

Verification showed that OpenRouter correctly accepts images and returns a structured response for the architecture diagram when:

- `OPENROUTER_API_KEY` is loaded
- reasoning is passed as an object
- the image has an acceptable size
