# Qwen3.5-35B-A3B for AskVLM

A brief internal reference card for the target local model used in `Video QA` and related multimodal scenarios.

## 1. What This Model Is

- Model: `Qwen/Qwen3.5-35B-A3B`
- Class: multimodal LLM (`text + image`)
- Architecture: MoE, total `35B` parameters, approximately `3B` active per token
- Practical meaning for the project: a single capable local model for Russian language, reasoning, and image-conditioned answers via LM Studio

## 2. Why It Fits AskVLM

- Supports multilingual scenarios including Russian.
- Suitable for grounded video-based answers when transcript excerpts and representative frames are passed in the prompt.
- Provides a strong enough reasoning layer to assemble a final answer across chunks, not just describe frames.
- Formally has a large context window, but the application still requires strict budget management and chunking.

## 3. Limitations That Cannot Be Ignored

- An MoE model on local hardware is heavily dependent on quantization and the quality of the specific build in LM Studio.
- Having "vision" in the name or model card does not guarantee that a specific local artifact actually accepts images in the required format.
- Structured output may be unstable in a local OpenAI-compatible stack; a fallback via validation and retry is required.
- A large context window does not cancel RAM, latency, and response quality limitations on long multimodal inputs.

## 4. Practical Guidelines for AskVLM

- Keep `Qwen3.5-35B-A3B` as the primary candidate for `Video QA`, not as an unconditionally fixed production standard.
- Use it after a budget preflight: count text more precisely, estimate images heuristically with a margin.
- Do not run concurrently with another heavy model without a scheduler.
- Test three scenarios separately:
  - text question without images;
  - question with transcript plus a few frames;
  - long question with attachments and an evidence-oriented JSON response.

## 5. LM Studio Caveats

- Work via the OpenAI-compatible endpoint, but do not assume full identity with cloud providers.
- Before enabling structured output in the main pipeline, verify how the model behaves on the real AskVLM JSON contract.
- Before enabling the production path, verify actual behavior on context overflow: error, partial output, `stopReason`, or other signal.
- If the vision path in a specific build is unstable, keep the model as a text aggregator and temporarily relax the multimodal pass.

## 6. Current Working Decision

- Yes, keeping this model's card in the repository is useful.
- This model remains the target candidate for the first pass of `Video QA` via LM Studio.
- The specific quantization choice and final code fixation are only done after a live verification of the multimodal scenario on target hardware.

## 7. What Still Needs to Be Verified Manually

- Which exact artifact is loaded in LM Studio: text-only or multimodal.
- What the real usable context is on target hardware.
- Whether the model produces a stable JSON response with evidence fields.
- How much latency and RAM pressure degrade when running alongside `Whisper`.

## 8. Sources

- Hugging Face: [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- Model README: [official README](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/README.md)
- Model line overview: [Qwen 3.5 blog](https://qwen.ai/blog?id=qwen3.5)

## 9. LM Studio — Internal Reference (verified from official docs)

Brief captured semantics for the backend (`Video QA`), without replacing live hardware verification.

- **OpenAI-compatible HTTP**: LM Studio documentation describes OpenAI-compatible routes including `/v1/responses` and `/v1/chat/completions`. The exact JSON form and field availability depend on the endpoint and version; termination metadata cannot be assumed to always match cloud OpenAI.
- **Multimodal payload**: requests with images follow the OpenAI-compatible format (e.g. `content` as a list of parts with `image_url` / base64); details and limitations are in the official LM Studio sections on OpenAI compat and image input.
- **Streaming**: in SSE/streaming mode the final `finish_reason` / `stopReason` may appear only on the last chunk or in a separate structure; the client must wait for stream completion before relying on termination normalization.
- **`stopReason` (prediction stats)**: documented values include `maxPredictedTokensReached`, `stopStringFound`, `toolCalls`, `contextLengthReached`. Normalization code: `core/video_qa_lm_studio_termination.py`.
- **`contextOverflowPolicy`**: supported modes — `stopAtLimit`, `truncateMiddle`, `rollingWindow`. This is a server-side policy on context handling; the application still relies on its own preflight and fallback pending separate verification on the target model.
- **Structured output**: compatibility with strict JSON/schema depends on the model and build; graceful fallback remains mandatory.

Limitation: end-to-end behavior on context overflow (error vs partial output vs silent truncation) on target hardware and LM Studio version remains a **manual** verification; this section does not replace the roadmap item for live overflow verification.
