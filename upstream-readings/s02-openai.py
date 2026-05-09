# Source: https://github.com/HKUDS/LightRAG  (path: lightrag/llm/openai.py, branch `main`, 2026-05-09)
# License: MIT (Copyright 2025 LightRAG Team)
# Excerpted for learn-lightrag/s02-provider — annotated, not for execution.
#
# This file collects the load-bearing slice of upstream that s02 condenses into
# Go (~210 LOC of provider_openai.go + supporting types):
#
#   1. The retry decorator (stop_after_attempt / wait_exponential / retry_if_*)
#   2. openai_complete_if_cache  — main chat-completion wrapper (~50 lines)
#   3. The message assembly snippet                  ([system?] + history + [user])
#
# Each block is followed by reading notes that point out: what s02 kept, what
# s02 simplified, and where Phase G picks up the loose ends.

# ─────────────────────────────────────────────────────────────────────────────
# (1) The retry decorator (the @retry on top of openai_complete_if_cache)
# ─────────────────────────────────────────────────────────────────────────────
# Upstream uses the `tenacity` library to wrap the LLM call. Three knobs are
# load-bearing: when to stop trying, how long to wait, and which exceptions
# qualify as "retry me".

@retry(
    stop=stop_after_attempt(3),                      # 3 total attempts (= 1 + 2 retries)
    wait=wait_exponential(multiplier=1, min=4, max=10),  # 4s, 8s, capped at 10s
    retry=(
        retry_if_exception_type(RateLimitError)      # 429-class — s02: *transientStatusError
        | retry_if_exception_type(APIConnectionError)  # network error — s02: net/http error
        | retry_if_exception_type(APITimeoutError)     # request-level timeout
        | retry_if_exception_type(InvalidResponseError)  # malformed JSON / schema mismatch
    ),
)

# Reading notes (retry decorator):
#
# - In s02 we re-implement this with a stdlib for-loop in Complete():
#       for attempt := 0; attempt <= p.MaxRetries; attempt++ { ... }
#   `MaxRetries=2` matches `stop_after_attempt(3)` (different naming: upstream
#   counts total tries, s02 counts "extra retries").
#
# - The `wait_exponential` defaults are tenacity's geometric series:
#   `wait = multiplier * (2 ** previous_attempt_number)` capped by `max`.
#   s02 uses 200ms × 3^(attempt-1) plus ±20% uniform jitter — same shape, faster
#   wall-clock for the teaching demo (200ms vs. 4s makes the "watch retry happen
#   in -v output" loop feel responsive; production code would use 4s+).
#
# - `retry_if_exception_type(...)` is the classification. The four types map to
#   s02 as follows:
#       RateLimitError       → HTTP 429              → *transientStatusError
#       APIConnectionError   → http.Client.Do error  → wrapped error (returned as-is in s02; not retried)
#       APITimeoutError      → ctx.DeadlineExceeded  → ctx.Err() in Go
#       InvalidResponseError → json decode fail      → permanent error in s02 (not retried)
#   So s02 is slightly less aggressive: s02 only retries 429/5xx (the upstream
#   "definitely transient" subset). Network-level errors and parse failures
#   are surfaced immediately. We document that delta in `provider_openai.go`'s
#   doOnce() return-error policy.

# ─────────────────────────────────────────────────────────────────────────────
# (2) openai_complete_if_cache — the main chat-completion wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,                 # → s02: WithBaseURL("...")
    api_key: str | None = None,                  # → s02: WithAPIKey / OPENAI_API_KEY env
    token_tracker: Any | None = None,            # → s02: Logger callback (decoupled)
    stream: bool | None = None,                  # → s02: CompleteRequest.Stream (forced false)
    timeout: int | None = None,                  # → s02: WithTimeout(time.Duration)
    keyword_extraction: bool = False,            # → s09 chapter (structured output for KW extraction)
    use_azure: bool = False,                     # → Phase G addendum (Azure profile)
    azure_deployment: str | None = None,
    api_version: str | None = None,
    **kwargs: Any,                               # → s02 anti-pattern: replaced with named struct fields
) -> str:
    if history_messages is None:
        history_messages = []

    # Drop kwargs that aren't used at the API call site.
    kwargs.pop("hashing_kv", None)               # → s09 LLM cache; not in s02
    client_configs = kwargs.pop("openai_client_configs", {})

    if keyword_extraction:                       # → s09 hooks structured output here
        kwargs["response_format"] = GPTKeywordExtractionFormat

    # Build a fresh AsyncOpenAI client every call. The factory handles
    # base_url / azure_deployment / proxy / timeout overrides.
    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        use_azure=use_azure,
        azure_deployment=azure_deployment,
        api_version=api_version,
        timeout=timeout,
        client_configs=client_configs,
    )

    # ─── Message assembly: [system?] + history + [user] ──────────────────
    # This is the snippet that s02's provider_openai.go mirrors line-for-line.
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # The actual HTTP call. The tenacity decorator on this function wraps
    # the WHOLE async function, so a RateLimitError thrown here triggers
    # the retry logic above.
    response = await openai_async_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,                                # temperature, max_tokens, …
    )
    if hasattr(response, "__aiter__"):
        # Streaming branch — yields chunks. s02 forces stream=false; Phase G enables.
        return inner_stream(response, token_tracker, ...)
    return response.choices[0].message.content

# Reading notes (openai_complete_if_cache):
#
# - Signature is one big positional+kwargs jumble (15 named params + **kwargs).
#   s02 splits this into:
#       OpenAIProvider struct (long-lived: APIKey/Model/BaseURL/HTTP/MaxRetries/Logger)
#       CompleteRequest struct (per-call: Model/System/Messages/MaxTokens/Temperature/Stream)
#   This is "named struct fields replace **kwargs" — a load-bearing simplification.
#
# - `keyword_extraction=True` switches the response into structured-output mode
#   (used by s09 for entity-extraction). s02 doesn't expose it; s09's extractor
#   will add a `WithStructuredOutput()` option instead of a boolean.
#
# - `token_tracker` is upstream's hook for streaming token-usage telemetry to
#   external observability. s02 keeps it simpler: a `Logger` callback receives
#   "input_tokens=N output_tokens=N" lines after each call. Phase G tightens this.
#
# - `use_azure=True` flips the client factory to use Azure's deployment-based URL.
#   s02 doesn't ship Azure; Phase G's multi-model guide adds an `AzureProvider`
#   (or just teaches `WithBaseURL` for Azure-compatible endpoints).
#
# - `**kwargs` propagation. The OpenAI SDK accepts a long tail of optional
#   params (top_p, presence_penalty, response_format, ...). Upstream forwards
#   them all blindly. s02 names the four most common (Model / MaxTokens /
#   Temperature / Stream); the rest are deferred to "future field on
#   CompleteRequest" if a chapter actually needs them.

# ─────────────────────────────────────────────────────────────────────────────
# (3) Why s02 stops here: companion functions deferred to other chapters
# ─────────────────────────────────────────────────────────────────────────────
#
# `lightrag/llm/openai.py` also contains:
#   - `openai_embed(...)`           — embedding wrapper (=> s06)
#   - `create_openai_async_client`  — factory for AsyncOpenAI w/ all the env knobs (=> Phase G)
#   - `gpt_4o_mini_complete`        — thin partial application of openai_complete_if_cache;
#                                      this is what `examples/lightrag_openai_demo.py` injects
#                                      as `llm_model_func` (=> Appendix B reading map)
#   - error classes: `InvalidResponseError`, etc. (=> Phase G if extending retry policy)
#
# We chose `openai_complete_if_cache` as the s02 anchor because it's the
# narrowest possible "LLM input → string output" wrapper that any later chapter
# (s09 extraction, s10 summarization, s11 query) calls. Each later chapter adds
# one feature ON TOP of the s02 contract:
#   s06 → adds the EmbeddingProvider sibling for the embed endpoint
#   s09 → adds the LLM response cache (hash-keyed) on top of Provider.Complete
#   s11 → finally enables Stream=true and surfaces token chunks to the caller

# ─────────────────────────────────────────────────────────────────────────────
# Reading map (where to go next)
# ─────────────────────────────────────────────────────────────────────────────
# After s02 you've seen the LLM-call shape. From here:
#
#   s06 → lightrag/llm/openai.py (openai_embed slice)   — embedding Provider sibling
#   s09 → lightrag/operate.py (extract_entities)         — uses Provider.Complete with
#                                                          structured-output mode
#   s10 → lightrag/operate.py (_summarize_descriptions)  — recursive Provider.Complete
#   s11 → lightrag/operate.py (kg_query)                 — adds streaming + multi-mode
#
# Phase G addendum (multi-model guide):
#   - lightrag/llm/anthropic.py     — Anthropic Messages API wrapper
#   - lightrag/llm/bedrock.py       — AWS Bedrock chat
#   - lightrag/llm/ollama.py        — Ollama local server (OpenAI-compatible URL)
#   - lightrag/llm/azure_openai.py  — Azure baseURL + api-version
#
# All four match s02's CompleteRequest / CompleteResponse shape; Phase G is just
# "one new file per provider, adding a switch case in main.go".
