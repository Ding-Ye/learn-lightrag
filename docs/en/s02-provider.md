---
title: "s02 · Provider interface (OpenAI chat completion)"
chapter: 2
slug: s02-provider
est_read_min: 10
---

# s02 · Provider interface (OpenAI chat completion)

> What this teaches: take s01's inline OpenAI HTTP call out of `provider.go` and grow it into a real chapter — the single-method `Provider` interface stays frozen, while we add functional options, retry-with-backoff, and three impls (OpenAI / Mock / Echo). `Provider` is the "LLM entry point" that every later chapter consumes; the shape pinned here will hold all the way through s11.

---

## Problem / 问题

s01's `provider.go` mashed the OpenAI HTTP call, error handling, and token counting into a 60-line `OpenAIProvider.Complete` method. It works, but it leaves four learning gaps:

1. **Construction is awkward.** `NewOpenAIProvider()` takes no args, the model is hard-coded to `"gpt-4o-mini"`, and switching the base URL (Azure / vLLM / LM Studio) means editing the source. Are we really going to keep a fat constructor when Phase G adds Anthropic / Bedrock?
2. **No retry.** Upstream LightRAG wraps the LLM call with tenacity's `stop_after_attempt(3) + wait_exponential` — it auto-retries on 429s and transient 5xx. s01 just translates 429 into a "use mock" error string. Brittle in production.
3. **Token counts unused.** `CompleteResponse` already has `InputTokens` / `OutputTokens` fields, but s01's mock just stuffs in a rough estimate and no caller demonstrates "how do I compute throughput from this".
4. **Tests and the mock are tangled.** s01's `MockProvider` is mainly used by `pipeline.go`; there's no easy way to do "did the caller pass the right system prompt?" assertions, no swappable response, no atomic counter, no record of every call's shape.

s02's goal: solve all four, **without changing any of the interface shapes pinned in s01**.

## Solution / 解决方案

Promote `Provider` to its own chapter:

1. **`provider.go`** — still `type Provider interface { Complete(...) }` with one method, types byte-identical to s01. Add the `ProviderOption func(*OpenAIProvider)` functional-option type and a `Logger` (printf-style sink) interface.
2. **`provider_openai.go`** — `NewOpenAIProvider(WithModel(...), WithBaseURL(...), WithTimeout(...), WithMaxRetries(...), WithLogger(...))`. `Complete()` runs a retry loop — only on `transientStatusError` (429 / ≥500), up to `MaxRetries` times, with 200ms → 600ms → 1.8s backoff plus ±20% jitter.
3. **`provider_mock.go`** — `MockProvider` defaults to sha256(full request JSON) for byte-identical responses across runs. Adds a `Resp func(req) string` injection point + `Count() int64` atomic counter + `Calls []CompleteRequest` history.
4. **`provider_echo.go`** — `EchoProvider`, 30 lines, returns `req.System` verbatim. The smallest-possible Provider impl, useful for "did the caller assemble the right system prompt?" smoke tests.

Caller code (`main.go`) is just a switch: `-provider openai | mock | echo`. Phase G adds Anthropic by adding one new file `provider_anthropic.go` plus one extra case — the `Provider` interface itself never changes.

## How It Works / 工作原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                  s02 Provider, three implementations                  │
│                                                                      │
│   Provider (interface, single method Complete)                       │
│        ▲   ▲   ▲                                                    │
│        │   │   └─ EchoProvider     ── 30 lines, returns req.System  │
│        │   └───── MockProvider     ── sha256-determ + Calls/Count   │
│        └───────── OpenAIProvider   ── HTTP + backoff retry + Logger │
│                       │                                              │
│                       ▼                                              │
│             ┌───────────────────┐                                    │
│             │ Complete(ctx, req)│                                    │
│             └───────────────────┘                                    │
│                       │                                              │
│             ┌─────────▼─────────┐                                   │
│             │  for attempt 0..N │ ←─ MaxRetries (default 2)         │
│             │      doOnce()     │                                    │
│             │   ↓ 200/4xx perm  │                                    │
│             │   ↓ 429/5xx trans │ → wait backoffDelay(attempt)       │
│             │                   │   200ms → 600ms → 1.8s + jitter    │
│             └───────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

Core 30 lines (excerpt from [`agents/s02-provider/provider_openai.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s02-provider/provider_openai.go)):

```go
func (p *OpenAIProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
    if p.APIKey == "" {
        return CompleteResponse{}, errors.New("openai: OPENAI_API_KEY is empty (set it or use MockProvider)")
    }
    // ...assemble messages, marshal body...

    var lastErr error
    for attempt := 0; attempt <= p.MaxRetries; attempt++ {
        if attempt > 0 {
            wait := backoffDelay(attempt) // 200ms / 600ms / 1.8s + jitter
            p.Logger("openai: retry attempt=%d wait=%s lastErr=%v", attempt, wait, lastErr)
            select {
            case <-ctx.Done():
                return CompleteResponse{}, fmt.Errorf("openai: cancelled while waiting to retry: %w", ctx.Err())
            case <-time.After(wait):
            }
        }
        resp, err := p.doOnce(ctx, url, body)
        if err == nil {
            p.Logger("openai: success attempt=%d input_tokens=%d output_tokens=%d",
                attempt, resp.InputTokens, resp.OutputTokens)
            return resp, nil
        }
        lastErr = err
        var tse *transientStatusError
        if !errors.As(err, &tse) {
            return CompleteResponse{}, err  // permanent: 401 / parse / non-429 4xx — no retry
        }
    }
    return CompleteResponse{}, fmt.Errorf("openai: exhausted %d retries: %w", p.MaxRetries, lastErr)
}
```

**4 non-obvious points**:

1. **`transientStatusError` is the load-bearing classification.** 429 and 5xx come back as `*transientStatusError`; 401 / JSON decode errors / other 4xx come back as plain `error`. The retry loop uses `errors.As(err, &tse)` to tell them apart — that's the real "retry on transient only" rule. Without this, a 401 would burn three attempts for nothing and confuse the logs.
2. **The backoff sleep listens on `ctx.Done()` too.** Each wait is a `select`, not a `time.Sleep`. A caller that cancels mid-retry returns immediately rather than blocking until the next attempt fires. `TestOpenAIProviderRespectsContextCancel` asserts exactly this.
3. **±20% jitter prevents thundering-herd.** `backoffDelay` puts a uniform ±20% jitter on top of `200ms × 3^(attempt-1)`. When many goroutines hit the same 429 wall, retries don't re-synchronize. Same idea as tenacity's default jitter.
4. **`MockProvider.Resp` is the injection point.** Default response is the sha256 hash of the request, but tests can pass `func(req) string { return "fake answer" }` for scripted answers. Lets the mock support both byte-stable regression and "the LLM said X next" play-style tests.

## What Changed / 与 s01 的变化

The core diff against s01 (compressed pseudo-diff):

```diff
 // provider.go
 type Provider interface {
     Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
 }
+type ProviderOption func(*OpenAIProvider)
+func WithModel(string) ProviderOption     { ... }
+func WithBaseURL(string) ProviderOption   { ... }
+func WithTimeout(time.Duration) ProviderOption { ... }
+func WithMaxRetries(int) ProviderOption   { ... }
+func WithLogger(Logger) ProviderOption    { ... }
+type Logger func(format string, args ...any)

 // provider_openai.go
 type OpenAIProvider struct {
     APIKey string
     Model  string
     HTTP   *http.Client
+    BaseURL    string  // default "https://api.openai.com/v1"
+    MaxRetries int     // default 2 (=> 3 total attempts)
+    Logger     Logger
 }
-func NewOpenAIProvider() *OpenAIProvider { ... }   // s01: zero-arg
+func NewOpenAIProvider(opts ...ProviderOption) *OpenAIProvider { ... }
 func (p *OpenAIProvider) Complete(...) (...) {
-    // s01: single attempt; 429 returns "use mock" error
+    // s02: retry loop on transientStatusError (429 / >=500)
+    //      backoff 200ms / 600ms / 1.8s + ±20% jitter
+    //      ctx-aware sleep so cancel propagates within waits
 }

 // provider_mock.go
 type MockProvider struct {
     Calls []CompleteRequest
+    Resp func(CompleteRequest) string  // test injection point
+    counter atomic.Int64                // Count() atomic counter
 }
-func NewMockProvider() *MockProvider { ... }
+func NewMockProvider(resp func(CompleteRequest) string) *MockProvider { ... }
+func (m *MockProvider) Count() int64 { ... }

 // provider_echo.go (NEW in s02)
+type EchoProvider struct{}
+func (*EchoProvider) Complete(...) { return CompleteResponse{Text: req.System, ...} }
```

Caller code is untouched: `provider.Complete(ctx, req)` is the same signature, `CompleteResponse{Text, InputTokens, OutputTokens}` has the same three fields. So Phase G adding Anthropic is one new file `provider_anthropic.go` + one extra `main.go` switch case — **zero existing imports change**.

## Try It / 动手试一试

```bash
cd agents/s02-provider

# 1. mock — what CI runs, deterministic
go run . -provider mock -q "What's the capital of France?"
# === Answer ===
# [mock-llm] sha256=4f3a... (deterministic echo)

# 2. echo — system prompt round-trips, validates "did the caller assemble it right?"
go run . -provider echo -q "(this question is ignored)"
# === Answer ===
# You are a concise assistant. Answer in one sentence.

# 3. real OpenAI (needs API key) + -v to see token counts
export OPENAI_API_KEY=sk-...
go run . -provider openai -model gpt-4o-mini -v -q "What's the capital of France?"
# === Answer ===
# Paris.
#
# === Tokens ===
# input=24 output=2  (s02 finally populates these — s01 left them best-effort)

# 4. tests (this is the CI matrix)
go test -v ./...
# 7 tests, all use httptest, never call out:
#   TestProviderInterfaceContract         compile-time interface contract
#   TestOpenAIProviderHappyPath           request body + response decode
#   TestOpenAIProviderRetriesOn429        429 x2 -> 200, asserts 3 attempts
#   TestOpenAIProviderRespectsContextCancel  cancel returns within 50ms
#   TestMockProviderDeterminism           two identical requests => byte-equal responses
#   TestEchoProviderReturnsSystemPrompt   System field round-trips
#   TestOpenAIProviderUnauthorizedNotRetried  401 must not retry
```

## Upstream Source Reading / 上游源码阅读

Upstream's `openai_complete_if_cache` in `lightrag/llm/openai.py` is the main LLM-call entry; it's wrapped with a tenacity retry decorator. Here's the load-bearing 30 lines (full annotated excerpt at [`upstream-readings/s02-openai.py`](../../upstream-readings/s02-openai.py)):

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
    **kwargs: Any,
) -> str:
    if history_messages is None:
        history_messages = []
    # build AsyncOpenAI client (base_url / api_key / timeout overrides)
    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, timeout=timeout, ...
    )
    # assemble messages: [system?] + history + [user]
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # the actual HTTP call (wrapped by tenacity for 3 retries + exponential backoff)
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response.choices[0].message.content
```

**Reading notes**:

- **The `@retry` decorator vs. s02's loop.** Upstream wraps the function with tenacity; s02 uses an explicit `for attempt := 0; attempt <= p.MaxRetries` loop. Go has no decorators, but stdlib backoff is 10 lines and arguably more legible.
- **`stop_after_attempt(3)` = 3 total attempts.** Maps to s02's `WithMaxRetries(2)` (1 initial + 2 retries). Same semantics, different naming convention — upstream counts total tries; s02 counts "extra retries".
- **`retry_if_exception_type(RateLimitError | APIConnectionError | APITimeoutError | InvalidResponseError)` is the classification.** Equivalent to s02's `transientStatusError` + `errors.As` check. Upstream relies on SDK-thrown exception types; s02 doesn't use the SDK and looks at HTTP status codes directly.
- **`messages = [system, history..., user]` assembly order.** Lines up exactly with s02's `provider_openai.go` `if req.System != "" { msgs = append(msgs, {Role:"system",...}) }` followed by `for _, m := range req.Messages { ... }`.
- **`**kwargs` is an anti-pattern.** Upstream lets callers pass `temperature` / `max_tokens` through kwargs; in Go we use named struct fields (`CompleteRequest.Temperature` / `MaxTokens`). Classic "types replace dicts".

**Want more**: upstream's `lightrag/llm/openai.py` also contains `openai_embed` (s06 reads it for embeddings), `create_openai_async_client` (where baseURL / Azure switching lives), and `gpt_4o_mini_complete` — a thin wrapper injected as `llm_model_func` in the example demo. Phase G's multi-model guide will map this same shape to Anthropic / Bedrock / Ollama.

---

**Next chapter preview**: s03 promotes `DocStatus` from s01's placeholder string constants into a state machine with persistence, transition validation, and preserved error messages. Combined with s05's persisted KVStore, this is what makes "kill the process mid-insert, restart, resume from PROCESSING" actually work — LightRAG's resume-on-failure killer feature.
