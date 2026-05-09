---
title: "s02 · 提供方接口 (OpenAI 聊天补全)"
chapter: 2
slug: s02-provider
est_read_min: 10
---

# s02 · 提供方接口 (OpenAI 聊天补全)

> 教什么：把 s01 内联在 `provider.go` 里的 OpenAI HTTP 调用提升成正式一节：单方法 `Provider` 接口冻结，加上 functional options、重试退避、三种实现（OpenAI / Mock / Echo）。`Provider` 是后面 9 节都要 import 的「LLM 入口」——这一节定的接口形态会一直保持到 s11。

---

## Problem / 问题

s01 的 `provider.go` 里 OpenAI 的 HTTP 调用、错误处理、token 统计是混在 `OpenAIProvider.Complete` 里 60 行直写的——能跑，但有四个学习痛点：

1. **构造方式不优雅**：`NewOpenAIProvider()` 是无参的，Model 写死成 `"gpt-4o-mini"`，想换 baseURL（Azure / vLLM / LM Studio）只能改源码。Phase G 加 Anthropic / Bedrock 时还得继续写「胖构造函数」吗？
2. **没有重试**：上游 LightRAG 用 `tenacity` 给 LLM 调用包了 `stop_after_attempt(3) + wait_exponential`——遇到 429 / 临时 5xx 自动重试。s01 直接把 429 翻译成「请用 mock」错误信息，到生产就脆。
3. **token 计数没真正用上**：`CompleteResponse` 已经有 `InputTokens` / `OutputTokens` 字段，但 s01 的 mock 只是塞个粗略估计，调用方也没演示「咋根据这个算吞吐」。
4. **测试和 mock 缠在一起**：s01 的 `MockProvider` 主要给 `pipeline.go` 用，没法做「caller 是不是把对的 system prompt 传过来了」这种细粒度断言。需要一个支持注入响应、记录调用、原子计数的 mock。

s02 的目标：在**完全不动 s01 已经定下来的接口形态**的前提下，把上面 4 个问题一次性解决。

## Solution / 解决方案

把 `Provider` 推进到一个完整章节，结构如下：

1. **`provider.go`**：仍然是 `type Provider interface { Complete(...) }` 这一个方法，类型形态和 s01 一字不差。新增 `ProviderOption func(*OpenAIProvider)` 函数式选项类型 + `Logger` 接口（可选 printf-style sink）。
2. **`provider_openai.go`**：`NewOpenAIProvider(WithModel(...), WithBaseURL(...), WithTimeout(...), WithMaxRetries(...), WithLogger(...))`。`Complete()` 内部走重试循环——只在 `transientStatusError`（429 / ≥500）时重试，最多 `MaxRetries` 次，间隔 200ms → 600ms → 1.8s 加 ±20% 抖动。
3. **`provider_mock.go`**：`MockProvider` 默认用 sha256(整个 request JSON) 作响应——相同请求字节相同回。新增 `Resp func(req) string` 注入点 + `Count() int64` 原子计数 + `Calls []CompleteRequest` 调用历史。
4. **`provider_echo.go`**：`EchoProvider`，30 行，把 `req.System` 原样返回。最小可能的 Provider 实现，给「caller 拼对 system prompt 了吗」这类烟测用。

调用方代码（`main.go`）只是个 switch：`-provider openai | mock | echo`。Phase G 加 Anthropic 就是加一个 `provider_anthropic.go` + 多一个 case，`Provider` 接口本体永远不动。

## How It Works / 工作原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                     s02 Provider 三家实现                            │
│                                                                      │
│   Provider (interface, 一个方法 Complete)                            │
│        ▲   ▲   ▲                                                    │
│        │   │   └─ EchoProvider     ── 30 行，return req.System      │
│        │   └───── MockProvider     ── sha256-determ + Calls/Count   │
│        └───────── OpenAIProvider   ── HTTP + 重试退避 + Logger      │
│                       │                                              │
│                       ▼                                              │
│             ┌───────────────────┐                                    │
│             │  Complete(ctx, req)│                                   │
│             └───────────────────┘                                    │
│                       │                                              │
│             ┌─────────▼─────────┐                                   │
│             │  for attempt 0..N │ ←─ MaxRetries (default 2)         │
│             │      doOnce()     │                                    │
│             │   ↓ 200/4xx 永久 │                                    │
│             │   ↓ 429/5xx 瞬时 │ → wait backoffDelay(attempt)        │
│             │                   │   200ms → 600ms → 1.8s + jitter    │
│             └───────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

核心 30 行（节选自 [`agents/s02-provider/provider_openai.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s02-provider/provider_openai.go)）：

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
            return CompleteResponse{}, err  // 永久错误：401 / 解析失败 / 4xx 不重试
        }
    }
    return CompleteResponse{}, fmt.Errorf("openai: exhausted %d retries: %w", p.MaxRetries, lastErr)
}
```

**4 个非显然之处**：

1. **`transientStatusError` 是分类的关键**：429 和 5xx 走 `*transientStatusError`，401 / JSON 解码失败 / 其他 4xx 走普通 `error`。retry 循环用 `errors.As(err, &tse)` 区分——这才是「retry on transient only」的载货实现。否则代码遇到 401 会重试 3 次，纯粹浪费 + 误导日志。
2. **退避中也要监听 `ctx.Done()`**：每次 sleep 用 `select`，不是 `time.Sleep`。这样调用方一 cancel 就立刻返回，不至于卡到下一次 attempt。`TestOpenAIProviderRespectsContextCancel` 就在测这个。
3. **抖动 ±20% 防雪崩**：`backoffDelay` 在 200ms × 3^(attempt-1) 上叠 ±20% 均匀抖动。多个 goroutine 同时撞 429 时，重试时间不会再次同步。这是 tenacity 默认行为的小翻版。
4. **`MockProvider.Resp` 是注入点**：默认用 sha256(请求) 哈希，但测试可以传 `func(req) string { return "fake answer" }` 走自定义。让 mock 既能做 byte-stable 回归，也能做「LLM 假装说了某句话」的剧本测。

## What Changed / 与 s01 的变化

s02 相对 s01 的核心 diff（简化伪 diff）：

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
+    Resp func(CompleteRequest) string  // 测试注入点
+    counter atomic.Int64                // Count() 原子计数
 }
-func NewMockProvider() *MockProvider { ... }
+func NewMockProvider(resp func(CompleteRequest) string) *MockProvider { ... }
+func (m *MockProvider) Count() int64 { ... }

 // provider_echo.go (NEW in s02)
+type EchoProvider struct{}
+func (*EchoProvider) Complete(...) { return CompleteResponse{Text: req.System, ...} }
```

调用方完全没动：`provider.Complete(ctx, req)` 还是同一个签名，`CompleteResponse{Text, InputTokens, OutputTokens}` 还是同三个字段。所以 Phase G 加 Anthropic 是：新文件 `provider_anthropic.go` + `main.go` 多一个 switch case，**0 个现有 import 需要改**。

## Try It / 动手试一试

```bash
cd agents/s02-provider

# 1. mock：CI 永远跑这个，确定性
go run . -provider mock -q "What's the capital of France?"
# === Answer ===
# [mock-llm] sha256=4f3a... (deterministic echo)

# 2. echo：把 system prompt 原样吐出来，验证「caller 拼对了吗」
go run . -provider echo -q "(this question is ignored)"
# === Answer ===
# You are a concise assistant. Answer in one sentence.

# 3. 真 OpenAI（要 API key）+ -v 看 token 计数
export OPENAI_API_KEY=sk-...
go run . -provider openai -model gpt-4o-mini -v -q "What's the capital of France?"
# === Answer ===
# Paris.
#
# === Tokens ===
# input=24 output=2  (s02 finally populates these — s01 left them best-effort)

# 4. 跑测试（CI 用这个矩阵）
go test -v ./...
# 7 个测试，全用 httptest，不打外网：
#   TestProviderInterfaceContract         编译期接口契约
#   TestOpenAIProviderHappyPath           请求体 + 响应解码
#   TestOpenAIProviderRetriesOn429        429 ×2 → 200，验证 3 次尝试
#   TestOpenAIProviderRespectsContextCancel  cancel 50ms 内返回
#   TestMockProviderDeterminism           两次相同请求 → 字节相同响应
#   TestEchoProviderReturnsSystemPrompt   System 字段直接回显
#   TestOpenAIProviderUnauthorizedNotRetried  401 不重试
```

## Upstream Source Reading / 上游源码阅读

上游 `lightrag/llm/openai.py` 里的 `openai_complete_if_cache` 是 LLM 调用的主入口，外面包了一层 tenacity retry 装饰器。下面这段是核心 30 行（详细注解见 [`upstream-readings/s02-openai.py`](../../upstream-readings/s02-openai.py)）：

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
    # 创建 AsyncOpenAI client（带 base_url / api_key / timeout 覆盖）
    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, timeout=timeout, ...
    )
    # 拼 messages: [system?] + history + [user]
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # 真正的 HTTP 调用（被 tenacity 包了 3 次重试 + exponential backoff）
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response.choices[0].message.content
```

**对照阅读要点**：

- **`@retry` 装饰器 vs. s02 的循环**：上游用 tenacity 把整个函数包起来；s02 用 `for attempt := 0; attempt <= p.MaxRetries` 显式循环。Go 没有装饰器但 stdlib 退避循环就 10 行，可读性反而更好。
- **`stop_after_attempt(3)` = 3 次总尝试**：对应 s02 的 `WithMaxRetries(2)`（=1 初始 + 2 重试）。语义一致，命名口径不同——上游算总次数，s02 算「额外重试次数」。
- **`retry_if_exception_type(RateLimitError | APIConnectionError | APITimeoutError | InvalidResponseError)` 是分类逻辑**：等价于 s02 的 `transientStatusError` + `errors.As` 判定。上游靠 SDK 抛的具体异常类型；s02 没用 SDK，直接看 HTTP status code。
- **`messages = [system, history..., user]` 拼装顺序**：和 s02 的 `provider_openai.go` 里那段 `if req.System != "" { msgs = append(msgs, {Role:"system",...}) }` + `for _, m := range req.Messages { ... }` 完全对齐。
- **`**kwargs` 反模式**：上游让 caller 通过 kwargs 塞 `temperature` / `max_tokens` 等等；Go 里我们用结构体显式字段（`CompleteRequest.Temperature` / `MaxTokens`）。这是「类型替代 dict」的典型例子。

**想读更多**：上游 `lightrag/llm/openai.py` 里还有 `openai_embed`（s06 嵌入章节会读它）、`create_openai_async_client`（baseURL/Azure 切换的载体）、`gpt_4o_mini_complete` 这个 thin wrapper（在 examples 里被当 `llm_model_func` 注入）。Phase G 的多模型指南会把同样形态对应到 Anthropic / Bedrock / Ollama。

---

**下一节预告**：s03 把 `DocStatus` 状态机从 s01 占位的字符串常量推进成有持久化、转移校验、错误信息保留的 `DocStatusStore`。整个状态机配合 s05 的 KVStore 持久化能让「插入到一半挂了，重启后从 PROCESSING 接着跑」成为可能——这是 LightRAG 的 resume-on-failure 卖点。
