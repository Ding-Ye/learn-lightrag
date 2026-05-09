---
title: "Multi-model integration guide"
slug: multi-model
est_read_min: 9
---

# Multi-model integration guide

> learn-lightrag uses **OpenAI** as the demo default, but s02's `Provider` and s06's `EmbeddingProvider` both accept `WithBaseURL(...)` — meaning any **OpenAI-compatible** endpoint (DeepSeek, Moonshot/Kimi, Qwen, Groq, OpenRouter, self-hosted vLLM/SGLang...) is a one-line config swap.

---

## Design

Upstream LightRAG uses a `llm_model_func: Callable` DI seam to adapt every LLM backend. We do the same with Go interfaces — `Provider` (s02) and `EmbeddingProvider` (s06). And **OpenAI's HTTP shape is the lowest common denominator**:

- DeepSeek / Moonshot / Qwen / Groq / Alibaba DashScope / OpenRouter all advertise "OpenAI compatible", meaning `/v1/chat/completions` and `/v1/embeddings` paths with the same JSON schema.
- Self-hosted vLLM / SGLang / LM Studio default to OpenAI protocol.
- The only true "native protocol" is Anthropic's `/v1/messages`. To use Claude you can either write a small adapter or proxy through OpenRouter.

So for learn-lightrag: **8 profiles share one `OpenAIProvider`, only `BaseURL`, `Model`, `APIKey` change**.

---

## 8 Provider Profiles

| profile | endpoint | suggested model | env var |
|---|---|---|---|
| `openai` | https://api.openai.com/v1 | gpt-4o-mini | `OPENAI_API_KEY` |
| `deepseek` | https://api.deepseek.com/v1 | deepseek-chat (or deepseek-reasoner) | `DEEPSEEK_API_KEY` |
| `moonshot` | https://api.moonshot.cn/v1 | moonshot-v1-8k / moonshot-v1-32k | `MOONSHOT_API_KEY` |
| `qwen` | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus / qwen-max | `DASHSCOPE_API_KEY` |
| `groq` | https://api.groq.com/openai/v1 | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| `openrouter` | https://openrouter.ai/api/v1 | openai/gpt-4o-mini (or anthropic/claude-sonnet-4) | `OPENROUTER_API_KEY` |
| `local` | http://localhost:8000/v1 | (vLLM/SGLang decides) | `OPENAI_API_KEY` (any non-empty) |
| `anthropic` | api.anthropic.com (**NOT OpenAI compatible**) | claude-sonnet-4-x | `ANTHROPIC_API_KEY` |

> ⚠️ The first seven are OpenAI-compatible — reuse s02's `OpenAIProvider`. The 8th (Anthropic native) needs a separate `AnthropicProvider` (~80 LOC; covered as an exercise below) or you can route through OpenRouter.

### Embedding profiles

Embedding endpoints are pickier — not every OpenAI-compatible chat provider has an embeddings path. Workable combinations:

| profile | endpoint | suggested embedding model | dimension |
|---|---|---|---|
| `openai` | api.openai.com/v1 | text-embedding-3-small | 1536 |
| `qwen` | dashscope.aliyuncs.com/compatible-mode/v1 | text-embedding-v3 | 1024 |
| `local` | localhost:8000/v1 | (bge-large-en-v1.5 etc.) | depends on model |
| `voyage` | api.voyageai.com/v1 | voyage-3 | 1024 |

Real projects typically mix chat from one provider and embedding from another — s06's `EmbeddingProvider` allows any combination.

---

## Hands-on

### s02 chat completion → DeepSeek

```bash
cd agents/s02-provider
export DEEPSEEK_API_KEY=sk-...
go run . -provider mock                 # CI mode (unchanged)
```

Code-side swap (in your demo / pipeline):

```go
import "learn-lightrag/s02"  // assuming your project imports s02

prov := NewOpenAIProvider(
    WithBaseURL("https://api.deepseek.com/v1"),
    WithModel("deepseek-chat"),
    WithAPIKey(os.Getenv("DEEPSEEK_API_KEY")),
    WithMaxRetries(3),
)
resp, err := prov.Complete(ctx, CompleteRequest{
    System:   "You are a helpful assistant.",
    Messages: []Message{{Role: "user", Content: "hello"}},
})
```

**One-line diff** — only `BaseURL`.

### s06 embedding → Qwen / DashScope

```go
emb := NewOpenAIEmbedder(
    WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
    WithModel("text-embedding-v3"),
    WithAPIKey(os.Getenv("DASHSCOPE_API_KEY")),
    WithBatchSize(10),  // DashScope caps at 10
)
vectors, err := emb.Embed(ctx, []string{"hello", "world"})
```

DashScope's batch limit is 10 (OpenAI is 2048); just set `WithBatchSize(10)`.

### s11 full query against Claude via OpenRouter

```go
prov := NewOpenAIProvider(
    WithBaseURL("https://openrouter.ai/api/v1"),
    WithModel("anthropic/claude-sonnet-4"),
    WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
)
// All the rest of s11's Pipeline is unchanged.
```

OpenRouter gives you Claude without writing an Anthropic adapter — OpenRouter handles the protocol translation.

---

## Backend gotchas

| backend | gotcha |
|---|---|
| DeepSeek | Occasionally returns `content` as an array instead of a string; the s02 parser assumes string and needs an `interface{}` + type switch. The `reasoner` model emits thinking tokens first; skip those. |
| Moonshot | Default 8K context; for long docs use `moonshot-v1-32k` or `-128k`. |
| Qwen / DashScope | Embed batch limit 10; model names need the `qwen-` prefix; for mixed Chinese-English text, chunking by char often beats by token. |
| Groq | Strict RPM limits on the free tier (default 30/min); great for small demos, not for ingesting large documents. |
| OpenRouter | Some models don't support tools — but our RAG path doesn't use tools, so this doesn't matter. |
| Self-hosted vLLM | Needs `--enable-auto-tool-choice` for tools; again, our RAG path doesn't need this. |
| Anthropic native | Wire format is fundamentally different; needs an extra ~80-line `AnthropicProvider`. Sketch below. |

---

## Writing an AnthropicProvider (optional exercise)

If you insist on the native Claude protocol (not via OpenRouter), ~80 LOC:

```go
type AnthropicProvider struct {
    APIKey string
    Model  string
    HTTP   *http.Client
}

func (a *AnthropicProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
    // POST https://api.anthropic.com/v1/messages
    // Header: x-api-key: <key>, anthropic-version: 2023-06-01
    // Body shape:
    //   {
    //     "model": req.Model,
    //     "max_tokens": req.MaxTokens,
    //     "system": req.System,                 // top-level field, NOT a message
    //     "messages": [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
    //   }
    // Response: { "content": [{"type": "text", "text": "..."}], "usage": {"input_tokens": N, "output_tokens": M} }
    // ...
}
```

**Key differences**: Anthropic's `system` is a top-level field (not a message); `content` is an array of blocks (not a string); `max_tokens` is required.

---

## One-line decision

- **China-based users**: default to DeepSeek (chat) + Qwen embed (DashScope) — best price-perf locally.
- **Overseas individual**: default to OpenAI gpt-4o-mini + text-embedding-3-small — fullest documentation.
- **Offline / privacy**: local vLLM + an embedding model (e.g. bge-large-en-v1.5).
- **Want Claude**: route through OpenRouter, skip writing the adapter.

Whichever you pick — s02, s06, s09, s11 source code is **unchanged**. That's the real value of the `Provider` interface.
