---
title: "多模型接入指南"
slug: multi-model
est_read_min: 9
---

# 多模型接入指南

> learn-lightrag 默认用 **OpenAI** 跑 demo，但 s02 和 s06 的 Provider/EmbeddingProvider 都设计成接收 `WithBaseURL(...)`——这意味着任何 **OpenAI-compatible** 端点（DeepSeek、Moonshot/Kimi、Qwen、Groq、OpenRouter、自托管 vLLM/SGLang…）都是一行配置切换的事。

---

## 设计思路

LightRAG 上游用一个 `llm_model_func: Callable` DI seam 适配所有 LLM 后端。我们用 Go interface `Provider`（s02）+ `EmbeddingProvider`（s06）做同一件事——**OpenAI 协议是接入面的最大公分母**：

- DeepSeek / Moonshot / Qwen / Groq / 阿里 DashScope / OpenRouter 都直接说"OpenAI compatible"，意味着 `/v1/chat/completions` 和 `/v1/embeddings` 路径 + 同样的 JSON schema。
- 自托管 vLLM / SGLang / LM Studio 默认就是 OpenAI 协议。
- 真正"原生协议"的只有 Anthropic（`/v1/messages`）；如果非要用 Claude，可以走 OpenRouter 这种代理。

所以对 learn-lightrag 而言：**8 个 profile 共用一个 `OpenAIProvider`，只换 `BaseURL`、`Model`、`APIKey` 三件套**。

---

## 8 个 Provider Profile

| profile | endpoint | 推荐 model | env var |
|---|---|---|---|
| `openai` | https://api.openai.com/v1 | gpt-4o-mini | `OPENAI_API_KEY` |
| `deepseek` | https://api.deepseek.com/v1 | deepseek-chat（也可 deepseek-reasoner） | `DEEPSEEK_API_KEY` |
| `moonshot` | https://api.moonshot.cn/v1 | moonshot-v1-8k / moonshot-v1-32k | `MOONSHOT_API_KEY` |
| `qwen` | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus / qwen-max | `DASHSCOPE_API_KEY` |
| `groq` | https://api.groq.com/openai/v1 | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| `openrouter` | https://openrouter.ai/api/v1 | openai/gpt-4o-mini（或 anthropic/claude-sonnet-4） | `OPENROUTER_API_KEY` |
| `local` | http://localhost:8000/v1 | 由 vLLM/SGLang 自决定 | `OPENAI_API_KEY`（任意非空） |
| `anthropic` | api.anthropic.com（**非 OpenAI 兼容**） | claude-sonnet-4-x | `ANTHROPIC_API_KEY` |

> ⚠️ 前 7 个都是 OpenAI-compatible，可直接复用 s02 的 `OpenAIProvider`。第 8 个 Anthropic 走原生 `/v1/messages`，需要单独写一个 `AnthropicProvider`（约 80 行，附录练习；或者绕道 OpenRouter）。

### Embedding profiles

embedding 端点更挑——不是所有 OpenAI 兼容的 chat 提供商都有 embedding 路径。可用组合：

| profile | endpoint | 推荐 embedding model | 维度 |
|---|---|---|---|
| `openai` | api.openai.com/v1 | text-embedding-3-small | 1536 |
| `qwen` | dashscope.aliyuncs.com/compatible-mode/v1 | text-embedding-v3 | 1024 |
| `local` | localhost:8000/v1 | 自决（如 bge-large-en-v1.5） | 模型决定 |
| `voyage` | api.voyageai.com/v1 | voyage-3 | 1024 |

实际项目通常 chat 用一家、embedding 用另一家——s06 的 `EmbeddingProvider` 接口允许任意组合。

---

## 实战：怎么用

### s02 chat completion 换 DeepSeek

```bash
cd agents/s02-provider
export DEEPSEEK_API_KEY=sk-...
go run . -provider mock                 # CI 模式（不变）
```

代码端切换（在你的 demo / pipeline 里）：

```go
import "learn-lightrag/s02"  // 假设你的项目引用 s02

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

**1 行差别**——`BaseURL` 换。

### s06 embedding 换 Qwen / DashScope

```go
emb := NewOpenAIEmbedder(
    WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
    WithModel("text-embedding-v3"),
    WithAPIKey(os.Getenv("DASHSCOPE_API_KEY")),
    WithBatchSize(10),  // DashScope 限 10 batch
)
vectors, err := emb.Embed(ctx, []string{"hello", "world"})
```

注意 DashScope 的 batch 上限是 10（OpenAI 是 2048）；改 `WithBatchSize(10)` 即可。

### s11 完整 query 跑 OpenRouter Claude

```go
prov := NewOpenAIProvider(
    WithBaseURL("https://openrouter.ai/api/v1"),
    WithModel("anthropic/claude-sonnet-4"),
    WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
)
// 后续 s11 的 Pipeline 用法完全不变。
```

通过 OpenRouter 你能拿到 Claude 而不需要写 Anthropic 适配器——OpenRouter 帮你做了协议翻译。

---

## 主流后端的小坑

| 后端 | 坑 |
|---|---|
| DeepSeek | 偶尔返回 `content` 是 array 而非 string；s02 的解析层假设 string，需要改成 `interface{}` 后判类型。`reasoner` 模型会先返 thinking tokens，需要跳过。 |
| Moonshot | 默认 8K context，长 doc 用 `moonshot-v1-32k` 或 `-128k`。 |
| Qwen / DashScope | embed batch 上限 10；模型名要前缀 `qwen-`；中文混合时 chunk size 用 char 而非 token 通常更准。 |
| Groq | 免费层 RPM 限制严格（默认 30/min），适合做小规模 demo 不适合 ingest 大文档。 |
| OpenRouter | 部分模型不支持 tools——但本课程的 RAG 不用 tools，无影响。 |
| 自托管 vLLM | 需要启动时加 `--enable-auto-tool-choice` 才支持 tools；同样 RAG 路径不需要。 |
| Anthropic 原生 | wire format 完全不同，需要额外一个 80 行的 `AnthropicProvider`。最小例子见下方。 |

---

## 写一个 AnthropicProvider（可选练习）

如果你坚持要走 Claude 原生协议（不通过 OpenRouter），约 80 行：

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

**关键差异**：Anthropic 的 `system` 是顶层字段（不是消息）；`content` 是 array of blocks（不是 string）；`max_tokens` 是必填。

---

## 一句话决策

- **国内用户**：默认 DeepSeek（chat）+ Qwen embed (DashScope) 组合，性价比最高。
- **海外个人**：默认 OpenAI gpt-4o-mini + text-embedding-3-small，文档最齐。
- **离线 / 隐私**：本地 vLLM + 嵌入模型（如 bge-large-en-v1.5）。
- **想用 Claude**：走 OpenRouter，省得写适配器。

不管选哪个——s02、s06、s09、s11 的代码一行不用改。这是 `Provider` interface 的真正价值。
