# s02 · 提供方接口 / Provider interface

> 把 s01 内联的 OpenAI HTTP 调用抽成正式的 Provider 章节：functional options + 重试 + 三种实现（OpenAI / Mock / Echo）。
> Promote s01's inline OpenAI HTTP call into a real Provider chapter: functional options + retry + three impls (OpenAI / Mock / Echo).

## 跑起来 / Run

```bash
cd agents/s02-provider

# 1. 全离线 mock，CI 跑这个 / fully offline mock, what CI runs
go run . -provider mock -q "What's the capital of France?"

# 2. echo provider — 直接把 system prompt 回显，最小烟测
# echo provider — system prompt round-trips verbatim, smallest smoke test
go run . -provider echo -q "(ignored — echo only uses System)"

# 3. 真 OpenAI（需要 OPENAI_API_KEY） / real OpenAI (needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
go run . -provider openai -model gpt-4o-mini -v -q "What's the capital of France?"

# 测试 / tests
go test -v ./...
```

## 文件 / Files

| 文件 / file | 作用 / role | LOC |
|---|---|---|
| `provider.go` | `Provider` 接口 + `CompleteRequest/Response` + 函数式选项 / `Provider` interface + types + functional options | ~115 |
| `provider_openai.go` | OpenAI 实现，含 429/5xx 指数退避重试 / OpenAI impl with 429/5xx exponential-backoff retry | ~210 |
| `provider_mock.go` | sha256 决定性回显 + 原子计数器 / sha256-deterministic echo + atomic counter | ~75 |
| `provider_echo.go` | 把 system prompt 原样返回 / returns the system prompt verbatim | ~30 |
| `main.go` | CLI demo：`-provider {openai\|mock\|echo}` / CLI demo across all 3 impls | ~85 |
| `provider_test.go` | 7 个测试，全用 `httptest`，不打外网 / 7 tests, all `httptest`, no external network | ~210 |

## 关键教学点 / Key teaching points

- **接口形态从 s01 起冻结 / Interface frozen since s01**：`Provider` / `CompleteRequest` / `CompleteResponse` 字段不变，s02 只是扩展实现。Phase G 加 Anthropic / Bedrock / Ollama 也是新增文件，签名不动。
- **函数式选项 vs. 巨大构造器 / Functional options vs. fat ctor**：`NewOpenAIProvider(WithModel(...), WithTimeout(...), WithMaxRetries(...))` 让后续每加一个旋钮都是一个 `WithFoo()`，调用方 0 改动。
- **重试只在 429/5xx 触发 / Retry only on 429/5xx**：`transientStatusError` 把瞬时和永久错误区分开——401 / 解析错误立刻返回，不浪费时间。指数退避 200ms→600ms→1.8s + 抖动。
- **`MockProvider` 决定性 / Deterministic mock**：默认响应是 sha256(请求) 的哈希片段，两次相同请求返回字节相同——这是 CI 能 assert 的载货性质。
- **`EchoProvider` 是最小 Provider / Echo is the smallest Provider**：5 行实现，演示 Provider 接口的最小满足者；用于「caller 拼对 system prompt 了吗」类型的烟测。

完整章节文档：[docs/zh/s02-provider.md](../../docs/zh/s02-provider.md) · [docs/en/s02-provider.md](../../docs/en/s02-provider.md)
