# s06 · 嵌入提供方与批处理 / Embedding provider with batching

> 把 s01 内联的「一次嵌一条」推进成正式章节：批处理 + 重试 + Dim() 内省 + Mock 决定性向量。
> Promote s01's inline "one chunk per call" embedder into a real chapter: batching + retry + Dim() introspection + deterministic Mock.

## 跑起来 / Run

```bash
cd agents/s06-embeddings

# 1. 默认 mock embedder，CI 跑这个 / fully offline (what CI runs)
go run . -provider mock

# 2. 真 OpenAI（需 OPENAI_API_KEY） / real OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
go run . -provider openai -model text-embedding-3-small -batch 128

# 3. 改批大小看 batching 行为 / tweak batch size
go run . -provider mock -batch 32 -manifest=false

# 测试 / tests
go test -v ./...
```

## 文件 / Files

| 文件 / file | 作用 / role | LOC |
|---|---|---|
| `embedder.go` | `EmbeddingProvider` 接口 + 函数式选项 + `withRetry` 助手 / interface + options + retry helper | ~135 |
| `embedder_openai.go` | 批处理 OpenAI 实现 / batched OpenAI impl | ~190 |
| `embedder_mock.go` | sha256 决定性单位向量 / deterministic-hash unit vector | ~80 |
| `main.go` | CLI demo: `-provider {openai\|mock}`, 报告吞吐 / CLI with throughput report | ~120 |
| `embedder_test.go` | 7 个测试，全部 `httptest`，零外网 / 7 tests, all `httptest`, no external network | ~210 |
| `testdata/sample.txt` | 35 行合成英文短句（Eleanor 灯塔故事） / 35 lines of synthetic English (Eleanor lighthouse story) | — |

## 关键教学点 / Key teaching points

- **接口形态自 plan.md 起冻结 / Interface frozen since plan.md**：`EmbeddingProvider` 两个方法不变；s06 只是把 OpenAI 实现拆出来 + 加 Mock。Phase G 加 Anthropic / Bedrock embedder 也是新文件，签名不动。
- **批处理是关键速率改善 / Batching is the load-bearing speedup**：`BatchSize=128` 让 100 个 chunk 从 100 次 RTT 降到 1 次（~80× 实测加速，demo 里打出来）。
- **`withRetry` ≈ 30 行 / `withRetry` ≈ 30 LOC**：和 s02 的指数退避完全一致——只对 429/5xx 触发，401 立即冒泡。s06 不能 import s02，所以是重新实现。
- **`MockEmbedder` 决定性 / Deterministic mock**：sha256(text) 铺成 1536 维，L2 归一化。同样输入 → 字节相同向量；任意输入 → ‖v‖² ≈ 1.0。这两个性质让 s07 的测试可以 exact-equal 比较向量。
- **空切片走零网络路径 / Empty slice = zero network**：`Embed(nil)` 返回 `[][]float32{}`，不打 OpenAI。doc-status pipeline 里的零 chunk doc 由此免单。

完整章节文档：[docs/zh/s06-embeddings.md](../../docs/zh/s06-embeddings.md) · [docs/en/s06-embeddings.md](../../docs/en/s06-embeddings.md)
