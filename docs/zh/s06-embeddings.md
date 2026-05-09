---
title: "s06 · 嵌入提供方与批处理"
chapter: 6
slug: s06-embeddings
est_read_min: 8
---

# s06 · 嵌入提供方与批处理

> 教什么：把 s01 那个「一次嵌一个 chunk」的占位实现拆成正式章节——`OpenAIEmbedder` 一次最多发 128 条到 `/embeddings`，对 429/5xx 做指数退避重试，`Dim()` 暴露向量维度供下游 s07 校验，新增 `MockEmbedder` 给 CI 用——把上游 `lightrag/llm/openai.py:openai_embed` 那个被 `@retry + @wrap_embedding_func_with_attrs` 装饰的函数翻成 Go。

---

## Problem / 问题

s01 的 `embedder.go` 大致长这样：

```go
for _, chunk := range chunks {
    vec, err := embedder.Embed(ctx, []string{chunk.Content})
    // ...
}
```

一次只送一条，每条要走一次完整的 HTTPS 握手 + JSON 编码 + 等回包。三件事崩坏：

1. **HTTP 开销吃掉吞吐**：单次嵌入调用网络 RTT 大概 100-300ms，OpenAI 的 `/v1/embeddings` 端点本身允许 ~2048 条 input 一起发，单批处理时间也只有几百毫秒。s01 嵌 100 个 chunk 串行走是 ~30s，而批量走 1 次只要 ~1s——**100× 的差距**直接定生死，没人愿意等半分钟才能 query 一次。
2. **没有重试**：429（rate limit）和 5xx（服务侧抖动）是日常事件，OpenAI 文档里明写了「请用指数退避」。s01 收到 429 就直接 `return err`，整条 ingest 流水线被一个临时性错误打断，doc-status 里那个 PROCESSING 永远卡在那不进 PROCESSED。
3. **没有维度内省**：s07 的 `VectorStore` 创建时要知道向量维度（`text-embedding-3-small` 是 1536，`text-embedding-3-large` 是 3072），s01 是把 1536 硬编死了——换模型就崩。
4. **测试只能联网**：s01 没有 `MockEmbedder`，CI 跑测试要么塞 OPENAI_API_KEY（成本+泄漏风险）要么 skip，没有第三种选项。

s06 一次性把这四件事补齐，对照 `lightrag/llm/openai.py:733-895` 的 `openai_embed` 把上游的「`@retry(stop_after_attempt(3), wait_exponential)` + 一次发一个 list 给 `embeddings.create`」搬到 Go 里。

## Solution / 解决方案

四件事：

1. **`EmbeddingProvider` 接口冻结自 plan.md**：只有两个方法，`Embed(ctx, texts) ([][]float32, error)` 和 `Dim() int`。s06 给两个实现（OpenAI 真接 + Mock 决定性 hash），后续 Phase G 加 Anthropic / Bedrock embedder 也是新文件，签名不动。
2. **`OpenAIEmbedder` 一次最多发 `BatchSize=128` 条**：调用方不用关心切片边界，`Embed(texts)` 内部按 `BatchSize` 切块，对每一块发一次 POST `/embeddings`，把每块返回的 `data[].embedding` 按下标拼回原顺序——所以 caller 给 `["a", "b", "c"]` 拿到的就是 `[vec_a, vec_b, vec_c]`，OpenAI 那侧返回顺序的细节被封装。
3. **`withRetry(maxRetries, fn)` 是私有助手**：~30 行，跟 s02 的 `OpenAIProvider.Complete` 的重试模式一致——指数退避 200ms / 600ms / 1.8s 加 ±20% 抖动，只对 429/5xx 触发，401 之类立即冒泡。s06 不能 import s02，所以这段是重新实现而不是引用。
4. **`MockEmbedder` 是决定性 hash → unit vector**：`sha256(text)` 拿到 32 字节，循环铺满 1536 维（每字节映射到 `[-1, 1]`），最后做 L2 归一化让 `sum(v[i]**2) == 1`。同样输入永远拿到字节相同的向量——CI 里的 `TestMockEmbedderDeterministic` 就靠这个。

## How It Works / 工作原理

```
Embed(ctx, ["t1", "t2", ..., "t25"])  with BatchSize=10
        │
        ├─ slice into 3 batches:  [t1..t10] [t11..t20] [t21..t25]
        │
        ▼
   for each batch:
     ┌──────────────────────────────────────────────┐
     │ POST <BaseURL>/embeddings                    │
     │   { model, input: [t_i..t_j], encoding_format} │
     │                                              │
     │   ┌─────────── retry loop ──────────┐        │
     │   │ attempt 0  → 429 → wait 200ms   │        │
     │   │ attempt 1  → 429 → wait 600ms   │        │
     │   │ attempt 2  → 200 OK            │        │
     │   └─────────────────────────────────┘        │
     │                                              │
     │ parse response.data[].embedding              │
     │ append to result in input order              │
     └──────────────────────────────────────────────┘
        │
        ▼
   [][]float32  (len == 25, dim == 1536)
```

载货核心 30 行（节选自 [`agents/s06-embeddings/embedder_openai.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s06-embeddings/embedder_openai.go) 的 `Embed`）：

```go
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    if len(texts) == 0 {
        return [][]float32{}, nil // 空切片：直接返回，不打网络。
    }
    out := make([][]float32, 0, len(texts))
    for start := 0; start < len(texts); start += e.BatchSize {
        end := start + e.BatchSize
        if end > len(texts) {
            end = len(texts)
        }
        var batchVecs [][]float32
        err := withRetry(e.MaxRetries, func() error {
            v, err := e.embedOnce(ctx, texts[start:end])
            if err != nil {
                return err
            }
            batchVecs = v
            return nil
        })
        if err != nil {
            return nil, err
        }
        out = append(out, batchVecs...)
    }
    return out, nil
}
```

**4 个不显眼但关键的设计点：**

1. **`out` 的顺序由分片顺序决定**：每批返回后立即 `append`，OpenAI 同一批内的顺序由 API 保证（`data[i]` 对应 `input[i]`）。所以 caller 的 `["a", "b", ..., "x"]` 永远能用相同下标拿到对应向量——这是 s07 的 `VectorStore.Upsert` 能正确把 chunkID → vector 关联的前提。
2. **空切片走零网络路径**：测试 `TestEmbedderHandlesEmptyInputSlice` 断言 `Embed(nil)` 不发任何 HTTP 请求且不报错。这件事在 ingest 流水线里很常见——某个 doc 只有标题没正文，chunking 给出 0 个 chunk，下游不该因此打 OpenAI 的 quota。
3. **`MaxRetries=3` 等于「初次 + 2 次重试 = 总共 3 次尝试」**：和 s02 的语义对齐（s02 那边是 `MaxRetries=2` → 3 次尝试，s06 选 `MaxRetries=3` 是因为嵌入这一步是 ingest 关键路径，多重试一轮影响不大）。`TestEmbedderRetriesOn429` 断言 429×2 + 200×1 后总尝试数为 3。
4. **`MockEmbedder` 的 sha256 → 1536 维向量**：sha256 只产生 32 字节（256 维），1536 维要把 32 字节循环铺 48 次。每个字节 `b` 映射成 `(float32(b) - 127.5) / 127.5`，落在 `[-1, 1]`；最后整个向量 L2 归一化。这样既保证「相同输入 → 字节相同向量」，又让向量是单位长度，方便 s07 测试余弦相似度（unit vector 的内积就是 cosine）。

## What Changed / 与 s01 的变化

| 维度 | s01（占位） | s06（正式章节） |
|---|---|---|
| 接口位置 | `pipeline.go` 内联 | 独立 `embedder.go` |
| 批处理 | for 循环单条调用 | `BatchSize=128`，一次 HTTP 多条 |
| 重试 | 无 | 429/5xx 指数退避 ×3 次 |
| 维度内省 | 硬编码 1536 | `Dim()` 方法返回 |
| 测试替身 | 无 mock | `MockEmbedder` 决定性 hash → unit vector |
| 函数式选项 | 巨构造器 | `WithModel/WithBatchSize/WithMaxRetries` |
| 空切片处理 | 未定义 | 显式返回 `[][]float32{}`，零网络 |

吞吐改善的「数学」：100 个 chunk 在 RTT=200ms 的网络上，s01 串行是 100×200ms = **20s**；s06 用 BatchSize=128 是 1×200ms（1 批） + payload 略大但仍是单次 RTT 主导 = **~250ms**。**~80× 实测加速**（demo 里 `main.go` 把这个数字打出来给学习者看）。

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s06-embeddings

# 1. 默认 mock embedder，CI 跑这个 / fully offline
go run . -provider mock

# 2. 真 OpenAI（需 OPENAI_API_KEY）
export OPENAI_API_KEY=sk-...
go run . -provider openai -model text-embedding-3-small -batch 128

# 3. 改批大小看吞吐变化（mock 不打网络但仍按 batch 切片，方便测试逻辑）
go run . -provider mock -batch 32

# 4. 跑全部 7 个测试
go test -v ./...
```

CLI 输出里有四个数字直接照在脸上：

```
=== s06 embedder demo ===
provider: mock
model:    mock
batch:    128
texts:    47
dim:      1536
elapsed:  3.2ms
throughput: 14687.5 lines/sec
```

测试矩阵（全部不打外网，OpenAI 路径靠 `httptest.NewServer`）：

| 测试 | 断言 |
|---|---|
| `TestEmbedderInterfaceContract` | 编译期保证两个实现都满足 `EmbeddingProvider` |
| `TestEmbedderBatchesAtLimit` | `BatchSize=10` 送 25 条 → 服务端收到正好 3 个 HTTP 请求 |
| `TestEmbedderDimMatchesProvider` | OpenAI 和 Mock 的 `Dim()` 都返回 1536 |
| `TestEmbedderHandlesEmptyInputSlice` | `Embed(nil)` 返回 `[][]float32{}`，0 网络调用 |
| `TestEmbedderRetriesOn429` | 429 → 429 → 200，断言尝试 3 次后拿到结果 |
| `TestMockEmbedderDeterministic` | 同样输入两次 `Embed` → 字节相同向量 |
| `TestMockEmbedderUnitNormalized` | 任意输入 `sum(v[i]*v[i])` ≈ 1.0 ± 1e-5 |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/llm/openai.py:733-895 (节选 ~50 行；原函数还有 ~50 行 Azure / 截断 / token tracker 逻辑略)

@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=8192,
    model_name="text-embedding-3-small",
    supports_asymmetric=True,
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    ...
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's API ..."""
    # ... (略：context prefix / max_token_size 截断)

    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, ...,
    )

    async with openai_async_client:
        api_model = azure_deployment if use_azure and azure_deployment else model
        api_params = {
            "model": api_model,
            "input": texts,
        }
        api_params["encoding_format"] = "base64" if EMBEDDING_USE_BASE64 else "float"
        if embedding_dim is not None:
            api_params["dimensions"] = embedding_dim

        response = await openai_async_client.embeddings.create(**api_params)

        return np.array([
            np.array(dp.embedding, dtype=np.float32)
            if isinstance(dp.embedding, list)
            else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
            for dp in response.data
        ])
```

**怎么看**：

- 上面那两个装饰器 `@wrap_embedding_func_with_attrs` 和 `@retry` 是上游的「门面」——前者把 `embedding_dim=1536` 这种元数据塞到函数对象上（这就是 `Dim()` 在 Go 里要做的事），后者是 `tenacity` 的指数退避（对应 `withRetry` 那 ~30 行）。
- 真正发请求是 `openai_async_client.embeddings.create(model=..., input=[texts])`——`input` 字段是 list，单次调用就处理多条，这就是「批处理」的来源。
- `response.data[i]` 严格对应 `input[i]`，所以拿出来直接 `[dp.embedding for dp in response.data]` 就保留了输入顺序——Go 实现里那个 `out = append(out, batchVecs...)` 逻辑等价。
- 上游兼顾 `base64` 和 `list`（`encoding_format=base64` 时回包是 base64 编码的 float32 数组，省带宽 25%），s06 走最简单的 `float` 路径以保 stdlib only——在「上游讨论但 s06 略过」一栏。

更完整的注解版（含 reading-map）：[`upstream-readings/s06-embeddings.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s06-embeddings.py)。

下一节 [s07 余弦相似向量库](s07-vector-store.md) 是这些向量的消费者：把 `[][]float32` 喂给三个独立索引（chunks / entities / relations），每个走 cosine ranking。Phase G 的 Anthropic / Bedrock embedding 实现走的是同一个 `EmbeddingProvider` 接口——s06 不变。
