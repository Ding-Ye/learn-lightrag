---
title: "s06 · Embedding provider with batching"
chapter: 6
slug: s06-embeddings
est_read_min: 8
---

# s06 · Embedding provider with batching

> What this teaches: promote s01's "embed one chunk at a time" placeholder into a real chapter — `OpenAIEmbedder` posts up to 128 inputs to `/embeddings` in a single HTTP call, retries 429/5xx with exponential backoff, exposes `Dim()` so s07 can size its index from the model, and ships a `MockEmbedder` for CI. The Go translation tracks `lightrag/llm/openai.py:openai_embed` — the function decorated by `@retry + @wrap_embedding_func_with_attrs`.

---

## Problem / 问题

s01's `embedder.go` looked like this:

```go
for _, chunk := range chunks {
    vec, err := embedder.Embed(ctx, []string{chunk.Content})
    // ...
}
```

One text per call, one full HTTPS handshake + JSON encode + RTT per chunk. Three things break:

1. **HTTP overhead destroys throughput.** Each embedding call costs ~100-300ms of network RTT. OpenAI's `/v1/embeddings` accepts up to ~2048 inputs in one batch, and a batched call still finishes in a few hundred milliseconds. Embedding a 100-chunk doc serially is ~30s; one batched call is ~1s. **A 100× gap** is the difference between "I'll wait" and "I'll go make coffee" — no learner sticks around for the latter.
2. **No retries.** 429 (rate limit) and 5xx (provider hiccup) are routine. OpenAI's docs explicitly tell you to use exponential backoff. s01 surfaces the first 429 as a hard error, freezing the doc-status state machine in PROCESSING with no way out.
3. **No dimension introspection.** s07's `VectorStore` needs the embedding dim at construction time (1536 for `text-embedding-3-small`, 3072 for `text-embedding-3-large`). s01 hard-codes 1536; switching models silently corrupts the index.
4. **Tests can't run offline.** Without a `MockEmbedder`, CI either ships an `OPENAI_API_KEY` (cost + leak risk) or skips embedding tests entirely.

s06 fixes all four by porting `lightrag/llm/openai.py:733-895` — the `openai_embed` function decorated with `@retry(stop_after_attempt(3), wait_exponential)` plus `@wrap_embedding_func_with_attrs(embedding_dim=1536, ...)`.

## Solution / 解决方案

Four moves:

1. **`EmbeddingProvider` interface frozen from plan.md.** Two methods: `Embed(ctx, texts) ([][]float32, error)` and `Dim() int`. s06 ships two impls (real OpenAI + deterministic Mock). Phase G's Anthropic / Bedrock embedders are new files with no signature changes.
2. **`OpenAIEmbedder` posts up to `BatchSize=128` inputs per HTTP call.** The caller doesn't track batch boundaries — `Embed(texts)` slices internally, POSTs each batch to `/embeddings`, and reassembles `data[].embedding` in input order. Hand it `["a", "b", "c"]` and you get `[vec_a, vec_b, vec_c]` regardless of how OpenAI orders its response chunks.
3. **`withRetry(maxRetries, fn)` is a private helper, ~30 LOC.** Same exponential-backoff shape as s02's `OpenAIProvider.Complete`: 200ms → 600ms → 1.8s with ±20% jitter, fires only on 429/5xx (401 surfaces immediately). s06 cannot import s02 — sessions are isolated — so the helper is re-implemented from scratch.
4. **`MockEmbedder` is a deterministic hash → unit vector.** `sha256(text)` gives 32 bytes; tile them across 1536 dims (each byte mapped to `[-1, 1]`) and L2-normalize so `sum(v[i]**2) == 1`. Same input always returns byte-identical vectors — `TestMockEmbedderDeterministic` asserts that. The unit-norm property makes cosine equal dot product, simplifying s07's tests.

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

Load-bearing 30 lines (excerpted from [`agents/s06-embeddings/embedder_openai.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s06-embeddings/embedder_openai.go), `Embed`):

```go
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    if len(texts) == 0 {
        return [][]float32{}, nil // empty input: zero network, no error.
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

**Four non-obvious points:**

1. **`out` ordering is determined by slice order.** Each batch's vectors append immediately upon return, and within a batch OpenAI guarantees `data[i]` corresponds to `input[i]`. So caller indices into `texts[]` and `out[]` align perfectly — that property is what lets s07's `VectorStore.Upsert(chunkID, vector)` zip the two slices safely.
2. **Empty slice takes a zero-network fast path.** `TestEmbedderHandlesEmptyInputSlice` asserts `Embed(nil)` makes no HTTP call and returns no error. That matters in the ingest pipeline: a doc with only a title produces zero chunks, and we shouldn't burn OpenAI quota on an empty list.
3. **`MaxRetries=3` means "1 initial + up to 2 retries = 3 total attempts."** This matches the `tenacity stop_after_attempt(3)` semantic upstream uses. `TestEmbedderRetriesOn429` asserts the server sees exactly 3 calls when it returns 429 twice then 200.
4. **`MockEmbedder` packs sha256 into a 1536-d unit vector.** sha256 emits 32 bytes (256 bits). To fill 1536 dims we tile the digest 48 times, mapping each byte `b` to `(float32(b) - 127.5) / 127.5` ∈ `[-1, 1]`, then L2-normalize the whole vector. This guarantees both determinism (same input → same bytes → same digest → same vector) and unit length (so cosine similarity reduces to dot product, making s07 tests cleaner).

## What Changed / 与 s01 的变化

| Dimension | s01 (placeholder) | s06 (real chapter) |
|---|---|---|
| Interface location | inline in `pipeline.go` | dedicated `embedder.go` |
| Batching | for-loop, one text per call | `BatchSize=128`, many texts per HTTP call |
| Retry | none | 429/5xx exponential backoff ×3 attempts |
| Dim introspection | hard-coded 1536 | `Dim()` method |
| Test double | none | `MockEmbedder` deterministic-hash → unit vector |
| Constructor | fat ctor | `WithModel/WithBatchSize/WithMaxRetries` functional options |
| Empty input | undefined | explicit `[][]float32{}`, zero network |

The throughput math: 100 chunks at 200ms RTT — s01 serial = 100×200ms = **20s**; s06 with BatchSize=128 = 1×200ms (1 batch) + slightly larger payload but still RTT-dominated = **~250ms**. **~80× speedup empirically** (the demo's `main.go` prints throughput in lines/sec to make this visible).

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s06-embeddings

# 1. Default mock embedder, what CI runs / 全离线
go run . -provider mock

# 2. Real OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
go run . -provider openai -model text-embedding-3-small -batch 128

# 3. Tweak batch size to see slicing behavior (mock makes no HTTP calls but still slices)
go run . -provider mock -batch 32

# 4. Run all 7 tests
go test -v ./...
```

The CLI surfaces four numbers right in the output:

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

Test matrix (no external network; OpenAI paths use `httptest.NewServer`):

| Test | Asserts |
|---|---|
| `TestEmbedderInterfaceContract` | compile-time guard: both impls satisfy `EmbeddingProvider` |
| `TestEmbedderBatchesAtLimit` | `BatchSize=10` + 25 texts → server sees exactly 3 HTTP calls |
| `TestEmbedderDimMatchesProvider` | OpenAI and Mock both return `Dim()==1536` |
| `TestEmbedderHandlesEmptyInputSlice` | `Embed(nil)` returns `[][]float32{}` with zero network calls |
| `TestEmbedderRetriesOn429` | 429 → 429 → 200, server sees 3 calls, final result correct |
| `TestMockEmbedderDeterministic` | same input embedded twice → byte-identical vectors |
| `TestMockEmbedderUnitNormalized` | any text yields `sum(v[i]*v[i])` ≈ 1.0 within 1e-5 |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/llm/openai.py:733-895 (excerpt ~50 lines; the full function adds Azure / truncation / token tracker)

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
    # ... (omitted: context prefixes / max_token_size truncation)

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

**How to read it:**

- The two decorators `@wrap_embedding_func_with_attrs` and `@retry` are the upstream "facade." The first attaches `embedding_dim=1536` metadata onto the function object (this is exactly what `Dim()` does in our Go port). The second is `tenacity`-driven exponential backoff (corresponds to our ~30 LOC `withRetry`).
- The actual API call is `openai_async_client.embeddings.create(model=..., input=[texts])`. The `input` is a list — that's where batching comes from. One HTTP round-trip handles many texts.
- `response.data[i]` strictly corresponds to `input[i]`, so a list comprehension `[dp.embedding for dp in response.data]` preserves caller order. Our Go `out = append(out, batchVecs...)` is the equivalent.
- Upstream supports both `base64` and `list` encoding (`encoding_format=base64` reduces wire size by ~25% by transmitting float32 as base64 instead of JSON numbers); s06 takes the simpler `float` path to stay stdlib-only — discussed but not implemented.

A more annotated version with reading-map: [`upstream-readings/s06-embeddings.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s06-embeddings.py).

The next chapter, [s07 cosine-similarity vector store](s07-vector-store.md), is the consumer of these vectors — it feeds `[][]float32` into three independent indices (chunks / entities / relations) using cosine ranking. Phase G's Anthropic / Bedrock embedders satisfy the same `EmbeddingProvider` interface; s06 doesn't change.
