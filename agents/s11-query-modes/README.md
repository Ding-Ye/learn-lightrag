# s11 — Dual-level retrieval, four modes / 双层检索四种模式

This is the chapter where LightRAG's signature dual-level retrieval finally
lights up. Sessions s01-s10 built the entire **ingestion** pipeline — chunks,
embeddings, KV, vectors, graph, extraction, summarization — but never used
the graph at query time. s11 closes the loop: four `QueryMode` strategies,
each compiling a context window inside a `MaxTotalTokens` budget, then
handing off to `Provider.Complete`.

s11 是「双层检索四种模式」一节，是整个仓库里 LightRAG 招牌算法真正点亮的地方。
前面十节做的全是入库；这一节才让图在查询时被使用。四种模式 — naive / local /
global / hybrid — 各自在 `MaxTotalTokens` 预算内组装上下文，再交给
`Provider.Complete` 合成最终答案。

## Run / 运行

```bash
cd agents/s11-query-modes
go run . -mode all -q "Who discovered the Hartwell Codex?"
```

The default `-mode all` runs every mode side by side so the comparison is
visible:

```
[naive]   references: [chunk-2 chunk-3 ...]   answer: ...
[local]   references: [chunk-1 chunk-2 ...]   answer: ...
[global]  references: [chunk-4 chunk-5 ...]   answer: ...
[hybrid]  references: [chunk-1 chunk-2 ...]   answer: ...
```

## Test / 测试

```bash
go test -count=1 -v ./...
```

Eight tests, all offline (MockProvider + MockEmbedder, no network):

| # | Test | What it asserts |
|---|---|---|
| 1 | `TestQueryNaiveReturnsTopChunks` | naive mode returns top-K chunks via `VDBChunks` |
| 2 | `TestQueryLocalUsesLowLevelKeywords` | local mode passes LL keyword embeddings to **VDBEntities** (not VDBRelations) |
| 3 | `TestQueryGlobalUsesHighLevelKeywords` | global mode passes HL keyword embeddings to **VDBRelations** (not VDBEntities) |
| 4 | `TestQueryHybridDedupesChunks` | hybrid does not duplicate chunk IDs in `References` |
| 5 | `TestQueryRespectsMaxTotalTokens` | system-prompt word count ≤ MaxTotalTokens (+ small header envelope) |
| 6 | `TestQueryReturnsCitedChunkIDs` | every successful query returns a non-empty `References` list |
| 7 | `TestKeywordExtractionParsesBothLevels` | parser handles plain JSON, markdown fences, and surrounding prose |
| 8 | `TestPipelineInterfaceContract` | (compile-time) every store satisfies its interface |

## File map / 文件地图

| File | Role |
|---|---|
| `types.go` | Re-declares Provider / Embedder / Entity / Relationship / VectorStore / KVStore / GraphStore / QueryParam |
| `pipeline.go` | `Pipeline` struct + `Query` dispatch + in-memory store stubs |
| `keywords.go` | `extractKeywords` — one LLM call → JSON parse → (HL, LL) |
| `context_builder.go` | `truncateByTokens` + `buildSystemPrompt` (entities/relations/chunks) |
| `query_naive.go` | Naive mode: chunk vectors only |
| `query_local.go` | Local mode: LL keywords → entities → subgraph → chunks |
| `query_global.go` | Global mode: HL keywords → relations → endpoints → chunks |
| `query_hybrid.go` | Hybrid: both halves + chunk dedup + ONE final LLM call |
| `main.go` | CLI demo + MockProvider + MockEmbedder + seeded fixture |
| `query_test.go` | Eight tests (the seven required + the contract assertion) |

## Why hybrid wins / 为什么 hybrid 强

A pure-vector RAG (naive) gets pulled toward chunks that *quote* the
question's words; it misses entity-anchored answers when the chunk uses a
synonym. Local mode fixes this for "who/where/what is X" queries by letting
the graph spread out from the named entity. Global mode fixes the converse
("what themes / kinds-of conflicts arise") by retrieving on relation
descriptions. Hybrid runs both, dedupes the chunk list, then makes ONE
synthesis call so the LLM reconciles the two views — that's the part that
matters. Two answers concatenated would have inconsistent voice; one prompt
with both retrievals gets you a single coherent response.

## Upstream reference / 上游对照

- `lightrag/operate.py:3164-3410` `kg_query` orchestrator
- `lightrag/operate.py:4930-5200` `naive_query`
- `lightrag/operate.py:3516-4055` retrieval helpers
- `lightrag/prompt.py:374` `PROMPTS["keywords_extraction"]` template

The annotated excerpt lives at
[`upstream-readings/s11-query.py`](../../upstream-readings/s11-query.py).
The full chapter narrative is in
[`docs/zh/s11-query-modes.md`](../../docs/zh/s11-query-modes.md) and
[`docs/en/s11-query-modes.md`](../../docs/en/s11-query-modes.md).
