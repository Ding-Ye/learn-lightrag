---
title: "s_full · End-to-end integration"
chapter: full
slug: s_full-integration
est_read_min: 14
---

# s_full · End-to-end integration

> What this teaches: lace the eleven session modules into one LightRAG pipeline by tracing upstream's 16-step execution path for `Insert(text)` + `Query(q, hybrid)`. No new code — just one architecture diagram that names every Go file we wrote.

---

## Full-stack architecture

```
                         INGESTION                                 QUERY
┌──────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│                                          │  │                                        │
│  ainsert(text)                           │  │  aquery(q, mode=hybrid)                │
│      │                                   │  │      │                                 │
│      ▼                                   │  │      ▼                                 │
│  ┌─────────────────┐  s03                │  │  ┌────────────────────┐  s11           │
│  │ DocStatusStore  │  PENDING            │  │  │ keyword extractor  │  high+low      │
│  └─────────────────┘                     │  │  └────────────────────┘                │
│      │                                   │  │      │                                 │
│      ▼                                   │  │      ▼                                 │
│  ┌─────────────────┐  s04 (tiktoken)     │  │  ┌────────────────────┐  s07           │
│  │ ChunkByTokenSize│  1200/100 sliding   │  │  │ vdbEntities.Query  │  cosine top-K  │
│  └─────────────────┘                     │  │  │ vdbRelations.Query │                │
│      │                                   │  │  └────────────────────┘                │
│      ▼                                   │  │      │                                 │
│  ┌─────────────────┐  s06                │  │      ▼                                 │
│  │ Embedder.Embed  │  batch=128 + retry  │  │  ┌────────────────────┐  s08           │
│  └─────────────────┘                     │  │  │ Graph.GetSubgraph  │  BFS depth=1   │
│      │                                   │  │  └────────────────────┘                │
│      ▼                                   │  │      │                                 │
│  ┌─────────────────┐  s07 (chunks)       │  │      ▼                                 │
│  │ vdbChunks.Upsert│  ─────────────┐     │  │  ┌────────────────────┐  s05           │
│  └─────────────────┘               │     │  │  │ chunk fetch via KV │                │
│      │                              │    │  │  └────────────────────┘                │
│      ▼                              │    │  │      │                                 │
│  ┌─────────────────┐  s09 (per chunk)    │  │      ▼                                 │
│  │ Extractor       │ → entities + edges  │  │  ┌────────────────────┐  s11           │
│  │  ├─ gleaning N rounds               │  │  │ context_builder    │  budget split   │
│  │  └─ cache by sha256(chunk)          │  │  │ (entities/rels/    │  Entity / Rel / │
│  └─────────────────┘                  │  │  │  chunks per budget)│  Chunk tokens    │
│      │                                │  │  └────────────────────┘                │
│      ▼                                │  │      │                                 │
│  ┌─────────────────┐  s10                │  │      ▼                                 │
│  │ MergeEntities   │  threshold-based    │  │  ┌────────────────────┐  s02           │
│  │ MergeRelations  │  LLM map-reduce     │  │  │ Provider.Complete  │  retry on 429  │
│  └─────────────────┘                     │  │  └────────────────────┘                │
│      │                                   │  │      │                                 │
│      ▼                                   │  │      ▼                                 │
│  ┌─────────────────┐  s07 + s08          │  │  QueryResult{Content,                  │
│  │ vdbEntities &   │  upsert vectors     │  │              References,               │
│  │ vdbRelations &  │  upsert nodes/edges │  │              Mode}                     │
│  │ Graph           │                     │  │                                        │
│  └─────────────────┘                     │  │                                        │
│      │                                   │  │                                        │
│      ▼                                   │  │                                        │
│  ┌─────────────────┐  s03                │  │                                        │
│  │ DocStatusStore  │  PROCESSED          │  │                                        │
│  └─────────────────┘                     │  │                                        │
└──────────────────────────────────────────┘  └────────────────────────────────────────┘

                  KV Store (s05): chunks, entities cache, llm_response_cache
                  ──────────────────────────────────────────────────────────────
                                  shared between INGESTION and QUERY
```

---

## The 16-step execution trace

Scenario from the research dossier: user calls `rag.Insert("Eleanor Hartwell ...")` then `rag.Query("Where did Eleanor's papers end up?", QueryParam{Mode: ModeHybrid})`. Each row's parens cite the upstream Python file + line; the right column names which Go file in our repo carries the same load.

| # | Upstream action | Upstream location | Our Go equivalent |
|---|---|---|---|
| 1 | `LightRAG.insert(input)` sync entry | `lightrag/lightrag.py:~2023` | s01 `pipeline.go:Insert` |
| 2 | Delegates to `ainsert()`, validates input, generates MD5 ID | `lightrag/lightrag.py:1237` | s03 `hash.go:MD5DocID` + s01 pipeline |
| 3 | `apipeline_enqueue_documents`: write doc-status PENDING, dedup by MD5 | `lightrag/lightrag.py:~1400` | s03 `doc_status_store.go:Enqueue` |
| 4 | `apipeline_process_enqueue_documents`: semaphore-bounded concurrency + fetch doc | `lightrag/lightrag.py:~1450` | s09 + main pipeline using a buffered channel |
| 5 | `chunking_by_token_size`: tokenize → sliding window | `lightrag/operate.py:102-166` | s04 `chunking.go:ChunkByTokenSize` |
| 6 | Batch-embed chunks into `vdbChunks` | `lightrag/operate.py + utils.py` | s06 `embedder_openai.go:Embed` + s07 vdbChunks |
| 7 | `extract_entities`: per-chunk LLM call, delimiter parser, gleaning | `lightrag/operate.py:~2883-3163` | s09 `extraction.go:Extract` + `gleaning.go` |
| 8 | `_merge_nodes_then_upsert`: entity dedup + summarize + upsert graph + vdbEntities | `lightrag/operate.py:167-303` | s10 `merge.go:MergeEntities` + s08 `UpsertNode` + s07 vdbEntities |
| 9 | `_merge_edges_then_upsert`: same flow for relations + vdbRelations | `lightrag/operate.py` | s10 `merge.go:MergeRelationships` + s08 `UpsertEdge` + s07 vdbRelations |
| 10 | doc-status flips to PROCESSED; track_id completes | `lightrag/lightrag.py + json_doc_status_impl.py` | s03 `MarkProcessed` |
| 11 | User calls `rag.query("...", QueryParam(mode=hybrid))` | `lightrag/lightrag.py:2622` | s11 `pipeline.go:Query` dispatch |
| 12 | hybrid path: extract high+low keywords, vector-search entities + relations | `lightrag/operate.py:3164-3410` `kg_query` | s11 `keywords.go:extractKeywords` + `query_local.go` + `query_global.go` |
| 13 | Optional reranker on top-K chunks | `lightrag/operate.py + rerank` | s11 leaves a hook; not implemented (Appendix B exercise) |
| 14 | Context assembly: entities / relations / chunks each within their token budget | `lightrag/operate.py kg_query` | s11 `context_builder.go:buildSystemPrompt` |
| 15 | LLM generates: `llm_model_func(context + query)` | `lightrag/lightrag.py:2884-2970 aquery_llm` | s02 `provider_openai.go:Complete` |
| 16 | Return `QueryResult{content, references}` | `lightrag/lightrag.py + base.py:759` | s11 `pipeline.go` returns `QueryResult` |

---

## Deliberate omissions

| Upstream feature | Could have landed in | Why we skip it |
|---|---|---|
| Reranker (cross-encoder) | between s07 and s11 | Pedagogical noise; obscures the retrieval core |
| `mode=mix` (heuristic mode picker) | s11 | The four modes already make the contrast |
| `mode=bypass` (no retrieval) | s11 | Diagnostic only; nothing to learn from |
| Streaming responses (`Stream=true`) | s02 + s11 | Phase G addendum mentions it as a one-line swap |
| 13 storage backends (Postgres / Neo4j / Mongo / …) | s05 / s07 / s08 | Interface unification suffices; concrete backends become Appendix B exercises |
| Multi-doc semaphore scheduling in `apipeline_enqueue_documents` | s09 | Go uses channels; full scheduler complexity is out of scope |
| Cross-process `llm_response_cache` | s09 cache | Teaching value is in the in-process cache |
| Training side / Atropos / eval harness | — | Orthogonal to RAG inference |

---

## One-line summary

> 11 chapters of code + s_full = the reader can sketch LightRAG's full ingestion + query data flow from memory and point to the Go file in our repo that provides the minimum implementation of every arrow.
