---
title: "s_full · 端到端集成"
chapter: full
slug: s_full-integration
est_read_min: 14
---

# s_full · 端到端集成

> 教什么：把 11 节的 Go 模块**逻辑上**串成一条 LightRAG 全栈流水线，跟着上游的 16 步执行轨迹追一遍 `Insert(text)` + `Query(q, hybrid)`。本章不写新代码——只把已经写过的零件拼成一张完整的架构图。

---

## 全栈架构图

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
│  │ vdbChunks.Upsert│  ─────────────┐     │  │  ┌────────────────────┐                │
│  └─────────────────┘               │     │  │  │ chunk fetch via KV │  s05           │
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

## 16 步执行轨迹

跟着研究 dossier 的轨迹场景：用户先 `rag.Insert("Eleanor Hartwell ...")` 再 `rag.Query("Where did Eleanor's papers end up?", QueryParam{Mode: ModeHybrid})`。每一步括号里是上游 Python 文件 + 行号，箭头后是我们 Go 实现里**对应哪一节**的哪个文件。

| # | 上游动作 | 上游位置 | 我们的 Go 等价 |
|---|---|---|---|
| 1 | `LightRAG.insert(input)` 入口（同步包装） | `lightrag/lightrag.py:~2023` | s01 `pipeline.go:Insert` |
| 2 | 委托给 `LightRAG.ainsert()`，校验输入、生成 MD5 ID | `lightrag/lightrag.py:1237` | s03 `hash.go:MD5DocID` + s01 pipeline |
| 3 | `apipeline_enqueue_documents`：写 doc-status 为 PENDING，按 MD5 去重 | `lightrag/lightrag.py:~1400` | s03 `doc_status_store.go:Enqueue` |
| 4 | `apipeline_process_enqueue_documents`：信号量并发 + 取 doc 文本 | `lightrag/lightrag.py:~1450` | s09 + 主 pipeline 用 buffered channel 模拟 |
| 5 | `chunking_by_token_size`：tokenize → 滑窗切分 | `lightrag/operate.py:102-166` | s04 `chunking.go:ChunkByTokenSize` |
| 6 | 批量 embed chunks 到 `vdbChunks` | `lightrag/operate.py + utils.py` | s06 `embedder_openai.go:Embed` + s07 vdbChunks |
| 7 | `extract_entities`：每 chunk 调 LLM，分隔符解析，gleaning | `lightrag/operate.py:~2883-3163` | s09 `extraction.go:Extract` + `gleaning.go` |
| 8 | `_merge_nodes_then_upsert`：实体 dedup + summarize + upsert graph + vdbEntities | `lightrag/operate.py:167-303` 等 | s10 `merge.go:MergeEntities` + s08 `adjacency_graph.go:UpsertNode` + s07 vdbEntities |
| 9 | `_merge_edges_then_upsert`：关系 dedup + summarize + upsert graph + vdbRelations | `lightrag/operate.py` | s10 `merge.go:MergeRelationships` + s08 `UpsertEdge` + s07 vdbRelations |
| 10 | doc-status 转 PROCESSED；track_id 完成 | `lightrag/lightrag.py + json_doc_status_impl.py` | s03 `MarkProcessed` |
| 11 | 用户调 `rag.query("...", QueryParam(mode=hybrid))` | `lightrag/lightrag.py:2622` | s11 `pipeline.go:Query` 路由 |
| 12 | hybrid 路径：抽 high+low 关键词，向量搜 entities + relations | `lightrag/operate.py:3164-3410` `kg_query` | s11 `keywords.go:extractKeywords` + `query_local.go` + `query_global.go` |
| 13 | 可选 reranker：对 top-K chunks 重排 | `lightrag/operate.py + rerank` | s11 留 hook，未实现（Appendix B 扩展练习） |
| 14 | context 拼装：entities + relations + chunks 各自 token 预算 | `lightrag/operate.py kg_query` | s11 `context_builder.go:buildSystemPrompt` |
| 15 | LLM 生成：`llm_model_func(context + query)` | `lightrag/lightrag.py:2884-2970 aquery_llm` | s02 `provider_openai.go:Complete` |
| 16 | 返回 `QueryResult{content, references}` | `lightrag/lightrag.py + base.py:759` | s11 `pipeline.go` 返回 `QueryResult` |

---

## 故意省略的特性

| 上游有，我们 mini 没做 | 在哪一节本可以加 | 为什么不做 |
|---|---|---|
| Reranker (cross-encoder) | s07 / s11 之间 | 教学冗余，掩盖检索本质 |
| `mode=mix` (启发式选 mode) | s11 | 4 个 mode 已足够说明对照 |
| `mode=bypass` (跳过检索) | s11 | 仅诊断价值，不教学 |
| 流式响应 (`Stream=true`) | s02 + s11 | 留给 Phase G 多模型 addendum 提及 |
| 13 个存储后端（Postgres / Neo4j / Mongo …）| s05 / s07 / s08 | 接口统一即可，具体后端是 Appendix B 练习 |
| `apipeline_enqueue_documents` 的 multi-doc 信号量调度 | s09 | Go 用 channel 替代，但完整调度复杂度被砍 |
| `llm_response_cache` 跨进程持久化 | s09 cache | 教学只展示进程内 cache 的价值 |
| 训练侧 / Atropos / 评测脚手架 | — | 与 RAG 推理无关 |

---

## 一句话总结

> 11 节代码 + s_full = 学完后能徒手画出 LightRAG 的 ingestion + query 双向数据流，并指出每条线背后哪一节的 Go 文件提供了它的最小实现。
