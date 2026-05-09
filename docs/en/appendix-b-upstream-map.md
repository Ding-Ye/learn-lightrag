---
title: "Appendix B · Upstream source-reading map"
chapter: appendix-b
slug: appendix-b-upstream-map
est_read_min: 12
---

# Appendix B · Upstream source-reading map

> A walkable map of the upstream [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) Python codebase (~50K LOC): every key file pinned to the chapter that anchors it in this repo, plus a recommended traversal order from the perspective of *understanding* upstream. After 11 chapters, every pointer in this map should look familiar.

---

## Recommended traversal

Walking this list once = a second pass over the 11 chapters from upstream's vantage:

```
1. lightrag/__init__.py             — public API (what is LightRAG / QueryParam?)
2. lightrag/base.py:1-700           — abstractions: Storage / DocStatus / QueryParam / Schema
3. lightrag/operate.py:1-400        — chunking + utility helpers
4. lightrag/kg/json_kv_impl.py      — first concrete storage impl, see dataclass + namespace
5. lightrag/kg/nano_vector_db_impl.py — how the vector DB wraps nano-vectordb
6. lightrag/kg/networkx_impl.py     — KG impl + BFS subgraph
7. lightrag/operate.py:400-3200     — extraction + summarization mega-loop (huge; read in slices)
8. lightrag/lightrag.py             — central orchestrator that ties everything together
9. lightrag/operate.py:3200-5200    — the four query modes in actual code
10. lightrag/llm/openai.py          — LLM provider wrapper (also Phase G multi-model baseline)
11. lightrag/api/lightrag_server.py — FastAPI server (optional; in the exercise list)
```

---

## Upstream file → chapter mapping

Each row tells you the most useful mini-chapter to anchor on when reading the upstream file.

| Upstream file | Cited lines | Chapter | What to read for |
|---|---|---|---|
| `lightrag/base.py:71` | TextChunkSchema | s04 | chunk field contract |
| `lightrag/base.py:77` | QueryParam | s11 | query-param dataclass |
| `lightrag/base.py:189` | BaseVectorStorage | s07 | vector store abstraction |
| `lightrag/base.py:308` | BaseKVStorage | s05 | KV abstraction + filter_keys |
| `lightrag/base.py:333` | BaseGraphStorage | s08 | graph store + GetSubgraph |
| `lightrag/base.py:662-697` | DocStatus + DocProcessingStatus | s03 | state machine data shape |
| `lightrag/base.py:759` | QueryResult | s11 + s_full | query return shape |
| `lightrag/operate.py:102-166` | chunking_by_token_size | s04 | token sliding window |
| `lightrag/operate.py:167-303` | _handle_entity_relation_summary | s10 | threshold decision |
| `lightrag/operate.py:304-385` | _summarize_descriptions | s10 | recursive merge |
| `lightrag/operate.py:560-846` | rebuild_knowledge_from_chunks | (exercise) | incremental rebuild |
| `lightrag/operate.py:~2883-3163` | extract_entities | s09 | extraction pipeline + gleaning |
| `lightrag/operate.py:3164-3410` | kg_query | s11 | 4-mode dispatch |
| `lightrag/operate.py:3516-4055` | retrieval helpers | s11 | subgraph expansion + chunk collection |
| `lightrag/operate.py:4930-5200` | naive_query | s11 | simplest mode |
| `lightrag/kg/json_kv_impl.py` | full | s05 | JSON KVStore impl |
| `lightrag/kg/json_doc_status_impl.py` | full | s03 | doc-status persistence |
| `lightrag/kg/nano_vector_db_impl.py` | full | s07 | vector DB impl |
| `lightrag/kg/networkx_impl.py` | full | s08 | graph storage |
| `lightrag/kg/{neo4j,memgraph,postgres,mongo,faiss,milvus,qdrant,redis,opensearch}_impl.py` | full | (exercise) | other backends |
| `lightrag/llm/openai.py` | first 120 lines | s02 + s06 + Phase G | provider wrapper |
| `lightrag/llm/{anthropic,azure_openai,ollama,bedrock,hf,gemini}.py` | full | Phase G multi-model addendum | other providers |
| `lightrag/lightrag.py:1-150` | LightRAG @dataclass | s_full | central config |
| `lightrag/lightrag.py:1237` | ainsert | s_full step 2 | insert entry |
| `lightrag/lightrag.py:2622` | aquery | s_full step 11 | query entry (wrapper) |
| `lightrag/lightrag.py:2884-2970` | aquery_llm | s_full step 12-15 | the real query work |
| `lightrag/prompt.py` | PROMPTS dict | s09 + s10 + Appendix A | extraction / merge / keyword prompts |
| `lightrag/utils.py` | retry / async helpers | s02 + s06 | utility funcs (we re-implemented in stdlib) |
| `lightrag/api/lightrag_server.py` | full | (exercise) | FastAPI service |

---

## 5 recommended extension exercises

> Not on the main syllabus, but each is a typical "lift the mini to production" path. ~100-300 LOC each.

### 1. Neo4j graph backend

**What you learn**: ground s08's `GraphStore` interface on a production database. Neo4j expresses graph queries in Cypher — more scalable than NetworkX's in-memory adjacency.

**Scope**:
- Add `neo4j_graph_store.go` next to `adjacency_graph.go` (don't touch the latter); implement `GraphStore`.
- Connect via `github.com/neo4j/neo4j-go-driver/v5`.
- `UpsertNode` → `MERGE (n:Entity {name: $name}) SET n.description = $desc`.
- `GetSubgraph` → Cypher `MATCH (n:Entity {name: $seed})-[r*1..$depth]-(m) RETURN n, r, m LIMIT $maxNodes`.
- Integration tests via `dockertest` spinning a Neo4j container.

**~300 LOC**. After: a `-backend neo4j` flag on s08's demo.

---

### 2. Streaming responses through s11

**What you learn**: turn on the `Provider.Complete` `Stream=true` path — LLM yields tokens as they arrive; UI can show typing effect.

**Scope**:
- Change `Provider` to `Complete(ctx, req) (<-chan Token, error)` (or add a separate `CompleteStream` for back-compat).
- s11's `pipeline.go:Query` returns `(chan TokenWithRefs, error)` — emits References first, then content tokens.
- main.go demo prints as it receives.

**Note**: Anthropic / OpenAI SSE formats differ slightly; the Phase G multi-model doc covers the difference.

**~200 LOC**.

---

### 3. Plug in a reranker

**What you learn**: insert a cross-encoder reranker between retrieval and LLM. Boosts hybrid mode recall by 5-10%.

**Scope**:
- Add a `Reranker` interface to s11: `Score(query string, candidates []string) []float32`.
- Implement `CohereReranker` or `JinaReranker` (both HTTP).
- `Pipeline.Query` calls `Reranker.Score` after retrieval and before context assembly; reorder top-K by score.

**~150 LOC**.

---

### 4. Postgres KV backend

**What you learn**: ground s05's `KVStore` interface on a real SQL database. Same interface → very different storage engine, all callers unchanged.

**Scope**:
- Add `postgres_kv_store.go` using `github.com/jackc/pgx/v5`.
- Schema: `CREATE TABLE kv (namespace TEXT, id TEXT, data JSONB, updated_at TIMESTAMPTZ, PRIMARY KEY (namespace, id))`.
- `Upsert` → `INSERT ... ON CONFLICT (namespace, id) DO UPDATE`.
- `FilterMissing` → `SELECT id FROM unnest($ids) AS u(id) LEFT JOIN kv ON kv.id = u.id WHERE kv.id IS NULL`.
- Integration tests via `dockertest`.

**~250 LOC**.

---

### 5. Tiny HTTP server in front of s11

**What you learn**: wrap the mini as a remote-callable service, mirroring upstream's `lightrag/api/lightrag_server.py`.

**Scope**:
- Add `server.go` next to s11; use `net/http` (**no web framework**).
- `POST /insert` (body: `{"text": "...", "doc_id": "..."}`) and `POST /query` (body: `{"q": "...", "mode": "hybrid"}`).
- JSON-encode `QueryResult`.
- `Authorization: Bearer <token>` middleware.

**~150 LOC**. After: `curl localhost:8080/query -d '{"q":"..."}'` works.

---

## One-line summary

> After all 11 chapters, this map tells you **"next time you want to read upstream feature X, here's the mini-chapter to anchor on"**. Every upstream file has a closest mini-chapter to dock with.
