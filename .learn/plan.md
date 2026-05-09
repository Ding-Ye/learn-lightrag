# Curriculum plan: learn-lightrag

## Locked decisions

| | |
|---|---|
| Upstream | https://github.com/HKUDS/LightRAG |
| Target language | Go 1.22+ |
| Chapter count | 11 code chapters (s01-s11) + s_full integration + 2 appendices |
| Has LLM layer | yes (Phase G addendum runs) |
| Default LLM provider in s01 | OpenAI chat-completion + OpenAI embeddings |
| Offline CI provider | `-provider mock` flag (deterministic fake embeddings + canned LLM responses) |
| Total LOC budget | ≤ 7000 across all sessions; current estimate ~5,600 |
| Module convention | `learn-lightrag/sNN`, `package main`, `go run .` runs the chapter demo |
| Storage on disk | mirrors upstream `./rag_storage/` layout under `./lightrag-data/` |

## Curriculum table

| # | slug | title (zh) | title (en) | mechanism (upstream file) | dependency | est. LOC |
|---|------|------------|------------|---------------------------|------------|----------|
| s01 | minimum-loop | 最小 RAG 闭环 | Minimum RAG loop | end-to-end pipeline scaffold (lightrag/lightrag.py:~2023 insert + :~2601 query, simplified) | (none) | ~400 |
| s02 | provider | 提供方接口 (OpenAI 聊天补全) | Provider interface (OpenAI chat completion) | lightrag/llm/openai.py first 120 lines + llm_model_func DI in lightrag/lightrag.py:~413 | s01 | ~350 |
| s03 | doc-status | 文档状态机 | Document status state machine | lightrag/base.py:662-697 + lightrag/kg/json_doc_status_impl.py | s01 | ~250 |
| s04 | chunking | 基于 token 的切分 | Token-based chunking with overlap | lightrag/operate.py:102-166 chunking_by_token_size | s01 | ~450 |
| s05 | kv-store | 键值存储与过滤 | KV store with filter_keys | lightrag/kg/json_kv_impl.py + lightrag/base.py:308 BaseKVStorage | s01, s03 | ~400 |
| s06 | embeddings | 嵌入提供方与批处理 | Embedding provider with batching | lightrag/llm/openai.py embedding wrapper + embedding_func contract in lightrag.py:~384 | s02 | ~350 |
| s07 | vector-store | 余弦相似向量库 | Cosine-similarity vector store | lightrag/kg/nano_vector_db_impl.py + lightrag/base.py:189 BaseVectorStorage | s05, s06 | ~500 |
| s08 | graph-store | 邻接图存储与子图 | Adjacency graph store and subgraph BFS | lightrag/kg/networkx_impl.py + lightrag/base.py:333 BaseGraphStorage | s05 | ~600 |
| s09 | extraction | 实体关系抽取与 gleaning | Entity/relation extraction with gleaning | lightrag/operate.py:~2883-3163 extract_entities + lightrag/prompt.py templates | s02, s04, s08 | ~800 |
| s10 | summarization | 描述归并(map-reduce) | Map-reduce description summarization | lightrag/operate.py:167-303 _handle_entity_relation_summary + :304-385 _summarize_descriptions | s02, s09 | ~500 |
| s11 | query-modes | 双层检索四种模式 | Dual-level retrieval, four modes | lightrag/operate.py:3164-3410 kg_query + :4930-5200 naive_query + :3516-4055 retrieval helpers | s07, s08, s10 | ~1000 |
| s_full | integration | 端到端集成 | End-to-end integration | (doc only — combines all; spine = 16-step trace from research-notes.md) | all | ~600 (docs) |
| App. A | prompt-secrets | 附录 A · 提示工程的秘密 | Appendix A · Prompt-engineering secret sauce | mental model (no code) | (none) | doc |
| App. B | upstream-map | 附录 B · 上游源码导读地图 | Appendix B · upstream map | reference table | (none) | doc |

**Properties verified:**
- All 11 sessions form a DAG: s01 has no deps; sN cites only s01..s(N-1).
- s01 is a complete runnable end-to-end (insert + naive query) using OpenAI in ~400 LOC; the other 10 chapters then deepen one piece each.
- s11 (query-modes) is the most architecturally distinctive (LightRAG's signature dual-level retrieval) and sits last per the brief.
- LOC sum ≈ 5,600 ≤ 7,000.

## Shared types catalog

These canonical Go signatures are the contract every session inherits / extends. Each session re-declares the types it needs (no cross-package imports), but the **shape** stays identical so a learner moving from s05 to s07 sees the same `KVStore` interface re-appear with one method added or implemented for real. The catalog compiles in isolation under `package types` against stdlib only; in actual sessions these are re-typed inside `package main`.

```go
package types

import (
    "context"
    "time"
)

// --- Provider abstraction for the LLM (chat completion, NOT agent-loop tool-use) ---
// Models the upstream `llm_model_func: Callable` signature: take a prompt + system, return a string.
// Introduced in s01 with a single OpenAI implementation; re-exported and made swap-friendly in s02.
// Phase G addendum adds Anthropic / Bedrock / Ollama implementations as additional files; the
// interface MUST NOT change after s02.
type Provider interface {
    Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
}

type CompleteRequest struct {
    Model       string
    System      string
    Messages    []Message // user/assistant turns, oldest first
    MaxTokens   int
    Temperature float64
    Stream      bool      // streaming reserved for s11 + Phase G; s01..s10 set false
}

type Message struct {
    Role    string // "user" | "assistant" | "system"
    Content string
}

type CompleteResponse struct {
    Text         string
    InputTokens  int
    OutputTokens int
}

// --- Embedding provider ---
// Introduced in s01 inline; promoted to its own interface in s06 once batching matters.
type EmbeddingProvider interface {
    Embed(ctx context.Context, texts []string) ([][]float32, error)
    Dim() int
}

// --- Tokenizer (for chunking) ---
// Introduced in s01 with a whitespace-stub; replaced in s04 with a real tiktoken-backed impl
// (github.com/pkoukk/tiktoken-go, cl100k_base, pinned in go.mod).
type Tokenizer interface {
    Encode(text string) []int
    Decode(tokens []int) string
}

// --- DocStatus state machine (upstream lightrag/base.py:662) ---
// Introduced as plain string constants in s01; promoted to a proper state machine + persisted
// store in s03 with transition validation.
type DocStatus string

const (
    DocStatusPending    DocStatus = "pending"
    DocStatusProcessing DocStatus = "processing"
    DocStatusProcessed  DocStatus = "processed"
    DocStatusFailed     DocStatus = "failed"
)

// DocProcessingStatus mirrors lightrag/base.py:667-697.  Defined in s03 and re-used by s09/s10/s11
// for resume-on-failure semantics.
type DocProcessingStatus struct {
    DocID          string
    ContentSummary string
    ContentLength  int
    FilePath       string
    Status         DocStatus
    CreatedAt      time.Time
    UpdatedAt      time.Time
    TrackID        string
    ChunksCount    int
    ChunksList     []string
    ErrorMsg       string
    Metadata       map[string]any
}

// --- Chunk (upstream TextChunkSchema, lightrag/base.py:71) ---
// Introduced in s01 (with naive newline-split content); s04 fills in real Tokens count.
type Chunk struct {
    ContentDocID    string // parent doc ID
    Content         string
    Tokens          int
    ChunkOrderIndex int
}

// --- Entity & Relationship (extracted KG nodes/edges) ---
// Introduced in s08 (graph-store) as the in-memory shape; s09 (extraction) populates them
// from LLM output; s10 (summarization) rewrites Description.
type Entity struct {
    Name        string   // canonical key (already normalized)
    Type        string   // PERSON | ORG | EVENT | CONCEPT | ...
    Description string   // merged from many chunks
    SourceIDs   []string // chunk IDs that mention it
}

type Relationship struct {
    SrcID, TgtID string
    Keywords     string
    Description  string
    Weight       float32
    SourceIDs    []string
}

// --- KVStore (upstream lightrag/base.py:308 BaseKVStorage) ---
// Stubbed as a sync.Map in s01; full JSON-on-disk implementation in s05 with FilterMissing
// (the upstream `filter_keys` semantic that drives "what new chunks need extracting").
type KVStore interface {
    Get(ctx context.Context, id string) (map[string]any, bool, error)
    GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error)
    Upsert(ctx context.Context, items map[string]map[string]any) error
    FilterMissing(ctx context.Context, ids []string) ([]string, error)
    Delete(ctx context.Context, ids []string) error
    Persist(ctx context.Context) error // flush to disk
}

// --- VectorStore (upstream lightrag/base.py:189 BaseVectorStorage) ---
// Stubbed as a slice scan in s01; real cosine-similarity index with thresholding in s07.
type VectorRecord struct {
    ID       string
    Vector   []float32
    Metadata map[string]any
}

type VectorHit struct {
    ID       string
    Score    float32 // cosine similarity in [-1, 1]; LightRAG conventionally clamps >= 0
    Metadata map[string]any
}

type VectorStore interface {
    Upsert(ctx context.Context, records []VectorRecord) error
    Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error)
    Delete(ctx context.Context, ids []string) error
    Persist(ctx context.Context) error
}

// --- GraphStore (upstream lightrag/base.py:333 BaseGraphStorage) ---
// Introduced in s08 as an undirected adjacency map.  GetSubgraph implements upstream's
// BFS-with-degree-priority for context retrieval (replaces NetworkX in ~100 LOC).
type Subgraph struct {
    Nodes []Entity
    Edges []Relationship
}

type GraphStore interface {
    UpsertNode(ctx context.Context, e Entity) error
    UpsertEdge(ctx context.Context, r Relationship) error
    GetNode(ctx context.Context, id string) (Entity, bool, error)
    GetSubgraph(ctx context.Context, seed string, maxDepth, maxNodes int) (Subgraph, error)
    Persist(ctx context.Context) error
}

// --- QueryParam & QueryResult (upstream lightrag/base.py:77 / :759) ---
// Introduced in s01 with only Mode + TopK populated; s11 fills out the rest of the budgets.
type QueryMode string

const (
    ModeNaive  QueryMode = "naive"
    ModeLocal  QueryMode = "local"
    ModeGlobal QueryMode = "global"
    ModeHybrid QueryMode = "hybrid"
)

type QueryParam struct {
    Mode              QueryMode
    TopK              int
    ChunkTopK         int
    MaxEntityTokens   int
    MaxRelationTokens int
    MaxTotalTokens    int
}

type QueryResult struct {
    Content    string    // final answer text from the LLM
    References []string  // chunk IDs cited in the context window
    Mode       QueryMode
}
```

**Provider note (load-bearing for Phase G).** Upstream LightRAG accepts any `llm_model_func: Callable`; our Go equivalent is the single-method `Provider` interface above. The s01 ships ONE concrete implementation, `OpenAIProvider`, hitting `https://api.openai.com/v1/chat/completions`. Phase G later adds `AnthropicProvider`, `BedrockProvider`, `OllamaProvider` profiles — each is one new `*_provider.go` file, no signature changes. The same shape rule applies to `EmbeddingProvider` — s01 starts with `OpenAIEmbedder`; Phase G adds others.

**Anti-patterns avoided.** The catalog deliberately rejects (1) `dict[str, Any]` map shapes (we use named structs everywhere except where upstream's metadata map is genuinely free-form); (2) dual sync/async APIs (each interface has one method, callers pass `context.Context`); (3) string factory keys (we use plain interface satisfaction + functional options).

## Per-session detail

### s01 — minimum-loop

- **Problem.** A learner needs to feel "I ingested a real doc and asked it a real question" before any abstraction lands. Without this, the next 10 sessions feel like building Lego pieces with no picture on the box.
- **Solution.** Hard-code a 5-stage pipeline inline: read text file → newline-chunk → embed each chunk via OpenAI → cosine-rank against the query embedding → stuff top-3 chunks into a system prompt → call OpenAI chat completion. Every storage is a `sync.Map`; every interface listed in the catalog is **declared but stubbed** so later chapters have the seams to swap into. The end-user runs `OPENAI_API_KEY=sk-... go run . "What did the doc say?"` and gets a real answer.
- **Code surface.** `provider.go` (Provider interface + OpenAIProvider 80 LOC), `embedder.go` (EmbeddingProvider + OpenAIEmbedder 60 LOC), `chunking.go` (newline + token-cap-by-rune-stub 40 LOC), `vectorstore.go` (slice-scan cosine 50 LOC), `kv.go` (sync.Map wrapper 30 LOC), `pipeline.go` (Insert + Query plumbing 80 LOC), `main.go` (CLI parsing + book.txt loader 50 LOC), `lightrag_test.go` (mock provider tests), `testdata/book.txt` (Christmas Carol excerpt mirroring the upstream demo).
- **Tests.** `TestInsertChunkCount`, `TestQueryUsesTopChunk`, `TestProviderInterfaceContract`, `TestMockProviderDeterministic`, `TestPipelineEndToEndWithMock`. CI uses `-provider mock` exclusively; OPENAI_API_KEY only needed for local runs.
- **Upstream ref.** `lightrag/lightrag.py:~2023` (insert entry) and `:~2601` (query entry), simplified — show the learner the 5-stage shape that the next 10 chapters will refine.
- **What changed vs. previous.** N/A (first session).

### s02 — provider

- **Problem.** s01 inlined the OpenAI HTTP call inside `pipeline.go`. To swap providers (Phase G's Anthropic / Bedrock / Ollama), the call site must depend on an interface, not a struct. Plus s01's mock provider is tangled with the test file.
- **Solution.** Promote `Provider` to a real package-internal abstraction: one `Provider` interface, three implementations (`OpenAIProvider`, `MockProvider`, `EchoProvider` for trivial smoke tests), constructed via functional options (`NewOpenAIProvider(WithModel("gpt-4o-mini"), WithTimeout(30*time.Second))`). Includes retry-with-backoff (`tenacity`-equivalent in stdlib) and structured logging of input/output tokens.
- **Code surface.** `provider.go` (interface + options 80 LOC), `provider_openai.go` (HTTP impl with retry 150 LOC), `provider_mock.go` (deterministic canned responses keyed by request hash 60 LOC), `main.go` (showcases provider swap via flag 40 LOC), `provider_test.go`.
- **Tests.** `TestOpenAIProviderHappyPath` (uses httptest server), `TestProviderRetryOn429`, `TestMockProviderDeterminism`, `TestProviderRespectsContextCancel`, `TestProviderTokenAccounting`.
- **Upstream ref.** `lightrag/llm/openai.py` first ~120 lines + `llm_model_func` injection at `lightrag/lightrag.py:~413`. Annotated extract goes to `upstream-readings/s02-openai.py`.
- **What changed vs. s01.** Pulls Provider out of `pipeline.go`; adds retry + token accounting + mock parity. No pipeline changes.

### s03 — doc-status

- **Problem.** s01 forgot a doc once embedded — re-running on the same file double-ingests. Upstream's killer feature is resume-on-failure: a partial run picks up at PROCESSING with no duplicate chunks.
- **Solution.** Implement the `DocStatus` state machine (PENDING → PROCESSING → PROCESSED/FAILED) with a JSON-persisted `DocStatusStore`. Transition validation lives in `MarkProcessing(docID)` etc. — invalid transitions return a typed error. The store is content-addressed (MD5 of doc text) so re-inserting identical text is a no-op.
- **Code surface.** `doc_status.go` (DocStatus + DocProcessingStatus types 60 LOC), `doc_status_store.go` (JSON persistence + transition validation 120 LOC), `main.go` (insert flow exercises all 4 transitions including a synthetic failure 50 LOC), `doc_status_test.go`.
- **Tests.** `TestStatusTransitionsValid`, `TestStatusInvalidTransitionRejected`, `TestStatusPersistAcrossRestart`, `TestDuplicateInsertIsNoop`, `TestFailedDocPreservesErrorMsg`.
- **Upstream ref.** `lightrag/base.py:662-697` (DocStatus + DocProcessingStatus dataclasses) + `lightrag/kg/json_doc_status_impl.py` (full file, ~150 LOC Python).
- **What changed vs. s01.** Adds the doc-status store in front of `pipeline.Insert`. Insert now writes PENDING first, transitions through PROCESSING, and only ends in PROCESSED if all chunks embedded. No swap in chunking / vector / provider logic.

### s04 — chunking

- **Problem.** s01's "split on newlines" loses semantic context across paragraph breaks and ignores token limits. Real OpenAI calls fail past 8K tokens. Plus chunk count is wildly variable across docs.
- **Solution.** Implement upstream's `chunking_by_token_size`: tokenize via `github.com/pkoukk/tiktoken-go` (cl100k_base, pinned), slide a window of `chunkTokenSize=1200` with `overlapTokenSize=100`, attach `ChunkOrderIndex`. Optional `splitByCharacter` fallback for non-English docs that tiktoken handles poorly.
- **Code surface.** `tokenizer.go` (Tokenizer interface + tiktoken wrapper + WhitespaceTokenizer fallback 80 LOC), `chunking.go` (windowed splitter with overlap 180 LOC), `main.go` (prints chunks + token counts for a sample text 40 LOC), `chunking_test.go`, `testdata/book.txt`.
- **Tests.** `TestChunkRespectsTokenLimit`, `TestChunkOverlapPreserved`, `TestChunkOrderIndexMonotonic`, `TestChunkSplitByCharacterFallback`, `TestChunkEmptyTextReturnsEmpty`, `TestChunkExactlyAtBoundary`.
- **Upstream ref.** `lightrag/operate.py:102-166` (full `chunking_by_token_size`).
- **What changed vs. s03.** Replaces the newline splitter inside `pipeline.go`; downstream chunk counts are now deterministic for a given (text, size, overlap) triple. Doc-status's `ChunksCount` field is now populated correctly.

### s05 — kv-store

- **Problem.** s01's `sync.Map` evaporates on restart and offers no batch APIs. The pipeline asks "of these 50 chunk IDs, which are not yet embedded?" thousands of times and needs a `FilterMissing` primitive — upstream's `filter_keys`.
- **Solution.** Implement `JSONKVStore` (one JSON file per namespace under `./lightrag-data/kv_<namespace>.json`) with per-key locking via `sync.Map[string]*sync.Mutex`. Atomic-rename persistence (write to `.tmp`, fsync, rename). `FilterMissing` is the centerpiece — given a list of IDs, return the subset NOT in the store. This is the primitive that drives "which new chunks need extracting" in s09.
- **Code surface.** `kv_store.go` (KVStore interface 30 LOC), `kv_json_store.go` (full impl with persistence 220 LOC), `main.go` (small CRUD demo 40 LOC), `kv_json_store_test.go`, `testdata/seed.json`.
- **Tests.** `TestKVUpsertGetRoundTrip`, `TestKVFilterMissingPartial`, `TestKVPersistAcrossRestart`, `TestKVConcurrentUpsertSafe`, `TestKVAtomicWriteOnCrash` (truncates `.tmp` mid-write), `TestKVDeleteRemovesKey`.
- **Upstream ref.** `lightrag/kg/json_kv_impl.py` (full file) + `lightrag/base.py:308` BaseKVStorage abstract interface.
- **What changed vs. s04.** Replaces s01's `sync.Map` everywhere. Doc-status from s03 now persists via the same `JSONKVStore` (proves the interface is reusable).

### s06 — embeddings

- **Problem.** s01 embedded one chunk at a time, paying per-call HTTP overhead. OpenAI's embedding endpoint accepts up to ~2048 inputs per call; ignoring this means 100x slower ingestion. Also: no retry, no dimension introspection, no test-double.
- **Solution.** Promote `EmbeddingProvider` to a real package: `OpenAIEmbedder` batches up to `BatchSize=128` inputs per HTTP call (matching upstream's `nano-vectordb` typical config), surfaces `Dim()` (1536 for `text-embedding-3-small`), and wraps the same retry-with-backoff helper from s02. Adds `MockEmbedder` that hashes input text to a deterministic 1536-d unit vector — used by s07's tests and CI.
- **Code surface.** `embedder.go` (interface + options 60 LOC), `embedder_openai.go` (batched HTTP call 150 LOC), `embedder_mock.go` (deterministic hash-to-vector 60 LOC), `main.go` (embeds 200 chunks and reports throughput 40 LOC), `embedder_test.go`.
- **Tests.** `TestEmbedderBatchesAtLimit`, `TestEmbedderDimMatchesProvider`, `TestEmbedderHandlesEmptyInputSlice`, `TestEmbedderRetriesOn429`, `TestMockEmbedderDeterministic`, `TestMockEmbedderUnitNormalized`.
- **Upstream ref.** `lightrag/llm/openai.py` (the `openai_embed` wrapper, ~30-line slice) + `embedding_func` contract referenced in `lightrag/lightrag.py:~384`.
- **What changed vs. s05.** Pulls embedding out of `pipeline.go`'s loop; same Provider pattern as s02.

### s07 — vector-store

- **Problem.** s01's vector store was a slice + linear cosine scan; that's O(N·d) per query but ALSO has no thresholding, no metadata, no persistence, and conflates the **three** vector indices LightRAG actually maintains (chunks / entities / relations). With graph data arriving in s08-s09, we cannot keep going with one bag-of-vectors.
- **Solution.** Implement an in-memory cosine-similarity index over `[]VectorRecord` with persistent JSON (or gob) snapshot — explicitly the upstream `nano_vector_db_impl.py` shape, minus the float16+zlib+base64 compression (we discuss but don't implement; appendix exercise). `Query(query, topK, threshold)` returns a sorted `[]VectorHit` filtered by threshold. The `pipeline` now spins up THREE distinct `VectorStore` instances (`vdbChunks`, `vdbEntities`, `vdbRelations`) using the same impl.
- **Code surface.** `vector_store.go` (interface + types 40 LOC), `cosine_index.go` (in-memory impl 250 LOC), `cosine_persist.go` (atomic JSON snapshot 80 LOC), `main.go` (loads embedded chunks from s06 testdata, runs three sample queries 60 LOC), `cosine_index_test.go`.
- **Tests.** `TestCosineRanksMostSimilarFirst`, `TestCosineThresholdFiltersLowScores`, `TestCosineRespectsTopK`, `TestCosinePersistAcrossRestart`, `TestCosineHandlesEmptyIndex`, `TestCosineMetadataRoundtrip`.
- **Upstream ref.** `lightrag/kg/nano_vector_db_impl.py` (full file) + `lightrag/base.py:189` BaseVectorStorage.
- **What changed vs. s06.** Pipeline's Query now uses the new `cosine_index` for chunk lookup; Insert calls `vdbChunks.Upsert(...)` after embedding.

### s08 — graph-store

- **Problem.** Without a graph layer there is no "Light**RAG**" — only a flat chunk vector index, indistinguishable from textbook RAG. We need entities (nodes) and relationships (undirected edges) with neighborhood traversal so s11's local/global modes have something to retrieve.
- **Solution.** Implement an undirected `AdjacencyGraph` using `map[string]Entity` for nodes and `map[edgeKey]Relationship` for edges (where `edgeKey` canonicalizes `(min, max)`). `GetSubgraph(seed, maxDepth, maxNodes)` is BFS with degree-based priority (visit higher-degree neighbors first, matching upstream's NetworkX behavior). Persistence uses a simple JSON adjacency dump (we discuss GraphML as exercise).
- **Code surface.** `graph_store.go` (interface + types 50 LOC), `adjacency_graph.go` (full impl 350 LOC), `subgraph_bfs.go` (BFS with degree priority 120 LOC), `main.go` (builds 6-node graph and prints subgraph at depth 2 50 LOC), `adjacency_graph_test.go`, `testdata/seed_graph.json`.
- **Tests.** `TestUpsertNodeIdempotent`, `TestUpsertEdgeUndirected`, `TestSubgraphBFSDepthLimit`, `TestSubgraphRespectsMaxNodes`, `TestSubgraphPrioritizesHighDegree`, `TestGraphPersistAcrossRestart`.
- **Upstream ref.** `lightrag/kg/networkx_impl.py` (full file) + `lightrag/base.py:333` BaseGraphStorage.
- **What changed vs. s07.** Adds the graph layer alongside the three vector stores; the pipeline still doesn't put anything in it (extraction in s09 will). Insert is unchanged; Query path unchanged.

### s09 — extraction

- **Problem.** A graph store is empty until something tells it about entities and relationships. Upstream's killer move is to use the LLM itself as the extractor, with a **delimiter-based** output format (not JSON) and a **gleaning** loop that re-asks "any more entities?" N times to improve recall. Without this, our graph stays empty and s11 can't do local/global retrieval.
- **Solution.** Implement `Extractor` that, given a chunk, calls `Provider.Complete` with the upstream extraction system prompt, parses delimited output (`<|#|>`-separated tuples for entities and relationships), validates fields, optionally runs `MaxGleaningRounds` continuation prompts. Caches by `hash(chunk_content)` in the `llm_response_cache` KV from s05 to avoid re-extracting the same chunk on re-run. Writes entities/relations to `graphStore` and to `vdbEntities`/`vdbRelations` (descriptions are embedded for vector lookup).
- **Code surface.** `extraction.go` (Extractor + parser 220 LOC), `extraction_prompt.go` (system prompt + continuation prompt verbatim from upstream `lightrag/prompt.py` 80 LOC of string literals), `gleaning.go` (continuation loop with same parser 80 LOC), `cache.go` (hash-keyed wrapper around s05 KV 60 LOC), `main.go` (extracts from 5-chunk demo doc, prints entities/relations, persists 80 LOC), `extraction_test.go`, `testdata/llm_responses/` (canned LLM outputs for parser tests).
- **Tests.** `TestExtractionParsesDelimitedTuples`, `TestExtractionGleaningAddsNewEntities`, `TestExtractionCacheAvoidsDoubleCall`, `TestExtractionMalformedOutputFallsBack`, `TestExtractionMergesSourceIDsAcrossChunks`, `TestExtractionRespectsContextCancel`.
- **Upstream ref.** `lightrag/operate.py:~2883-3163` `extract_entities` (slice fetched in 100-line windows; do NOT fetch the whole 5K-LOC file) + `lightrag/prompt.py` (`entity_extraction_system_prompt`, `entity_continue_extraction_user_prompt`).
- **What changed vs. s08.** Adds the extraction step between chunking and graph-store; pipeline now: chunk → embed-chunks → extract-per-chunk → upsert-graph + upsert-entity-vdb + upsert-relation-vdb. Cache hit ratio printed in CLI output.

### s10 — summarization

- **Problem.** When the same entity ("Scrooge") appears in 47 chunks, we end up with 47 description fragments. Concatenating them blows past the embedding model's token limit and confuses the LLM at retrieval time. Upstream uses a clever heuristic: under threshold, just concat; over threshold, recursively LLM-summarize.
- **Solution.** Implement `SummarizeDescriptions(descriptions []string, budgetTokens int) (string, bool, error)` mirroring `_handle_entity_relation_summary` + `_summarize_descriptions`. Decision tree: if `total_tokens < budget` AND `count < threshold (default 6)`, return `strings.Join(descriptions, "\n\n")` with `wasLLMUsed=false`; otherwise chunk descriptions into `summary_context_size`-token windows, call Provider.Complete with the summary system prompt on each window, recursively re-summarize the partial summaries until output fits under `summary_max_tokens`. The pipeline plugs this in front of every entity/relation upsert.
- **Code surface.** `summarize.go` (decision tree + recursive reduce 200 LOC), `summarize_prompt.go` (verbatim upstream summary system prompt 40 LOC), `merge.go` (entity/relation merge that calls summarize 120 LOC), `main.go` (demo: 12 fake descriptions for one entity, print before/after 60 LOC), `summarize_test.go`.
- **Tests.** `TestSummarizeShortListSkipsLLM`, `TestSummarizeLongListInvokesLLM`, `TestSummarizeRespectsBudget`, `TestSummarizeRecursesUntilFits`, `TestSummarizeEmptyDescriptionsReturnsEmpty`, `TestMergeEntityAggregatesSourceIDs`.
- **Upstream ref.** `lightrag/operate.py:167-303` `_handle_entity_relation_summary` + `:304-385` `_summarize_descriptions`.
- **What changed vs. s09.** Extraction's "merge" step now calls `summarize.go` instead of naive concat. Description quality measurably improves (test asserts the LLM-merged description is shorter than the raw concat).

### s11 — query-modes

- **Problem.** All ten previous sessions built the **ingestion** pipeline; nothing yet exploits the graph at query time. This is where LightRAG actually wins on benchmarks: dual-level retrieval that combines vector chunks (low-level) with KG traversal (high-level) into a single context window. Without this chapter, we have GraphRAG-the-data-structure but not GraphRAG-the-retrieval-strategy.
- **Solution.** Implement four `QueryMode` strategies, each compiling a context window inside a `MaxTotalTokens` budget then handing off to `Provider.Complete`:
  - **Naive** — `vdbChunks.Query(queryEmbed, TopK)` only; no graph (this is the s01 baseline, formalized).
  - **Local** — entity-centric: extract low-level keywords from query (one LLM call), `vdbEntities.Query(keywordEmbed, TopK)`, expand via `graphStore.GetSubgraph(entity, depth=1)`, gather connected chunks via `SourceIDs`.
  - **Global** — relation-centric: extract high-level keywords, `vdbRelations.Query(...)`, expand to connected entities, gather chunks.
  - **Hybrid** — runs both local and global retrieval, deduplicates chunks, applies token budgeting (`MaxEntityTokens` + `MaxRelationTokens` + remainder for chunks), then one LLM call.
  All modes share a `truncate_by_tokens` helper (port of upstream's per-section budget logic). Optional reranker (cosine on chunk embeddings vs query embedding) before final synthesis.
- **Code surface.** `query.go` (router + QueryParam handling 100 LOC), `query_naive.go` (50 LOC), `query_local.go` (180 LOC), `query_global.go` (180 LOC), `query_hybrid.go` (140 LOC), `keywords.go` (LLM keyword extraction with high/low split 100 LOC), `context_builder.go` (token-budgeted assembly 150 LOC), `main.go` (CLI runs all 4 modes on the same query and prints answers + citations 100 LOC), `query_test.go`.
- **Tests.** `TestQueryNaiveReturnsTopChunks`, `TestQueryLocalUsesLowLevelKeywords`, `TestQueryGlobalUsesHighLevelKeywords`, `TestQueryHybridDedupesChunks`, `TestQueryRespectsMaxTotalTokens`, `TestQueryReturnsCitedChunkIDs`, `TestKeywordExtractionParsesBothLevels`.
- **Upstream ref.** `lightrag/operate.py:3164-3410` `kg_query` orchestrator + `:3516-4055` retrieval helpers + `:4930-5200` `naive_query`.
- **What changed vs. s10.** Pipeline now exposes `Query(q, QueryParam) (QueryResult, error)` with all four modes; this is the chapter where the learner finally **compares** modes and sees why hybrid wins on multi-hop questions.

### s_full — Integration

- **Architecture diagram.** ASCII / Mermaid showing: `Insert(text)` → DocStatus → Chunker → Embedder → VectorStores ×3 → Extractor → Summarizer → GraphStore. `Query(q, mode)` → KeywordExtractor → VectorStores ×3 → GraphStore.GetSubgraph → ContextBuilder → Provider.Complete → QueryResult.
- **Body = 16-step trace from research-notes.md L470-487**, but each step links to the session that owns the corresponding Go file. Step 1 (`LightRAG.insert`) → s01 + s09 pipeline.go. Step 5 (chunking) → s04. Step 7 (extract_entities) → s09. Step 12 (hybrid keyword + vector search) → s11. And so on.
- **No new code.** All earlier sessions composed in one runnable demo, with the wiring shown side-by-side as: "here's the s01 sketch / here's the s11 production version".
- **What learners get.** A mental map locking every Go file we wrote into the upstream Python file it shadows.

### Appendix A — Prompt-engineering secret sauce

- **Topic.** Why LightRAG's extraction prompt + gleaning + dual-level keywords + summarization-on-merge are the secret sauce. This is the single biggest factor behind the EMNLP'25 numbers, and it's easy to miss because the code looks like "just call the LLM".
- **Why appendix not chapter.** It's design philosophy without code. Each topic below references the chapter that ships the corresponding Go code, but the appendix's job is to explain **why** these choices win — the mental model that survives even if you re-implement in another language.
- **Sections (six, ~150 words each).**
  1. Why entity extraction is delimiter-based, not JSON. Robustness to LLM format errors; `<|#|>` is unlikely to appear in natural text; cheap to parse with `strings.Split`. Discusses trade-off vs structured output mode.
  2. The gleaning loop: why N rounds beats one big prompt. Recall vs precision; LLMs hit a "stopping condition" too early on entity extraction; continuation prompts force them to look harder.
  3. Description summarization: why merging via LLM beats concatenation past a threshold. Token bloat, redundancy, retrieval-time confusion. The threshold itself is folklore — covers our empirical pick.
  4. Dual-level keywords (high-level "themes" vs low-level "entity names") and why they enable mode routing. High-level keywords drive global mode; low-level drive local mode; without the split, neither mode beats naive RAG by enough to matter.
  5. The cost-quality trade-off. Each gleaning round is one LLM call per chunk; full extraction on a 100-chunk doc with 2 gleaning rounds is 300 calls. When to crank up gleaning vs accept lower recall.
  6. What we deliberately omitted. Reranking (cross-encoder) and streaming responses are noted as Phase G / extension exercises; they're orthogonal to the core insight that good prompts > clever code.

### Appendix B — Upstream map

- **Reading order.** A suggested traversal of upstream Python source: `__init__.py` → `base.py` (read top to bottom) → `operate.py:1-400` → `kg/json_kv_impl.py` → `kg/nano_vector_db_impl.py` → `kg/networkx_impl.py` → `operate.py:400-3200` (extraction + summarization) → `lightrag.py` (the orchestrator that ties them together) → `operate.py:3200-5200` (query modes) → `llm/openai.py`.
- **Per-file → which session(s) reference it.** Reuses the file-to-session table from research-notes.md, expanded with our final chapter numbers:
  - `lightrag/base.py:71` (TextChunkSchema) → s04
  - `lightrag/base.py:77` (QueryParam) → s11
  - `lightrag/base.py:189` (BaseVectorStorage) → s07
  - `lightrag/base.py:308` (BaseKVStorage) → s05
  - `lightrag/base.py:333` (BaseGraphStorage) → s08
  - `lightrag/base.py:662-697` (DocStatus, DocProcessingStatus) → s03
  - `lightrag/operate.py:102-166` (chunking_by_token_size) → s04
  - `lightrag/operate.py:167-385` (summarization) → s10
  - `lightrag/operate.py:~2883-3163` (extract_entities) → s09
  - `lightrag/operate.py:3164-3410` + `:4930-5200` (query modes) → s11
  - `lightrag/kg/json_kv_impl.py` → s05
  - `lightrag/kg/json_doc_status_impl.py` → s03
  - `lightrag/kg/nano_vector_db_impl.py` → s07
  - `lightrag/kg/networkx_impl.py` → s08
  - `lightrag/llm/openai.py` first 120 lines → s02 + s06 + Phase G addendum
  - `lightrag/lightrag.py:1-150` (dataclass) + `:2023+` (insert) + `:2601+` (query) + `:2884-2970` (aquery_llm) → s11 + s_full
  - `lightrag/prompt.py` (PROMPTS dict) → s09 + s10 + Appendix A
  - `lightrag/api/lightrag_server.py` → noted in Appendix B as "extension: Go HTTP server, see exercise list"
- **Suggested extension exercises (five).**
  1. Port a Neo4j storage backend by implementing `GraphStore` against `neo4j-driver-go`. ~300 LOC; reuses everything from s08 except the impl file.
  2. Add streaming responses (Server-Sent Events) to s11's `Query` so the LLM tokens arrive incrementally — wire `Stream=true` through `Provider.Complete`.
  3. Add a reranker provider — interface `Reranker.Score(query string, candidates []string) []float32` — and slot it between context retrieval and synthesis in s11.
  4. Port the s05 KV store to PostgreSQL using `pgx`, demonstrating the same `KVStore` interface scales to a real DB. Includes a migration script.
  5. Build a tiny HTTP server (`net/http` only, no framework) that exposes `POST /insert` and `POST /query` over the s11 pipeline. ~150 LOC; this is the LightRAG-API equivalent.

## Risks & open questions

1. **Tiktoken in Go.** Pick `github.com/pkoukk/tiktoken-go` v0.1.7+; pin in `s04/go.mod`. Risk: cl100k_base loading from network on first call. Mitigation: vendor the BPE merges file under `s04/testdata/cl100k_base.tiktoken` and load via `tiktoken.NewTokenizerWithVocab(path)`.
2. **NanoVectorDB equivalent.** No direct Go port. We implement an in-memory cosine-similarity index in s07 (~250 LOC including persistence). Limitation: O(N·d) query time. We document this and link the brute-force-IVF appendix exercise.
3. **NetworkX equivalent.** Likewise no direct port. We implement an undirected adjacency-map graph in s08 (~350 LOC including BFS subgraph). Limitation: no GraphML export (we use JSON). Discussed in Appendix B exercise list.
4. **OpenAI dependency in s01.** Real OpenAI calls require an API key, which is a non-starter for CI. Mitigation: ship `-provider mock` flag for offline runs (deterministic fake embeddings via input hash + canned LLM responses keyed by request hash). CI matrix runs ONLY mock; local development uses real OpenAI. The mock implementation lives in `s02/provider_mock.go` and is re-imported (via copy, not import — sessions are isolated) by every later chapter that calls a Provider.
5. **operate.py is 5K LOC.** WebFetch will time out / truncate on whole-file requests. Subagents writing s09-s11 MUST fetch in 100-200 line slices using GitHub line-anchor URLs (`?L102-L166`), or use the `gh api` raw-content path. Each chapter's `upstream-readings/sNN-<file>.py` extract should be ≤ 300 lines.
6. **Async semantics gap.** Python's asyncio cooperative concurrency does not map 1:1 to Go's goroutines. For sessions that exercise concurrency (s05 per-key locking, s09 semaphore-bounded extraction), we explicitly contrast: upstream uses `asyncio.Semaphore`; we use a buffered channel. Documented in s09's README as a sidebar.
7. **Provider interface stability.** Phase G adds Anthropic/Bedrock/Ollama AFTER s11 ships. If those providers need extra fields (e.g. Anthropic's `system` parameter is a separate top-level field, not a message), the `CompleteRequest` shape might need to change. Mitigation: the catalog already includes a top-level `System string`; any provider-specific knobs go in a `map[string]any` `ProviderOptions` field added in Phase G if needed (does NOT break existing impls).

## Naming conventions

- Folder per session: `agents/sNN-<slug>/` (e.g. `agents/s04-chunking/`).
- Doc files (bilingual): `docs/zh/sNN-<slug>.md`, `docs/en/sNN-<slug>.md`.
- Upstream reading extracts: `upstream-readings/sNN-<slug-prefix>.py` (Python source slice with annotations; ≤ 300 lines per file; exact line ranges cited).
- Go module per session: `module learn-lightrag/sNN` in each `agents/sNN-<slug>/go.mod`. No cross-session imports — every session is self-contained.
- Each session has `package main` so it's runnable via `cd agents/sNN-<slug> && go run . [args...]`.
- Test files: `*_test.go` co-located with the file under test. Test data: `testdata/`.
- Common helper file naming across sessions: `provider.go`, `provider_openai.go`, `provider_mock.go`, `embedder.go`, `chunking.go`, `tokenizer.go`, `kv_store.go`, `kv_json_store.go`, `vector_store.go`, `cosine_index.go`, `graph_store.go`, `adjacency_graph.go`, `extraction.go`, `summarize.go`, `query.go`, `query_<mode>.go`, `pipeline.go`, `main.go`. Every learner sees the same filenames re-appearing; this is the core didactic move.

## Quality bar

This plan is detailed enough that a fresh agent reading just `plan.md` + `research-notes.md` can write any one session correctly. Per-session entries spell out the four pieces a writer needs: Problem, Solution, Code surface (specific filenames + LOC), Tests (concrete names), Upstream ref (file + lines). The shared types catalog gives the canonical Go shapes that every session re-types verbatim, so cross-session consistency is mechanical, not creative. No placeholder tokens — every cell is decided.
