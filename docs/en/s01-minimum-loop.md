---
title: "s01 · Minimum RAG loop"
chapter: 1
slug: s01-minimum-loop
est_read_min: 12
---

# s01 · Minimum RAG loop

> What this teaches: a complete "insert document → ask question → get a cited answer" path in ~1000 lines of Go. This chapter is the scaffold for the next ten — every interface (Provider / EmbeddingProvider / VectorStore / KVStore) is **declared here** so each subsequent chapter can swap one stub for a real implementation without changing call sites.

---

## Problem

The first thing that pushes learners away from upstream [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) is "where do I even start?" — the `lightrag/` package has 30+ files, `lightrag.py` alone is 3000 lines, `operate.py` is 5000, and the code is full of `dict[str, Any]`, dual sync/async APIs, and a 60-field `@dataclass`. The only question a learner actually wants answered is: **"how does this thing turn a chunk of text into a system that can answer questions?"**

The s01 problem: before we touch token-based chunking, state machines, knowledge graphs, or entity extraction, get the **whole loop** running. After `go run . -q "..."` the learner sees a real answer with chunk citations. That loop becomes the baseline against which every later chapter is measured.

## Solution

Decompose RAG into 5 minimal stages: **chunk** → **embed** → **store** → **retrieve** → **complete**. Each stage gets a stub implementation plus an explicit interface, so subsequent chapters can swap one piece without touching the others.

3 key design decisions:
1. **Interface before implementation**: `Provider`, `EmbeddingProvider`, `VectorStore`, `KVStore` are all declared as Go interfaces; the s01 implementations are `sync.Map` and slice-scan cosine. s05 swaps KV for JSON-on-disk; s07 swaps VectorStore for a real index. Call sites never change.
2. **Mock-first for offline CI**: CI cannot depend on OpenAI, so s01 ships a `MockProvider` (deterministic echo) and `MockEmbedder` (sha256 hash → unit vector). Real OpenAI runs only when `OPENAI_API_KEY` is set.
3. **`chunkID` is the contract**: `<docID>::chunk-<index>` doubles as the KV key, the VDB ID, and the final `References` output. Every chapter from here preserves this convention.

## How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│                     s01 minimum RAG loop (~1000 LOC)                   │
│                                                                        │
│   Insert(docID, text)                                                  │
│         │                                                              │
│         ▼                                                              │
│   ChunkByNewlines  ──→  Embedder.Embed  ──→  VDB.Upsert  +  KV.Upsert  │
│                          (mock|openai)      (in-mem cosine) (sync.Map) │
│                                                                        │
│   Query(q, topK=3)                                                     │
│         │                                                              │
│         ▼                                                              │
│   Embedder.Embed(q) → VDB.Query → KV.GetByIDs → build system prompt    │
│         │                                                              │
│         ▼                                                              │
│   Provider.Complete  ──→  QueryResult{Content, References, Mode}       │
│       (mock|openai)                                                    │
└────────────────────────────────────────────────────────────────────────┘
```

The core 30 lines (excerpt from [`agents/s01-minimum-loop/pipeline.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s01-minimum-loop/pipeline.go)):

```go
func (p *Pipeline) Query(ctx context.Context, q string, topK int) (QueryResult, error) {
    qVecs, err := p.Embedder.Embed(ctx, []string{q})
    if err != nil { return QueryResult{}, err }

    hits, err := p.VDB.Query(ctx, qVecs[0], topK, -1.0) // -1.0 = no threshold in s01
    if err != nil { return QueryResult{}, err }

    ids := make([]string, len(hits))
    for i, h := range hits { ids[i] = h.ID }
    contents, err := p.KV.GetByIDs(ctx, ids)
    if err != nil { return QueryResult{}, err }

    var ctxBuf strings.Builder
    ctxBuf.WriteString("Context:\n")
    for _, id := range ids {
        body, _ := contents[id]["content"].(string)
        fmt.Fprintf(&ctxBuf, "[chunk %s]\n%s\n\n", id, body)
    }
    system := "You are a helpful assistant. Answer using ONLY the context below. " +
              "Cite chunk IDs in [brackets].\n\n" + ctxBuf.String()

    resp, err := p.Provider.Complete(ctx, CompleteRequest{
        System:   system,
        Messages: []Message{{Role: "user", Content: q}},
    })
    if err != nil { return QueryResult{}, err }
    return QueryResult{Content: resp.Text, References: ids, Mode: ModeNaive}, nil
}
```

**4 non-obvious points**:

1. **`threshold = -1` is a placeholder**: s01's VectorStore doesn't filter by similarity — it just returns the top-K. s07 turns the threshold into a real knob (drop hits below it). The `-1.0` here means "no filter".
2. **`Pipeline.KV` is keyed by chunkID**: the same ID lives in both VDB (for similarity retrieval) and KV (for the chunk content). All 11 chapters preserve this 1:1 mapping.
3. **The citation rule is hard-coded in the system prompt**: "Cite chunk IDs in [brackets]" makes the LLM produce text like `…as discussed in [book::chunk-2]…`. This is the hook the mock provider uses to verify "which chunks did the LLM see?", and real OpenAI honors it too.
4. **`References` returns retrieval order verbatim**: no dedup, no rerank. VDB's order is the answer's order. s11 introduces token-budget truncation and rerank-by-relevance.

## What Changed (vs. s00)

s01 is the first chapter — there is no s00 to diff against. What this chapter **adds** to the empty repo are the interfaces and contracts that every later chapter evolves:

- **`Provider` interface**: single method `Complete(ctx, req) (resp, error)`. s02 promotes `OpenAIProvider` to its own file with retry; Phase G adds `AnthropicProvider` and friends — signature never changes.
- **`EmbeddingProvider` interface**: `Embed([]string) → [][]float32` + `Dim() int`. s06 adds batching + retry.
- **`VectorStore` interface**: `Upsert / Query / Delete / Persist`. s07 brings a real index + JSON persistence + thresholding.
- **`KVStore` interface**: `Get / GetByIDs / Upsert / FilterMissing / Delete / Persist`. `FilterMissing` is the primitive that s09 uses to decide "which chunks still need entity extraction".
- **`chunkID = <docID>::chunk-<idx>`** naming convention: all chapters use this ID for cross-store joins.
- **`QueryResult{Content, References, Mode}`** output shape: every chapter's final return is this struct.

Semantically, s01 is a **single-process synchronous pipeline**. After s05 we have real disk persistence; after s09 we have semaphore-bounded extraction. But the **caller-facing interface shape is locked from s01 onward**.

## Try It

```bash
cd agents/s01-minimum-loop

# 1. Fully offline mock — proves "no API key needed"
go run . -provider mock -q "What did Eleanor patent in 1872?"

# 2. -v opens up retrieval debug lines
go run . -provider mock -v -q "Where did Eleanor's papers end up?"

# 3. Real OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
go run . -q "Why was the Tarvin Highlands trip cancelled?"

# 4. Run the tests (this is what CI runs)
go test -v ./...
```

Expected output shape:

```
[s01] ingested doc "testdata/book.txt" (provider=mock)
=== Answer ===
[mock] retrieved 3 chunks (book::chunk-1, book::chunk-0, book::chunk-2)
preview of top chunk: "Eleanor's most famous patent, granted in 1872, covered ..."

=== References ===
- book::chunk-1
- book::chunk-0
- book::chunk-2

[mode=naive]
```

The mock provider's "answer" is the first ~200 chars of the system prompt echoed back — **deterministic**, so CI can stably assert it. Real OpenAI rewrites the answer in natural language and follows the system prompt's instruction to embed `[book::chunk-N]` citations inline.

## Upstream Source Reading

The equivalent path in upstream LightRAG is `lightrag/lightrag.py:1237` (`ainsert()`) plus `:2622` (`aquery()`) plus `:2884` (`aquery_llm()` — the function that does the real work). s01 collapses that path into ~80 lines of `Pipeline.Insert/Query`, dropping doc-status, entity extraction, and the four query modes — those are the next ten chapters.

```upstream:lightrag/lightrag.py#L2884-L2940
async def aquery_llm(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Asynchronous complete query API: structured retrieval + LLM generation."""
    global_config = asdict(self)

    if param.mode in ["local", "global", "hybrid", "mix"]:
        # Graph-aware modes: extract keywords → traverse KG → gather chunks.
        # → s11 implements all of these.
        query_result = await kg_query(
            query.strip(),
            self.chunk_entity_relation_graph,    # ← s08 GraphStore
            self.entities_vdb,                   # ← s07 entities vector index
            self.relationships_vdb,              # ← s07 relations vector index
            self.text_chunks,                    # ← s05 chunk-content KV
            param, global_config,
            hashing_kv=self.llm_response_cache,
            system_prompt=system_prompt,
            chunks_vdb=self.chunks_vdb,
        )
    elif param.mode == "naive":
        # The simple branch: vector similarity only, no graph.
        # → s01 Pipeline.Query mirrors this.
        query_result = await naive_query(
            query.strip(),
            self.chunks_vdb,
            param, global_config,
            hashing_kv=self.llm_response_cache,
            system_prompt=system_prompt,
        )
    elif param.mode == "bypass":
        # No retrieval at all; ask LLM directly. Diagnostic only.
        # → not implemented in s01..s11; mentioned in Appendix A.
        ...
```

**Reading notes**:

- **Mode dispatch vs. single naive branch**: upstream has one function with four mode branches; s01 only implements the `naive_query` equivalent and hard-codes `Pipeline.Query.Mode = ModeNaive`. s11 fills in the `kg_query` branch.
- **`global_config = asdict(self)` is an anti-pattern**: upstream passes the entire 60-field dataclass as a `dict` through every helper. Go would never tolerate this — s01 gives `Pipeline` an explicit struct, and s11 keeps it that way.
- **`hashing_kv=self.llm_response_cache` reuse**: upstream consults an LLM-response cache on every query (avoid repaying for identical calls); s01 omits caching since quickstart queries rarely repeat. s09 introduces the cache (uses the same `KVStore` interface with hash-keyed storage).
- **Streaming**: the `is_streaming` branch returns `AsyncIterator[str]`. s01's `Provider` interface already declares `Stream bool` but always passes `false`. Phase G wires it through.
- **The 16-line `naive_query` core**: it's the Python sibling of s01's `Pipeline.Query` — embed query → `chunks_vdb.query(top_k)` → `text_chunks.get_by_ids(...)` → assemble context → call LLM. You can put those 16 Python lines next to s01's 30 Go lines side-by-side ([upstream-readings/s01-lightrag.py](../../upstream-readings/s01-lightrag.py) does exactly this with annotations).

**Read further**: start at `lightrag/lightrag.py`'s `LightRAG.ainsert()` (L1237), follow `apipeline_process_enqueue_documents` into `lightrag/operate.py` to see the real chunking + extraction loop (~L100-500), then read `lightrag/operate.py:3164`'s `kg_query` for the 4-mode dispatch. That trace is the s01 → s04 → s09 → s11 real-source map.

---

**Next**: s02 promotes the `Provider` interface to its own chapter — adds retry, functional options, three implementations (OpenAI / Mock / Echo). Phase G later layers Anthropic and Bedrock on top of that interface without touching any caller.
