---
title: "s11 · Dual-level retrieval, four modes"
chapter: 11
slug: s11-query-modes
est_read_min: 14
---

# s11 · Dual-level retrieval, four modes

> What this chapter does: s01-s10 built the entire **ingestion** pipeline — chunks, embeddings, KV, vectors, graph, extraction, summarization. None of that is yet used at *query* time. s11 closes the loop. Four `QueryMode` strategies (naive / local / global / hybrid) each compile a context window inside a `MaxTotalTokens` budget, then hand off to `Provider.Complete`. The hybrid mode is LightRAG's signature dual-level retrieval — running both an entity-centric local pass and a relation-centric global pass, deduping chunks, and making ONE final synthesis call. Mirrors upstream `kg_query` (`lightrag/operate.py:3164-3410`) + `naive_query` (`:4930-5200`).

---

## Problem / 问题

After ten chapters, the data we built is impressive: a knowledge graph with entities, relations, three vector indices (chunks, entities, relations), a KV store with chunk content. But every chapter so far stopped at *insert*. Nothing yet *queries* anything.

The naive answer — "embed the question, find the closest chunks, hand them to the LLM" — is what s01 already did. It's also what plain RAG does. **It loses on every benchmark question that needs more than one hop.**

Three concrete failure modes naive vector RAG hits:

1. **Synonym blindness.** The doc says "the chief archivist resigned"; the question says "who left their post?". Embedding similarity cosines them at 0.4 — below the threshold, they don't match. A graph that linked the entity "chief archivist" to a relation "resignation" would surface the chunk via either path.

2. **Theme questions miss entities.** "What conflicts arose around the codex?" needs to find chunks tagged with conflict/dispute/inquiry — *relations* between entities, not entities themselves. Chunk-level cosine similarity will surface the most-overlapping word-for-word, which is rarely the relation-bearing chunk.

3. **Multi-hop questions need joins.** "Who verified the manuscript that Hartwell discovered?" requires (a) find Hartwell's discoveries → (b) for each, find verifiers. A vector hit on "Hartwell" gets you part one; you still need the graph for part two.

The fix has been sitting unused since s08: **traverse the graph at query time**. But how should we traverse — from which seeds, with which budget, and how do we keep the prompt small enough that the LLM can actually consume it?

## Solution / 解决方案

Four modes, sharing the same `Pipeline.Query(ctx, q, param)` entry point. Each mode is the answer to a different shape of question:

- **Naive** — vector similarity over chunks only; no graph. Use when the question quotes the answer word-for-word. The s01 baseline, formalized so we can A/B against it.
- **Local** — entity-centric. Extract LOW-level keywords (entity names) from the query → `VDBEntities.Query` → for each top entity, expand a one-hop subgraph → gather the chunks those entities/edges came from. Use when the question names a thing and asks about it.
- **Global** — relation-centric. Extract HIGH-level keywords (themes / question types) → `VDBRelations.Query` → expand to the relations' endpoint entities → gather chunks via relation `SourceIDs`. Use when the question is thematic.
- **Hybrid** — runs BOTH local and global retrievals, dedupes chunks, applies token budgeting (entity + relation + remainder for chunks), then ONE final LLM call. Use when you don't know which shape the question is.

The dual-level keyword design is the unsung hero: with one LLM call, we extract two arrays from the query — `high_level_keywords` (drives global) and `low_level_keywords` (drives local). Hybrid mode pays for that one extraction once and uses both arrays. This is the architectural move that lets `local` and `global` co-exist as orthogonal-but-composable retrieval axes.

A shared `truncateByTokens` helper (port of upstream's per-section budget logic) trims each section independently inside `MaxEntityTokens` / `MaxRelationTokens`; chunks get the remainder under `MaxTotalTokens`. The final prompt is bounded but never starved.

## How It Works / 工作原理

### Mode dispatch

```
Pipeline.Query(ctx, q, param)
  │
  ├── ModeNaive  ──▶ queryNaive   ──▶ VDBChunks ──▶ KV ──▶ Complete
  │
  ├── ModeLocal  ──▶ queryLocal   ──▶ extractKeywords (LL only)
  │                                ──▶ VDBEntities ──▶ Graph.GetSubgraph
  │                                ──▶ collect chunk IDs ──▶ KV ──▶ Complete
  │
  ├── ModeGlobal ──▶ queryGlobal  ──▶ extractKeywords (HL only)
  │                                ──▶ VDBRelations ──▶ endpoint entities
  │                                ──▶ collect chunk IDs ──▶ KV ──▶ Complete
  │
  └── ModeHybrid ──▶ queryHybrid  ──▶ extractKeywords (HL + LL once)
                                   ──▶ retrieveLocal  (no LLM)
                                   ──▶ retrieveGlobal (no LLM)
                                   ──▶ merge + dedup chunks
                                   ──▶ ONE Complete call
```

### Naive mode (the baseline)

```
                     ┌─────────────────┐
   query string ────▶│  Embedder.Embed │
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ VDBChunks.Query │  topK=ChunkTopK, threshold=-1
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │   KV.GetByIDs   │
                     └────────┬────────┘
                              ▼
              ┌───────────────────────────────┐
              │  buildSystemPrompt(           │
              │    nil entities,              │
              │    nil relations,             │
              │    chunks,                    │
              │    q, param )                 │
              └───────────┬───────────────────┘
                          ▼
                  Provider.Complete
```

Naive is what every plain-RAG codebase ships. It's our floor; everything else has to beat this.

### Local mode (entity-centric)

```
   query
     │
     ├──▶ extractKeywords ──▶ (HL ignored, LL used)
     │
     ▼
   for each LL keyword:
     Embedder.Embed(keyword)
     VDBEntities.Query(vec, TopK)
     │
     ▼
   dedupHits(entity hits)
     │
     ▼
   for each top entity:
     Graph.GetSubgraph(entity_id, depth=1, maxNodes=TopK)
     │
     ▼
   collect entSet, relSet
     │
     ▼
   chunkIDs = unique(SourceIDs(entSet) ++ SourceIDs(relSet))[:ChunkTopK]
     │
     ▼
   KV.GetByIDs(chunkIDs) ──▶ buildSystemPrompt ──▶ Complete
```

The didactic point: low-level keywords are entity-shaped because the LLM is told to extract "specific entities, proper nouns, technical jargon". They're cheap to embed (one Embed call per keyword) and they index VDBEntities exactly the way an entity-name embedding wants to be indexed.

### Global mode (relation-centric)

```
   query
     │
     ├──▶ extractKeywords ──▶ (HL used, LL ignored)
     │
     ▼
   for each HL keyword:
     Embedder.Embed(keyword)
     VDBRelations.Query(vec, TopK)
     │
     ▼
   for each relation hit:
     reconstruct Relationship from metadata
     Graph.GetNode(SrcID), Graph.GetNode(TgtID)
     │
     ▼
   collect relSet, entSet
     │
     ▼
   chunkIDs = unique(SourceIDs(relSet) ++ SourceIDs(entSet))[:ChunkTopK]
     │
     ▼
   KV.GetByIDs(chunkIDs) ──▶ buildSystemPrompt ──▶ Complete
```

The relations index has to be seeded at ingestion time with embeddings of `keywords + description`. That's why s10's summarization step matters: the description fed to the embedder *is* the merged-summary description, not the 12-fragment unmerged blob.

### Hybrid mode (the headline)

```
   query
     │
     └──▶ extractKeywords  ──▶  (HL, LL)   ── one call, both arrays
              │
              ├──▶ retrieveLocal  (no LLM)  ──▶ entSet_L, relSet_L, chunkIDs_L
              │
              └──▶ retrieveGlobal (no LLM)  ──▶ entSet_G, relSet_G, chunkIDs_G
              │
              ▼
          merge entSet_L ∪ entSet_G
          merge relSet_L ∪ relSet_G
          chunkIDs = dedup(chunkIDs_L ++ chunkIDs_G)[:ChunkTopK]
              │
              ▼
          buildSystemPrompt(entities, relations, chunks, q, param)
              │
              ▼
          Provider.Complete   ── ONE call, ONE answer
```

Three non-obvious points:

1. **One synthesis call, not two.** Hybrid does NOT run local-then-global and concatenate the two answers. It runs both *retrievals* and feeds the merged context into a single Complete call. Two answers concatenated would have inconsistent voice and contradictory framing; one prompt with both retrievals lets the LLM reconcile internally.
2. **Token budget is per-section, then capped overall.** `MaxEntityTokens` and `MaxRelationTokens` are independent budgets. Chunks get whatever's left of `MaxTotalTokens` after entities + relations are filled. If `MaxEntityTokens + MaxRelationTokens > MaxTotalTokens` already, chunks get nothing — that's the user's fault for over-budgeting.
3. **Chunk dedup is by ID, not by content.** Two chunks with overlapping text but different IDs both go in (they're different chunks of different docs); two retrievals that pull the same chunk-3 see it once. Good enough; content-aware dedup is a Phase G exercise.

### `buildSystemPrompt` (truncate then compose)

```go
// from context_builder.go
entSection := truncateByTokens(renderEntities(entities), param.MaxEntityTokens)
relSection := truncateByTokens(renderRelations(rels),    param.MaxRelationTokens)

used := tokenCount(entSection) + tokenCount(relSection)
chunkBudget := param.MaxTotalTokens - used
if chunkBudget < 0 { chunkBudget = 0 }
chunkRows := truncateChunksByTokens(chunks, chunkBudget)

// then concat: ## Entities / ## Relations / ## Chunks / ## User Query
```

The order matters: entities first, relations second, chunks last. Entities are the most reusable across questions; chunks are the most expensive in tokens. Trimming the tail of the chunks section is the lowest-cost way to stay under budget.

## What Changed / 与 s10 的变化

s10 finished the ingestion polish: 47 description fragments per entity → one merged summary, with a threshold-driven decision tree on whether the LLM is even needed. After s10, every entity in the graph has a clean single-sentence description, and the relations index is populated with embeddings of `keywords + description`.

s11 finally **uses** all of that at query time. Three changes are load-bearing:

1. **`Query` is the new public surface.** Earlier sessions exposed `Insert`, `Upsert`, `Embed` — write-side APIs. s11 introduces `Query(ctx, q, QueryParam) (QueryResult, error)`, which is what callers will actually use. `Pipeline` now reads from every storage; nothing writes.
2. **Dual-level keywords are the routing primitive.** Without the high/low split, neither `local` nor `global` would beat naive enough to matter. With it, we can route a question by *shape*: entity-named questions go to local, theme questions go to global, ambiguous questions go to hybrid.
3. **Token budgeting is per-section.** Earlier sessions had no token-budget concept (each function dealt with one chunk or one entity). s11's three-section prompt with three budgets is the first time the codebase has to reason about a holistic prompt size.

## Try It / 动手试一试

```bash
cd agents/s11-query-modes

# Run all four modes side by side on the same question.
go run . -mode all -q "Who discovered the Hartwell Codex?"

# Run just one mode.
go run . -mode hybrid -q "What conflicts arose around the codex?"
go run . -mode local  -q "Where does Eleanor Hartwell work?"
go run . -mode global -q "What kinds of academic verification has Hartwell pursued?"

# Tests (offline, deterministic, ~0.3s).
go test -count=1 -v ./...
```

Expected output for `-mode all`:

```
=== s11: query="Who discovered the Hartwell Codex?" ===

[naive]
  references: [chunk-2 chunk-3 ...]
  answer    : Eleanor Hartwell ...
[local]
  references: [chunk-1 chunk-2 chunk-6 chunk-4 chunk-5]
  answer    : Eleanor Hartwell ...
[global]
  references: [chunk-1 chunk-4 chunk-5 ...]
  answer    : Eleanor Hartwell ...
[hybrid]
  references: [chunk-1 chunk-2 chunk-6 chunk-4 chunk-5]
  answer    : Eleanor Hartwell ...
```

Compare the `references` lists across modes — naive ranks differently from local because naive measures cosine on chunk content, while local ranks by which chunks the top entities (and their subgraph) cite. Hybrid contains the union (deduped).

## Upstream Source Reading / 上游源码阅读

The full annotated excerpt lives at [`upstream-readings/s11-query.py`](../../upstream-readings/s11-query.py). The load-bearing slice — kg_query mode dispatch — is reproduced here in ≤ 50 LOC:

```python
# operate.py:3164-3232  (kg_query, mode dispatch + keyword fallback)
async def kg_query(
    query, knowledge_graph_inst, entities_vdb, relationships_vdb,
    text_chunks_db, query_param, global_config, hashing_kv=None,
    system_prompt=None, chunks_vdb=None,
) -> QueryResult | None:
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    # ONE LLM call extracts BOTH high-level and low-level keywords.
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv,
    )
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            ll_keywords = [query]              # seed with the query
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    context_result = await _build_query_context(
        query, ll_keywords_str, hl_keywords_str,
        knowledge_graph_inst, entities_vdb, relationships_vdb,
        text_chunks_db, query_param, chunks_vdb,
    )
    if context_result is None:
        return None

    # [omitted: cache, only_need_context, only_need_prompt, streaming branches]

    sys_prompt = PROMPTS["rag_response"].format(
        response_type=query_param.response_type or "Multiple Paragraphs",
        user_prompt=query_param.user_prompt or "n/a",
        context_data=context_result.context,
    )
    response = await use_model_func(query, system_prompt=sys_prompt, ...)
    return QueryResult(content=response, raw_data=context_result.raw_data)
```

Three things to notice when reading:

- **Mode dispatch is implicit.** `kg_query` itself doesn't have a `match query_param.mode:` block; the dispatch is hidden inside `_build_query_context` (operate.py:3700+), which branches by mode and produces different `(entities, relations, chunks)` tuples. Our Go port externalizes that branch into `query_local.go` / `query_global.go` / `query_hybrid.go` for readability.
- **The "no keywords" fallback.** When the LLM returns empty arrays for both, upstream falls back to using the query *itself* as a low-level keyword (when short) or fails out (when long). We replicate this in `query_local.go` / `query_global.go` (`if len(keys) == 0 { keys = []string{q} }`).
- **Naive mode is a peer, not a sibling.** `naive_query` is a separate top-level function (operate.py:4953); upstream has no superclass relating naive to kg_query. Our `Pipeline.Query` dispatch unifies them at the API level, which is closer to how a library user would actually want it.

For the deeper trace of how `_build_query_context` differs across local / global / hybrid, see the `s_full` integration chapter — it walks the full insert→query path with line-anchor citations into upstream.
