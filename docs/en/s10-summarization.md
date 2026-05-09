---
title: "s10 · Map-reduce description summarization"
chapter: 10
slug: s10-summarization
est_read_min: 9
---

# s10 · Map-reduce description summarization

> What this chapter does: when the same entity shows up in 47 chunks, we have 47 independent description fragments on hand. Plain concatenation blows the embedding model's token limit; one big LLM call gives a vague summary that confuses retrieval-time semantics. s10 mirrors upstream `_handle_entity_relation_summary` + `_summarize_descriptions` (lightrag/operate.py:167-385): a threshold-driven decision tree that takes the cheap path on small inputs and the recursive map-reduce LLM path on big inputs.

---

## Problem / 问题

s09's Extractor only sees one chunk at a time. Each `Entity{Name:"Scrooge", Description:"..."}` it produces carries the description from that one chunk only. But the same Scrooge may appear in 47 chunks of one doc — so the graph store ends up with 47 Scrooge records, each with a different description.

"Just merge them" — but how? Four dead ends:

1. **Plain concatenation: tokens explode.** 47 descriptions × 30 tokens = 1410 tokens; OpenAI's `text-embedding-3-small` can absorb up to 8192. Push descriptions to 200 tokens each and you're at 9400, blown. Even when it fits, the vector gets diluted by noise — retrieval recall drops.
2. **Keep only the longest: lose information.** "Scrooge is a businessman" and "Scrooge meets Marley's ghost on Christmas Eve" are both load-bearing; the longest sentence may be unrelated. You drop key facts.
3. **Always use the LLM: cost spirals + degrades.** Calling GPT-4 for every entity at every ingest is wasteful — when "Scrooge appears 2 times", concatenation suffices; only "Scrooge appears 47 times" actually needs synthesis.
4. **JSON-schema merging: LLMs are bad at it.** Forcing the LLM to emit `{"merged":"..."}` underperforms free-form prose, same root cause as s09: JSON delimiters trip the LLM into dropping a `}` and ruining the whole output.

LightRAG's elegant move: a **dual-threshold decision tree** — `(count < threshold AND total < budget) → concat, otherwise → recursive LLM merge`. Both thresholds matter: count alone fails when 4 descriptions are 2k tokens each; budget alone fails when 30 short descriptions confuse retrieval semantically.

s10 ports this decision tree to Go and uses a `tokenCount()` stub (whitespace word count — the real cl100k_base lives in s04) to keep the module isolated.

## Solution / 解决方案

Four pieces:

1. **`SummarizeDescriptions(ctx, p Provider, descriptions, budgetTokens, contextSize, countThreshold)` is the single entry point.** Returns `(summary, llmUsed, err)`. Provider has the same shape as s02 / s09 (re-declared in every session — no imports between sessions).
2. **Decision tree (direct port of upstream operate.py:222-226):** `len==0` → `("", false, nil)`; `len==1` → return as-is; `len < threshold AND total < budget` → `\n\n`-join; otherwise → map-reduce.
3. **Map phase:** split descriptions into windows of `contextSize` tokens, **at least 2 descriptions per window** (upstream's "ensure progress" rule at operate.py:255-263 — a single-description window would just pass through and we'd loop forever).
4. **Reduce phase:** each multi-description window invokes `Provider.Complete(summaryPrompt + JSONL)`; single-description windows pass through; the partial summaries are concatenated, and if they still exceed budget we recurse (capped at `DefaultMaxRecursionDepth=3`; upstream's `while True` is replaced with a depth limit so test costs don't run away).

`MergeEntities` and `MergeRelationships` are caller-side glue: group by Name (entities) or canonical(min,max) edge (relationships), call `SummarizeDescriptionsForName` to merge fragments, return one combined record with SourceIDs unioned and sorted.

## How It Works / 工作原理

```
                     SummarizeDescriptions(ctx, p, descs, budget, ctxSize, threshold)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │   Step 1  edge cases:                                                   │
   │             len(descs) == 0  →  ("", false, nil)                        │
   │             len(descs) == 1  →  (descs[0], false, nil)                  │
   │                                                                         │
   │   Step 2  cheap path:                                                   │
   │             if len(descs) < countThreshold &&                           │
   │                totalTokens(descs) < budgetTokens:                       │
   │                  return strings.Join(descs, "\n\n"), false, nil         │
   │                                                                         │
   │   Step 3  map: chunk descs by contextSize, ensuring each chunk has ≥2   │
   │                                                                         │
   │   Step 4  reduce: for each chunk:                                       │
   │             if len(chunk) == 1: pass-through (no LLM)                   │
   │             else: Provider.Complete(prompt + JSONL of chunk)            │
   │                                                                         │
   │   Step 5  converge:                                                     │
   │             if len(partials) == 1                  →  return            │
   │             if joined fits budget AND under thresh →  return joined     │
   │             if depth+1 >= MaxRecursionDepth         →  one final LLM    │
   │             else: recurse on partials (depth+1)                         │
   │                                                                         │
   └─────────────────────────────────────────────────────────────────────────┘

   MergeEntities/MergeRelationships:
       group by Name (or canonical edge) ──► SummarizeDescriptionsForName ──► merged
       SourceIDs = union(sources), sorted
       Weights summed (relationships only)
```

[`agents/s10-summarization/summarize.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s10-summarization/summarize.go), the core decision path (~30 LOC):

```go
// Step 1: edge cases.
if len(descriptions) == 0 {
    return "", false, nil
}
if len(descriptions) == 1 {
    return descriptions[0], false, nil
}

// Step 2: heuristic — under threshold AND under budget → naive concat.
total := totalTokens(descriptions)
if len(descriptions) < countThreshold && total < budgetTokens {
    return strings.Join(descriptions, descriptionJoinSeparator), false, nil
}

// Step 3: map — chunk into contextSize-token windows.
chunks := chunkByTokens(descriptions, contextSize)

// Step 4: reduce — single-description chunks pass through.
llmUsed := false
partials := make([]string, 0, len(chunks))
for _, chunk := range chunks {
    if len(chunk) == 1 {
        partials = append(partials, chunk[0])
        continue
    }
    summary, err := callLLMSummary(ctx, p, descriptionType, descriptionName, chunk, budgetTokens)
    if err != nil { return "", llmUsed, fmt.Errorf("summarize chunk: %w", err) }
    partials = append(partials, summary)
    llmUsed = true
}

// Step 5: recurse if still too big.
joined := strings.Join(partials, descriptionJoinSeparator)
if tokenCount(joined) < budgetTokens && len(partials) < countThreshold {
    return joined, llmUsed, nil
}
return summarizeWithName(ctx, p, descriptionType, descriptionName, partials, ..., depth+1)
```

**Four non-obvious points:**

1. **Threshold-AND-Budget** — both must be satisfied to take the concat path. Count alone fails (4 × 200-token descriptions still blow the budget); budget alone fails (30 short descriptions still confuse retrieval). Upstream operate.py:222-226 locks them together with `AND`.
2. **Each chunk has at least 2 descriptions** — upstream's "ensure progress" rule at operate.py:255-263. If a chunk has only 1 description, the reduce phase passes it through, the next iteration sees the same size, and we loop. We force-append one more so the chunk has 2.
3. **Single-description chunks skip the LLM** — upstream's `if len(chunk) == 1: pass-through` optimization; merging one description is the identity operation, no need to spend tokens.
4. **Recursion-depth cap** — upstream uses `while True`, theoretically valid because each reduce round strictly compresses. But a mock provider that returns a fixed-length response (test scenarios) doesn't shrink — so we cap at `DefaultMaxRecursionDepth=3` and run one final LLM call to flatten the partials.

## What Changed / 与 s09 的变化

s09's Extractor produces one `Entity` record per chunk (with one SourceID). s10 is the consumer: collapse N same-name Entities into 1, merge descriptions, union SourceIDs.

| Dimension | s09 (extraction) | s10 (summarization) |
|---|---|---|
| Input | One chunk's text | N description fragments (one per chunk) |
| Output | `[]Entity, []Relationship` (each carries 1 SourceID) | 1 merged Entity/Relationship (SourceIDs unioned) |
| LLM trigger | Always (initial + N gleaning) | Conditional: count >= threshold OR total > budget |
| Map-reduce | No | Yes — chunk → reduce → recurse |
| Cache | sha256(chunk + ver) | (Caller's responsibility; upstream uses cache_type="summary") |
| Upstream cite | operate.py:2883-3170 + prompt.py | operate.py:167-385 + prompt.py:185-218 |

**Why is caching mandatory in production (even though s10 doesn't ship it)?** Because if 2 of 47 descriptions changed, you should only re-summarize the window containing those 2, not re-run the whole entity. Upstream's `cache_type="summary"` namespace exists exactly for this. s10 leaves it as caller homework — wrap `Provider.Complete` with a memoizer.

**Why default budget 500 / threshold 6?** Upstream's `summary_max_tokens=500` and `force_llm_summary_on_merge=6` config defaults. 500 is the empirical ceiling where description embeddings stay stable in OpenAI's vector space; above that they dilute. 6 is folklore: ≤5 fragments are usually different phrasings of the same fact, so concat is fine.

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s10-summarization

# Default mock provider (offline; what CI uses)
go run .

# -provider openai is intentionally unwired — s02 owns the real one
# go run . -provider openai     # → returns clear error

# Run tests (6 spec + interface contract + 2 sanity)
go test -v ./...
```

Demo output (excerpt):

```
=== s10 demo: Eleanor Hartwell appears in 12 chunks ===
[a] raw concatenation (what s09 would have stored):
  ... 12 description fragments, 188 words ...
[b] s10 summarized version:
  The merged subject ... professional accomplishments ...
[c] llmUsed: true
[d] compression: 188 → 66 tokens (35% of original)

=== fast-path demo: 3 descriptions, plenty of budget ===
llmUsed: false (expect false — under threshold AND under budget)
output is exact concat? true
```

Test matrix:

| Test | Asserts |
|---|---|
| `TestSummarizeShortListSkipsLLM` | 3 descs + budget=500 → llmUsed=false, output is exact `\n\n`-joined concat |
| `TestSummarizeLongListInvokesLLM` | 12 descs + budget=60 → llmUsed=true, MockProvider.Calls > 0 |
| `TestSummarizeRespectsBudget` | Long input compresses; final tokens < 2× budget |
| `TestSummarizeRecursesUntilFits` | 30 descs + very tight budget → MockProvider.Calls > 1 (recursion) |
| `TestSummarizeEmptyDescriptionsReturnsEmpty` | nil / `[]string{}` → `("", false, nil)` |
| `TestMergeEntityAggregatesSourceIDs` | 3 Eleanor records with [c1]/[c2]/[c3] → 1 merged, SourceIDs sorted=[c1,c2,c3] |

Plus 3 sanity tests: `TestProviderInterfaceContract` (compile-time `*MockProvider` satisfies `Provider`), `TestMergeRelationshipsCanonicalizesEdge` (A→B and B→A canonicalize, weights summed), `TestSingleDescriptionPassesThrough` (1-element input → no LLM).

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/operate.py:167-227 (~50 LOC, function head + decision tree)
async def _handle_entity_relation_summary(
    description_type, entity_or_relation_name, description_list,
    separator, global_config, llm_response_cache=None,
) -> tuple[str, bool]:
    """Decision tree:
       1. If total tokens < context_size AND len < threshold → no LLM
       2. Otherwise summarize with LLM, possibly recursively
    """
    if not description_list:
        return "", False
    if len(description_list) == 1:
        return description_list[0], False

    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]
    llm_was_used = False

    while True:
        total_tokens = sum(len(tokenizer.encode(d)) for d in current_list)

        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens):
                # no LLM needed, just join
                return separator.join(current_list), llm_was_used
            else:
                final_summary = await _summarize_descriptions(...)
                return final_summary, True

        # Map phase: chunk descriptions into context_size windows;
        # each chunk has >= 2 descriptions to ensure progress.
        chunks = ...  # see operate.py:240-285

        # Reduce phase: for each chunk, len==1 pass-through, else LLM.
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                new_summaries.append(chunk[0])
            else:
                summary = await _summarize_descriptions(...)
                new_summaries.append(summary)
                llm_was_used = True
        current_list = new_summaries  # loop and re-evaluate
```

**How to read this:**

- `force_llm_summary_on_merge` is an awkward name for what is really "count threshold" — the minimum number of descriptions before merge MUST go through the LLM. We call it `countThreshold` in Go for clarity.
- `summary_max_tokens` vs `summary_context_size`: the former is the **final** output budget (s10's `budgetTokens`); the latter is the **per-map-window** input size (s10's `contextSize`). Two parameters with different roles, easily confused.
- `while True` looks scary but each reduce round strictly compresses (multi → single), so it terminates. Our Go port adds `DefaultMaxRecursionDepth=3` because the mock provider in tests can return fixed-length responses that don't shrink.
- `cache_type="summary"` is the namespace inside upstream's `llm_response_cache` (separated from extract / query). We don't ship caching in s10 (caller's responsibility), but the cache key should always include `description_name + summaryPromptVersion` — otherwise editing the prompt won't invalidate stale entries.

Annotated extract + reading map: [`upstream-readings/s10-summarization.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s10-summarization.py).

The next chapter — [s11 dual-level retrieval](s11-query-modes.md) — is this chapter's downstream consumer: local mode does `vdbEntities.Query()` over the description embeddings produced by s10. Without summarization that vector index would be diluted by 47 near-duplicate vectors per entity, hurting recall.
