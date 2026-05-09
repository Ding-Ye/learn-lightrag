# s10 — map-reduce description summarization

> 同一个实体可能出现在 47 个 chunk 里，于是有 47 段描述。直接拼接会撑爆 embedding 的 token 上限。s10 用「阈值决策 + 递归 LLM 归并」把这 47 段压缩成一段。
>
> When the same entity appears in 47 chunks, we end up with 47 description fragments. Concatenating blows past the embedding model's token limit. s10 uses a threshold-driven decision tree and a recursive LLM merge to squash N fragments into one.

## What this session teaches / 这一节教什么

After s09 ships per-chunk extraction, the pipeline has many `Entity{Name:"Scrooge", Description:"…"}` records — one per chunk. Without a merge step, the graph store would hold 47 copies of "Scrooge" with 47 different descriptions. s10 collapses them, mirroring upstream `_handle_entity_relation_summary` (lightrag/operate.py:167-303) and `_summarize_descriptions` (operate.py:304-385).

Decision tree:

1. `len(descriptions) == 0` → `("", false, nil)`.
2. `len(descriptions) == 1` → return as-is, no LLM.
3. `len(descriptions) < countThreshold` AND `totalTokens(descriptions) < budgetTokens` → naive concat with `\n\n`, no LLM.
4. Otherwise: chunk into `contextSize`-token windows; LLM-summarize each multi-description window; recurse on partial summaries until result fits or recursion depth maxes out.

## Files / 文件

| Path | Role |
|---|---|
| `summarize.go` | `SummarizeDescriptions` core + decision tree + recursive map-reduce + token-stub |
| `summarize_prompt.go` | Faithful summary of upstream `summarize_entity_descriptions` prompt (cited inline) |
| `merge.go` | `MergeEntities` / `MergeRelationships` — group by name, dispatch to summarizer |
| `main.go` | CLI demo: 12 fake descriptions for "Eleanor Hartwell"; raw vs summarized vs llmUsed flag |
| `summarize_test.go` | 6 spec tests + interface contract + 2 bonus sanity tests |
| `testdata/` | Empty (tests construct strings inline) |

## Run / 跑起来

```bash
cd /Users/yeding/learn-lightrag/agents/s10-summarization

# Default mock provider (offline; what CI uses)
go run .

# -provider openai is intentionally unwired — s02 owns that.
# go run . -provider openai   # → returns clear error

# Run tests
go test -v ./...
```

## Test list / 测试

| # | Test | Asserts |
|---|------|---------|
| 1 | `TestSummarizeShortListSkipsLLM` | 3 descs + budget=500 → llmUsed=false, output is exact `\n\n`-joined concat |
| 2 | `TestSummarizeLongListInvokesLLM` | 12 descs + budget=60 → llmUsed=true, MockProvider.Calls > 0 |
| 3 | `TestSummarizeRespectsBudget` | Long input compresses; final tokens within tolerance of budget |
| 4 | `TestSummarizeRecursesUntilFits` | 30 descs + tight budget → MockProvider.Calls > 1 (recursion) |
| 5 | `TestSummarizeEmptyDescriptionsReturnsEmpty` | nil and `[]string{}` → `("", false, nil)` |
| 6 | `TestMergeEntityAggregatesSourceIDs` | 3 Entity{Name:"Eleanor"} from [c1],[c2],[c3] → 1 Entity, SourceIDs sorted=[c1,c2,c3] |

Plus `TestProviderInterfaceContract` (compile-time `*MockProvider` satisfies `Provider`), `TestMergeRelationshipsCanonicalizesEdge` (A→B and B→A canonicalize to one edge, weights summed), `TestSingleDescriptionPassesThrough` (1-element input → no LLM call).

## What changed vs s09 / 与 s09 的差异

s09 produces ONE description per chunk. s10 collapses N descriptions of the same entity (or undirected edge) into one — using a heuristic to skip LLM when the input is already small enough.

| Dimension | s09 (extraction) | s10 (summarization) |
|---|---|---|
| Input | One chunk's text | N description fragments (one per chunk) |
| Output | `[]Entity, []Relationship` (one per record found) | `Entity` / `Relationship` with merged Description and unioned SourceIDs |
| LLM trigger | Always (1 + N gleaning rounds) | Conditional: only if count >= threshold OR total > budget |
| Map-reduce | No | Yes — chunk into windows, summarize, recurse |
| Cache | sha256(chunk + version) per chunk | (Out of scope — left to caller; upstream uses cache_type="summary") |

**Why both a count threshold AND a token budget?**  Because either alone is wrong:

- Just count: 4 long descriptions might already exceed embedding token limit.
- Just budget: 30 short descriptions still confuse the LLM at retrieval time even if they technically fit.

Upstream uses both in an `AND`: skip LLM only if BOTH are under their thresholds (operate.py:222-226).

## tokenCount stub / token 计数桩

`tokenCount(s)` in s10 returns `len(strings.Fields(s))` — whitespace word count. The real cl100k_base tokenizer lives in s04. Why a stub here?

- **Module isolation.** s10 has zero deps; pulling `pkoukk/tiktoken-go` would couple us to s04's BPE merges file.
- **Conservative under-count.** Real tokens are typically ~1.3x word count for English prose. A budget that fits in word-count fits in real tokens too.
- **Easy swap.** Replace the function body with a tiktoken call and every test still passes.

If you wire this into a real pipeline, inject the tokenizer via interface; don't propagate the stub.

## Upstream cite / 上游引用

```
lightrag/operate.py:167-303    _handle_entity_relation_summary
lightrag/operate.py:304-385    _summarize_descriptions
lightrag/prompt.py:185-218     PROMPTS["summarize_entity_descriptions"]
lightrag/prompt.py:203         length-constraint clause
```

Annotated extract: [`upstream-readings/s10-summarization.py`](../../upstream-readings/s10-summarization.py).

The next chapter — [s11 dual-level retrieval](../../docs/zh/s11-query-modes.md) — is where the merged descriptions actually pay off: local mode does `vdbEntities.Query()` over the summarized descriptions to seed graph traversal. Without s10, that vector index would be confused by 47 near-duplicate fragments per entity.
