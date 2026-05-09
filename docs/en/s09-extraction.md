---
title: "s09 · Entity/relation extraction with gleaning"
chapter: 9
slug: s09-extraction
est_read_min: 12
---

# s09 · Entity/relation extraction with gleaning

> What this chapter does: use the LLM itself as the extractor, with a **delimiter-based** (not JSON) output contract — `entity<|#|>name<|#|>type<|#|>desc` one record per line, relations are `relation<|#|>src<|#|>tgt<|#|>keywords<|#|>desc`, the whole block ends with `<|COMPLETE|>`. Then a **gleaning loop**: feed the previous turn back as history and ask "anything missing?" N times. Cache by `sha256(chunk + prompt_version)` to skip already-extracted chunks. Mirrors upstream `lightrag/operate.py:2883-3170` `extract_entities` + the `entity_extraction_system_prompt` from `lightrag/prompt.py`.

---

## Problem / 问题

s08 finished the graph store — `UpsertNode` / `UpsertEdge` are ready to use. But **the graph is empty**. Given a doc, who is going to tell the graph "Scrooge is a PERSON, Marley is a PERSON, they were partners"?

You might think "just write a regex" — that path dies in one second:

1. **Regex doesn't understand semantics.** "Scrooge was a partner of Marley" needs the preposition `of` plus noun-before — but a regex can only match a literal token sequence. The next sentence "Marley, Scrooge's old partner, ..." flips everything around and the regex pattern stops matching.
2. **Schema is open-world.** Entity types may be `PERSON / ORG / EVENT / CONCEPT / ...`; relationship keywords are even worse — arbitrary phrases like "power dynamics" / "conflict resolution". You can't enumerate.
3. **Cross-paragraph references.** "Scrooge ... [400 words] ... Marley ... [200 words] ... they were partners" — long-range coreference is unreachable for a regex; you need semantic-level "read the whole thing then summarize".
4. **No gold dataset to train a classifier.** And even if you had one, no model trained on it can zero-shot every doc.

LightRAG just hands the whole thing off to the LLM: use GPT-4o-mini as the extractor, a few cents per call, recall is shockingly good. But this introduces two new problems:

- **JSON output is fragile.** The LLM occasionally forgets a quote, drops a `}`, emits unescaped unicode. `json.Unmarshal` panics, the whole chunk's data is lost.
- **One pass under-recalls.** The LLM "stops too early" — extracts 5–6 entities, decides it's done, emits `<|COMPLETE|>`. But the chunk may actually contain 10.

s09 fixes both at once — delimiters instead of JSON, gleaning loop instead of single-pass.

## Solution / 解决方案

Four steps:

1. **`Extractor` interface holds a `Provider` + `MaxGleaningRounds` + optional `Cache` + optional `Logger`.** `Extract(ctx, chunk Chunk) ([]Entity, []Relationship, error)` is the single entry point. Provider matches the s02 contract (re-declared in s09 — sessions are isolated, no imports); Chunk matches s04's shape (also re-declared).
2. **Delimiter contract (not JSON).** Upstream uses `<|#|>` as the field delimiter (`tuple_delimiter`), `\n` as the record separator, `<|COMPLETE|>` as the end-of-block. Entity has 4 fields: `entity<|#|>name<|#|>type<|#|>description`; Relation has 5: `relation<|#|>src<|#|>tgt<|#|>keywords<|#|>description`. This sequence is essentially impossible in natural text, so even if the LLM emits a paragraph as the description, the delimiters still split correctly.
3. **Gleaning loop.** After the first round, pack `(user_prompt, assistant_response)` as history and send a continuation user prompt ("what did you miss? add it"), then parse, dedup, merge. `MaxGleaningRounds == 1` (default) means two passes total (initial + 1 gleaning), matching upstream's `entity_extract_max_gleaning=1`. Early stop: if the round produces zero **new** entities (already-known ones don't count), break.
4. **Cache uses a subset of the `KVStore` interface.** Hash key = `sha256(chunk.Content || promptVersion)` — `promptVersion` is a const string; bumping it invalidates old cache. `Cache.Get(key)` hit → skip Provider, parse the cached raw text. Miss → call Provider, then `Cache.Upsert(key, raw_text)`. The CLI prints the hit ratio at the end.

## How It Works / 工作原理

```
                       Extractor.Extract(ctx, chunk)
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │   Step 1  cache key = sha256(chunk.Content || version)      │
   │           │                                                 │
   │           ├─ hit  ──► parse cached raw_text  ──► merge      │
   │           │                                                 │
   │           └─ miss ──► Provider.Complete(sysPrompt+chunk)    │
   │                            │                                │
   │                            ▼                                │
   │   Step 2  parser:                                           │
   │     split by "\n" + strip "<|COMPLETE|>"                    │
   │     for each line:                                          │
   │       fields = strings.Split(line, "<|#|>")                 │
   │       if fields[0] == "entity"   && len==4 → Entity         │
   │       if fields[0] == "relation" && len==5 → Relationship   │
   │       else: skip silently                                   │
   │                                                             │
   │   Step 3  gleaning loop (MaxGleaningRounds times):          │
   │     msgs = [user₀, assistant₀, user_continue, ...]          │
   │     resp = Provider.Complete(...)                           │
   │     parse resp → newEnts, newRels                           │
   │     dedup by Name (entity) / (Src,Tgt) (rel)                │
   │     if no NEW entity: break early                           │
   │                                                             │
   │   Step 4  Cache.Upsert(key, raw_text)                       │
   │                                                             │
   │   return entities, relationships, nil                       │
   └─────────────────────────────────────────────────────────────┘

   Cross-chunk merge (caller-side dedup over Extractor's output):

       chunk-1 ──► Extract ──► [Entity{X, sources:[chunk-1]}]
       chunk-2 ──► Extract ──► [Entity{X, sources:[chunk-2]}]
                                  │
                                  ▼  caller merges by Name
       merged: Entity{X, sources:[chunk-1, chunk-2]}
```

[`agents/s09-extraction/parser.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s09-extraction/parser.go), parser hot-path (~30 LOC):

```go
func parseExtractionOutput(s string) ([]Entity, []Relationship, error) {
    s = strings.ReplaceAll(s, completionDelimiter, "")
    var entities []Entity
    var relationships []Relationship
    for _, raw := range strings.Split(s, "\n") {
        line := strings.TrimSpace(raw)
        if line == "" {
            continue
        }
        // strip optional surrounding parens like "(entity<|#|>...)"
        line = strings.Trim(line, "()")
        fields := strings.Split(line, tupleDelimiter)
        if len(fields) < 2 {
            continue
        }
        kind := strings.ToLower(strings.TrimSpace(fields[0]))
        switch {
        case strings.Contains(kind, "entity") && len(fields) >= 4:
            entities = append(entities, Entity{
                Name:        strings.TrimSpace(fields[1]),
                Type:        strings.ToLower(strings.TrimSpace(fields[2])),
                Description: strings.TrimSpace(fields[3]),
            })
        case strings.Contains(kind, "relation") && len(fields) >= 5:
            relationships = append(relationships, Relationship{
                SrcID:       strings.TrimSpace(fields[1]),
                TgtID:       strings.TrimSpace(fields[2]),
                Keywords:    strings.TrimSpace(fields[3]),
                Description: strings.TrimSpace(fields[4]),
                Weight:      1.0,
            })
        }
    }
    return entities, relationships, nil
}
```

**Four non-obvious points:**

1. **The "record separator" is `\n`, not `<|#|>`.** Upstream uses **two** delimiter levels: `tuple_delimiter=<|#|>` for fields, `\n` for records, `<|COMPLETE|>` to mark end-of-block. We strip `<|COMPLETE|>` first, split by `\n`, then split each line by `<|#|>` — two independent levels. This is why the description field can hold a full sentence without being mis-split.
2. **`entity` prefix uses `Contains`, not `==`.** Upstream's `_handle_single_entity_extraction` uses `"entity" in record_attributes[0]` — tolerates LLM output like `("entity"<|#|>...)` with parens, quotes, leading punctuation. Our `strings.Contains(kind, "entity")` does the same in one line. `relation` / `relationship` are interchangeable too (upstream's comment: "treat interchangeable").
3. **Missing fields don't error — just skip.** Robustness rule #1: if the LLM occasionally drops a field, lose **that line** rather than panicking the whole chunk. `len(fields) < 4` falls through the case, that line is gone. The test `TestExtractionMalformedOutputFallsBack` is dedicated to this.
4. **Gleaning is history-based, not prompt-stuffing.** Upstream's continuation prompt is short — just "find what's missing" — because the previous user prompt + assistant response are **already in `history_messages`**. The LLM sees what was extracted last round and naturally knows what to add. Our Go side replicates this with `Messages: append(prev, userContinue)`, stacking previous user/assistant pairs back into messages each round.

## What Changed / 与 s08 的变化

s08 finished the graph store but **the graph is empty** — `UpsertNode` only got called from main.go's hand-coded demo. s09 is the first place where the graph **automatically** gets populated: feed in a chunk, the LLM emits entities/relationships, the caller turns around and feeds them to `UpsertNode` / `UpsertEdge`.

| Dimension | s08 (graph store) | s09 (extraction) |
|---|---|---|
| Data origin | Hand-passed Entity / Relationship literals | Extracted from chunk text via LLM |
| LLM calls | 0 | 1 + N per chunk (N = MaxGleaningRounds) |
| Output contract | Go struct (Name, Type, ...) | Delimited text → parsed into the same struct |
| Cache | None (graph IS the state) | Yes — `sha256(chunk+ver)` hit skips LLM |
| Robustness | Brittle (broken edge = crash) | Lenient parser (drop fields, skip lines) |
| Test focus | BFS subgraph, tie-break | Parsing correctness, gleaning accumulation, cache hit |
| Upstream ref | `kg/networkx_impl.py` | `operate.py:2883-3170 + prompt.py` |

**Why is the cache mandatory?** A 100-chunk doc with `MaxGleaningRounds=1` is 200 LLM calls. If you change a graph-side BFS algo and re-run ingestion, you should **not** re-extract — the chunk text didn't change, the prompt didn't change, the result should be identical. This is the infrastructure that makes iterative debugging on large docs viable.

**Why is gleaning on by default (rounds=1)?** Upstream's `entity_extract_max_gleaning=1` is the default, not optional. Single-pass recall is too low — LLMs hit a stopping condition around entity 5-6 and emit `<|COMPLETE|>`, but the chunk may have 10. One round of gleaning lifts recall from ~60% to ~85% (paper-reported). The second round only adds ~5%, so the default stops at 1.

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s09-extraction

# 1. Run the mock provider on 3 chunks (no network — what CI uses)
go run . -provider mock -rounds 1 -doc ./testdata/sample.txt

# 2. See gleaning accumulate (rounds=2 runs one more pass than rounds=1)
go run . -provider mock -rounds 2 -doc ./testdata/sample.txt

# 3. Run the same doc twice — second time the cache hits everything
go run . -provider mock -rounds 1 -doc ./testdata/sample.txt
go run . -provider mock -rounds 1 -doc ./testdata/sample.txt   # cache hit ratio = 100%

# 4. Run tests (6 + 1 interface contract)
go test -v ./...
```

Demo output (excerpt):

```
=== chunk 1: extracted 3 entities, 2 relations  (cache MISS) ===
  entity   Scrooge       type=person       "a miserly businessman..."
  entity   Marley        type=person       "Scrooge's deceased business partner"
  entity   Christmas     type=event        "the central holiday in the story"
  relation Scrooge -- Marley   keywords=partnership   "...were business partners"

=== chunk 2: extracted 2 entities, 1 relation  (cache MISS) ===
  ...

cache hit ratio: 0/3 chunks (0%)   ← first run

=== second run on same doc ===
cache hit ratio: 3/3 chunks (100%)   ← all served from sha256-keyed cache
```

Test matrix:

| Test | Asserts |
|---|---|
| `TestExtractorInterfaceContract` | Compile-time: `*Extractor` satisfies internal interface contract |
| `TestExtractionParsesDelimitedTuples` | Given a known delimited blob, parser returns correct entities + relations |
| `TestExtractionGleaningAddsNewEntities` | rounds=2, round-1 yields 2, round-2 yields 1 new → 3 unique total |
| `TestExtractionCacheAvoidsDoubleCall` | Same chunk extracted twice; provider call count == 1 |
| `TestExtractionMalformedOutputFallsBack` | Missing fields / wrong prefix / blank lines → parser returns empty, no panic |
| `TestExtractionMergesSourceIDsAcrossChunks` | chunk-1 + chunk-2 both extract entity X → merged sources=[chunk-1, chunk-2] |
| `TestExtractionRespectsContextCancel` | Provider sleeps + ctx cancel → Extract returns context.Canceled |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/operate.py:2949-3050 (excerpt; the full extract_entities is ~290 lines)
# Format the system prompt, call the LLM, parse, run gleaning, merge — the
# hot-path is just these few dozen lines.

# Get initial extraction
entity_extraction_system_prompt = PROMPTS[
    "entity_extraction_system_prompt"
].format(**context_base)
entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
    **{**context_base, "input_text": content}
)
entity_continue_extraction_user_prompt = PROMPTS[
    "entity_continue_extraction_user_prompt"
].format(**{**context_base, "input_text": content})

final_result, timestamp = await use_llm_func_with_cache(
    entity_extraction_user_prompt,
    use_llm_func,
    system_prompt=entity_extraction_system_prompt,
    llm_response_cache=llm_response_cache,
    cache_type="extract",
    chunk_id=chunk_key,
)

history = pack_user_ass_to_openai_messages(
    entity_extraction_user_prompt, final_result
)

# Initial round: delimiter parse
maybe_nodes, maybe_edges = await _process_extraction_result(
    final_result, chunk_key, timestamp, file_path,
    tuple_delimiter=context_base["tuple_delimiter"],
    completion_delimiter=context_base["completion_delimiter"],
)

# Gleaning loop — note upstream actually runs ONE extra round, not N rounds:
# the param name is `entity_extract_max_gleaning` but semantically it's "do
# this extra round, or skip it".
if entity_extract_max_gleaning > 0:
    glean_result, timestamp = await use_llm_func_with_cache(
        entity_continue_extraction_user_prompt,
        use_llm_func,
        system_prompt=entity_extraction_system_prompt,
        llm_response_cache=llm_response_cache,
        history_messages=history,
        cache_type="extract",
        chunk_id=chunk_key,
    )
    glean_nodes, glean_edges = await _process_extraction_result(
        glean_result, chunk_key, timestamp, file_path,
        tuple_delimiter=context_base["tuple_delimiter"],
        completion_delimiter=context_base["completion_delimiter"],
    )
    # Merge: new entity → just add. Existing entity → compare description
    # length and keep the longer one.
    for entity_name, glean_entities in glean_nodes.items():
        if entity_name in maybe_nodes:
            if glean_desc_len > original_desc_len:
                maybe_nodes[entity_name] = list(glean_entities)
        else:
            maybe_nodes[entity_name] = list(glean_entities)
```

**How to read this:**

- `pack_user_ass_to_openai_messages(user_prompt, response)` packs a `(user, assistant)` pair into a messages list for the next round. Our Go side appends `[]Message{{Role:"user", ...}, {Role:"assistant", ...}}` directly — same semantic, Python's helper is one dict, Go's is two structs.
- `use_llm_func_with_cache`'s `cache_type="extract"` is a namespace inside the cache — LightRAG caches extract / summarize / query calls in one KV store, separated by namespace. Our Go side only caches extract calls, so we drop the namespace and hash directly into the root keyspace.
- The merge strategy "compare description length, keep the longer" is a detail: gleaning rounds may produce a **more detailed** description for the same entity (because the LLM has the prior context). Upstream uses length as a proxy. Our impl takes a simpler "first writer wins" approach (`if !exists` then insert) and leaves length comparison as an extension exercise — the most useful upgrade is when you start A/B testing.
- `_process_extraction_result`'s robustness is famous: split by `\n`, then use split-by-multi-markers to repair LLM-side mistakes like using `<|#|>` as a record separator or omitting the `entity` prefix. Our Go `parser.go` keeps the spirit (split + len-check + skip) without trying to repair — robustness via permissive paths, not repair paths.

The annotated extract + reading map: [`upstream-readings/s09-extraction.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s09-extraction.py).

The next chapter [s10 description merge](s10-summarization.md) handles the "same entity in 47 chunks" case — LLM-summarizing those 47 description fragments into one. After that, s11's `local` mode uses entities extracted in s09 as seeds for `vdbEntities.Query`. s09 is the inflection point: the graph layer goes from empty to populated.
