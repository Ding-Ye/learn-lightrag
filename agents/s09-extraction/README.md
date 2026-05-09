# s09 — entity / relation extraction with gleaning

> 用 LLM 当 extractor，分隔符替代 JSON、gleaning 多轮续写、`sha256(chunk + ver)` 缓存。
>
> Use the LLM as the extractor — delimiters instead of JSON, multi-round gleaning, and a `sha256(chunk + ver)` cache to skip already-extracted chunks.

## What this session teaches / 这一节教什么

After s08 the graph is **empty** — no one's filled it. s09 plugs in an LLM-driven extractor that, given a chunk:

1. Sends a delimited extraction system prompt + the chunk to a `Provider`.
2. Parses the response by `<|#|>` (field) / `\n` (record) / `<|COMPLETE|>` (terminator).
3. Optionally runs N gleaning rounds — each round packs the previous (user, assistant) pair as history and asks "what did you miss?".
4. Caches the raw LLM output by `sha256(chunk + promptVersion)` so re-running the pipeline doesn't re-extract.

Mirrors upstream `lightrag/operate.py:2883-3170` (`extract_entities`) + `lightrag/prompt.py` (`entity_extraction_system_prompt`).

## Files / 文件

| Path | Role |
|---|---|
| `extraction.go` | `Extractor` orchestrator + Provider/Chunk/Entity/Relationship type contracts |
| `extraction_prompt.go` | Faithful summary of upstream system / continuation prompts + delimiter constants |
| `parser.go` | Delimiter-based parser (lenient: skips malformed lines) |
| `gleaning.go` | Standalone gleaning loop helper (also re-implemented in extraction.go for call-counting) |
| `cache.go` | `KVStore` minimal interface + `MemoryKVStore` impl + `cacheKey()` |
| `main.go` | CLI demo + `MockProvider` (deterministic offline-friendly responses) |
| `extraction_test.go` | 6 spec tests + interface contract + fixtures roundtrip |
| `testdata/sample.txt` | 3-paragraph "Christmas Carol" excerpt |
| `testdata/llm_responses/*.txt` | Canned delimited LLM outputs for parser sanity tests |

## Run / 跑起来

```bash
cd /Users/yeding/learn-lightrag/agents/s09-extraction

# Default mock provider, 1 gleaning round
go run .

# Two gleaning rounds (more thorough recall on each chunk)
go run . -rounds 2

# Run twice to see cache hit ratio jump from 0% to 100%
go run . -runs 2

# Run tests
go test -v ./...
```

`-provider openai` is intentionally unwired in s09 — that responsibility lives in s02.  This session sticks to the deterministic `MockProvider` so tests never touch the network.

## Test list / 测试

| # | Test | Asserts |
|---|------|---------|
| 1 | `TestExtractionParsesDelimitedTuples` | parser correctly splits 4-field entity / 5-field relation records |
| 2 | `TestExtractionGleaningAddsNewEntities` | gleaning round-2 adds 1 new entity (3 total unique) |
| 3 | `TestExtractionCacheAvoidsDoubleCall` | second `Extract` on same chunk is served from cache |
| 4 | `TestExtractionMalformedOutputFallsBack` | wrong prefix / missing fields / blank input → empty parse, no panic |
| 5 | `TestExtractionMergesSourceIDsAcrossChunks` | same entity name from 2 chunks merges to one record with both chunk IDs |
| 6 | `TestExtractionRespectsContextCancel` | provider sleep + ctx cancel returns `context.Canceled` |

Plus `TestExtractionFixturesParse` (sanity — every `testdata/llm_responses/*.txt` round-trips through the parser).

## What changed vs s08 / 与 s08 的差异

- s08's graph is filled by hand-coded `UpsertNode`. s09 fills it via LLM extraction.
- New types: `Provider` + `MaxGleaningRounds` + `KVStore` (re-declared, sessions are isolated).
- New robustness model: parser skips malformed lines instead of crashing; cache rebuilds raw text on hit.

## Upstream cite / 上游引用

```
lightrag/operate.py:2883-3170   extract_entities (main path)
lightrag/operate.py:937-1063    _process_extraction_result (parser)
lightrag/operate.py:386-540     _handle_single_{entity,relationship}_extraction
lightrag/prompt.py:11-61        entity_extraction_system_prompt
lightrag/prompt.py:84-100       entity_continue_extraction_user_prompt
lightrag/prompt.py:8-9          DEFAULT_TUPLE_DELIMITER / DEFAULT_COMPLETION_DELIMITER
```

Annotated extract: [`upstream-readings/s09-extraction.py`](../../upstream-readings/s09-extraction.py).
