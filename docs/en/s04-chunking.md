---
title: "s04 · Token-based chunking with overlap"
chapter: 4
slug: s04-chunking
est_read_min: 8
---

# s04 · Token-based chunking with overlap

> What this teaches: promote s01's "split-on-double-newlines + hard-cut long paragraphs" placeholder into the real upstream `chunking_by_token_size` — a 1200/100 sliding window over BPE tokens, with every adjacent chunk pair sharing 100 tokens for context preservation. This chapter introduces the repo's first external Go dependency, `github.com/pkoukk/tiktoken-go`, and ships a stdlib-only `WhitespaceTokenizer` fallback so offline CI / tests don't need a network.

---

## Problem / 问题

s01's `ChunkByNewlines(text, maxRunes)` breaks down in three concrete ways:

1. **Rune count is not token count.** OpenAI's `gpt-4o-mini` input limit is in tokens, not characters. A 1200-rune chunk of English is ~250 tokens, but 1200 runes of CJK / code can be 800-1500 tokens — feeding such a "rune-bounded" chunk into chat completion occasionally blows past 8K context. Production code can't rely on rune counts as a stand-in.
2. **Hard-cutting paragraphs drops context.** s01 takes `runes[start:end]` straight through whenever a paragraph exceeds `maxRunes`, which means the sentence straddling that boundary gets cut in half. Retrieval then surfaces "half a sentence" as a top-k chunk, and no amount of LLM cleverness restores the missing words. This is exactly why upstream keeps a 100-token overlap — the last 100 tokens of chunk N are also the first 100 tokens of chunk N+1, so a sentence crossing the boundary is visible in both.
3. **Chunk count is non-deterministic.** `ChunkByNewlines` produces a chunk count that depends on paragraph structure; tweak the document's whitespace and the same content gives a different `ChunkOrderIndex` range. That breaks LightRAG's contract that "chunk_id is a stable citation key" — s09's extraction stores chunk IDs inside entity `source_ids`, persisted to disk; if those IDs shift on re-run, downstream KV / VDB go out of sync.

Upstream's fix lives in `lightrag/operate.py:102-166` as `chunking_by_token_size`: BPE-encode the whole document once, slide a fixed-size window across the token stream, decode each window, attach `chunk_order_index`. s04's job is to translate those 65 lines of Python into ~200 lines of Go and pair them with a network-free fallback tokenizer for offline tests.

## Solution / 解决方案

Four pieces:

1. **`Tokenizer` is a 2-method interface** — `Encode(text) []int` + `Decode(tokens) string`. This mirrors upstream's `Tokenizer` Protocol exactly: `chunking_by_token_size` only ever calls those two methods, so the interface surface needs nothing more.
2. **`TiktokenTokenizer` wraps `pkoukk/tiktoken-go` cl100k_base** — the same BPE OpenAI uses for `text-embedding-3-small` / `gpt-4o-mini`. The first call to `tiktoken.GetEncoding("cl100k_base")` downloads a ~1.6 MB merges file from the vendor URL and caches it under `~/.tiktoken/`; later calls hit the cache.
3. **`WhitespaceTokenizer` is the stdlib fallback** — `Encode = strings.Fields`, `Decode = strings.Join(parts, " ")`, vocabulary IDs minted on first sight inside an in-memory map. Round-trip isn't byte-stable (runs of whitespace collapse), but the two properties chunking actually needs hold: token-count is deterministic, and re-tokenizing a concatenated chunk pair recovers the original tokens. Good enough for offline tests.
4. **`ChunkByTokenSize(docID, text, tokenizer, chunkTokenSize, overlapTokenSize, splitByCharacter)`** — the default path is "encode → step = chunkTokenSize-overlap → decode → emit"; `splitByCharacter=true` falls back to a rune-based split (good for CJK / very long no-whitespace strings). Validation: `chunkTokenSize > overlapTokenSize > 0` must hold; empty input returns `ErrEmptyText`.

The `Chunk` struct is the Go translation of upstream's `TextChunkSchema` (lightrag/base.py:71): `ContentDocID` + `Content` + `Tokens` + `ChunkOrderIndex`. s01 already declared this struct; s04 finally replaces the `Tokens` field's "rune-count stub" with the real BPE token count.

## How It Works / 工作原理

```
text = "lorem ipsum dolor sit amet ..."
        │
        ▼  tokenizer.Encode(text)
tokens = [101, 4023, 1991, 3920, 2017, 1232, ...]   len = 714

       ┌─────────────────────────────── chunkTokenSize = 60 ──────────────────────────────┐
window 0: tokens[0:60]                                                         step = 50
                ┌──── overlap 10 ────┐
                ▼                    ▼
window 1:        tokens[50:110]
                                ┌──── overlap 10 ────┐
                                ▼                    ▼
window 2:                       tokens[100:160]                                      ...

Each window: tokenizer.Decode(slice).TrimSpace() → Chunk{
    ContentDocID:   docID,
    Content:        decoded text,
    Tokens:         min(chunkTokenSize, len(tokens) - start),   ← last window may be shorter
    ChunkOrderIndex: i,                                          ← 0, 1, 2, ...
}
```

The load-bearing 30 lines (excerpt from [`agents/s04-chunking/chunking.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s04-chunking/chunking.go)):

```go
func chunkByTokens(
    docID, text string,
    tokenizer Tokenizer,
    chunkTokenSize, overlapTokenSize int,
) ([]Chunk, error) {
    tokens := tokenizer.Encode(text)
    if len(tokens) == 0 {
        return nil, ErrEmptyText
    }

    step := chunkTokenSize - overlapTokenSize
    out := make([]Chunk, 0, (len(tokens)/step)+1)
    idx := 0
    for start := 0; start < len(tokens); start += step {
        end := start + chunkTokenSize
        if end > len(tokens) {
            end = len(tokens)
        }
        body := strings.TrimSpace(tokenizer.Decode(tokens[start:end]))
        if body == "" { continue } // defensive: skip pure-whitespace decodes
        out = append(out, Chunk{
            ContentDocID:    docID,
            Content:         body,
            Tokens:          end - start,           // == min(chunkTokenSize, len-start)
            ChunkOrderIndex: idx,
        })
        idx++
        if end == len(tokens) { break } // last window done; don't loop again
    }
    return out, nil
}
```

**4 non-obvious points**:

1. **`step = chunkTokenSize - overlapTokenSize`, not `chunkTokenSize`.** The window slides; it doesn't relay-race. Each iteration advances `start` by `step` tokens but the window length stays at `chunkTokenSize` — so the last `overlapTokenSize` tokens of chunk N are byte-for-byte the first `overlapTokenSize` tokens of chunk N+1. That's the implementation of "100-token overlap": not a bridge inserted between chunks, but every chunk's tail re-used as the next chunk's head.
2. **The last chunk's `Tokens` is `end - start`, not `chunkTokenSize`.** When `start + chunkTokenSize > len(tokens)`, `end` is clamped to `len(tokens)`, so `end - start` equals upstream's `min(chunk_token_size, len(tokens) - start)`. The test `TestChunkExactlyAtBoundary` locks down this boundary: N=chunkSize → 1 chunk; N=chunkSize+1 → 2 chunks.
3. **Why `break` when `end == len(tokens)`.** Without the break, a degenerate next iteration could end up writing `tokens[start:end]` where `start ≥ len(tokens)` — at best it triggers the `body == ""` skip, at worst it's a phantom empty chunk. The explicit break is the readable termination condition.
4. **Compile-time interface assertions** at the bottom of `chunking.go`: `var _ Tokenizer = (*TiktokenTokenizer)(nil)` and `var _ Tokenizer = (*WhitespaceTokenizer)(nil)`. Any future rename of `Encode` → `Tokenize` blows up at `go build`, not at the first runtime call into tiktoken.

## What Changed / 与 s01 的变化

```diff
 // s01 chunking.go
-func ChunkByNewlines(docID, text string, maxRunes int) []Chunk {
-    if maxRunes <= 0 { maxRunes = 1200 }
-    var out []Chunk
-    idx := 0
-    for _, para := range strings.Split(text, "\n\n") {
-        para = strings.TrimSpace(para)
-        if para == "" { continue }
-        runes := []rune(para)
-        for start := 0; start < len(runes); start += maxRunes {
-            end := start + maxRunes
-            if end > len(runes) { end = len(runes) }
-            out = append(out, Chunk{
-                ContentDocID:    docID,
-                Content:         string(runes[start:end]),
-                Tokens:          end - start,            // ← rune-count stub
-                ChunkOrderIndex: idx,
-            })
-            idx++
-        }
-    }
-    return out
-}

+// s04 chunking.go
+func ChunkByTokenSize(
+    docID, text string,
+    tokenizer Tokenizer,
+    chunkTokenSize, overlapTokenSize int,
+    splitByCharacter bool,
+) ([]Chunk, error) {
+    if tokenizer == nil { return nil, fmt.Errorf("s04: tokenizer is nil") }
+    if !(overlapTokenSize > 0 && chunkTokenSize > overlapTokenSize) {
+        return nil, fmt.Errorf("%w: got chunk=%d overlap=%d",
+            ErrInvalidChunkSize, chunkTokenSize, overlapTokenSize)
+    }
+    if strings.TrimSpace(text) == "" { return nil, ErrEmptyText }
+    if splitByCharacter {
+        return chunkByCharacter(docID, text, tokenizer, chunkTokenSize, overlapTokenSize)
+    }
+    return chunkByTokens(docID, text, tokenizer, chunkTokenSize, overlapTokenSize)
+}
```

Key differences:

- **`Tokens` field semantics changed.** s01's `Tokens = end - start` was a rune count; s04's `Tokens = end - start` is a BPE token count. For a 600-word English doc, s01 reports `Tokens` ~3000+ while s04 with cl100k_base reports ~700. Downstream s07 vector-store needs a token budget, so this fix is load-bearing.
- **Signature gains `tokenizer Tokenizer` + `splitByCharacter bool` parameters.** Dependency-injecting the tokenizer lets CI use `WhitespaceTokenizer` while production uses `TiktokenTokenizer`, with no caller changes. `splitByCharacter` mirrors upstream's same-named knob for non-English long-input fallback.
- **Errors are explicit.** s01 never errored (worst case: empty slice); s04 validates inputs and returns `ErrEmptyText` / `ErrInvalidChunkSize`. With `errors.Is` the pipeline can distinguish "doc is genuinely empty" from "caller passed bad config".

s05's KV chapter persists chunks to JSON; at that point `Chunk.Tokens` and `Chunk.ChunkOrderIndex` become part of the serialized value, so locking down their semantics now is exactly the right time.

## Try It / 动手试一试

```bash
cd agents/s04-chunking

# Default: cl100k_base + 1200/100; the 600-word sample fits in 1 chunk.
go run .
# == 1 chunks ==
#   [00] tokens= 714  Eleanor Hartwell was born in 1843...

# Shrink chunkSize to see windowing in action.
go run . -tokenizer ws -size 60 -overlap 10
# == 11 chunks ==
#   [00] tokens=  60  Eleanor Hartwell was born in 1843...
#   [01] tokens=  60  in her father's careful hand, and presented herself...
#   ...adjacent chunks share 10 tokens at the tail/head boundary

# Fully offline demo (no tiktoken download).
go run . -tokenizer ws

# splitByCharacter rune-based fallback.
go run . -split-by-character -size 200 -overlap 30

# Run the tests (7 of them, all WhitespaceTokenizer — fully offline).
go test -v ./...
# === RUN   TestChunkRespectsTokenLimit          every chunk Tokens <= chunkTokenSize
# === RUN   TestChunkOverlapPreserved            adjacent chunks share `overlap` tokens
# === RUN   TestChunkOrderIndexMonotonic         indices = 0, 1, 2, ... contiguously
# === RUN   TestChunkSplitByCharacterFallback    no-whitespace blob splits into multiple
# === RUN   TestChunkEmptyTextReturnsEmpty       empty input → 0 chunks + ErrEmptyText
# === RUN   TestChunkExactlyAtBoundary           chunkSize → 1 chunk; chunkSize+1 → 2
# === RUN   TestTokenizerInterfaceContract       compile-time + WS round-trip smoke
```

The first `-tokenizer tiktoken` run downloads ~1.6 MB of BPE merges from the vendor URL into `~/.tiktoken/`. On air-gapped machines, use `-tokenizer ws` instead.

## Upstream Source Reading / 上游源码阅读

The load-bearing 50 lines from `lightrag/operate.py:102-166` (full annotated excerpt at [`upstream-readings/s04-chunking.py`](../../upstream-readings/s04-chunking.py)):

```python
def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        # ... (omitted: split_by_character_only strict mode + per-piece length check)
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        for chunk in raw_chunks:
            _tokens = tokenizer.encode(chunk)
            if len(_tokens) > chunk_token_size:
                for start in range(
                    0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                ):
                    chunk_content = tokenizer.decode(
                        _tokens[start : start + chunk_token_size]
                    )
                    new_chunks.append(
                        (min(chunk_token_size, len(_tokens) - start), chunk_content)
                    )
            else:
                new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append({
                "tokens": _len,
                "content": chunk.strip(),
                "chunk_order_index": index,
            })
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append({
                "tokens": min(chunk_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
    return results
```

**Reading notes**:

- **`Tokenizer` is a Protocol, not a base class.** Upstream's `Tokenizer` is a `typing.Protocol` — structural / duck-typed. Go interfaces are also structural, so this translates one-to-one. The same Protocol → interface translation pattern shows up in s06's `OpenAIEmbedder`.
- **`split_by_character: str | None` upstream is a string; s04 uses a bool.** Upstream takes a separator string (default `None`); on match, it follows the split-then-window path. s04 simplifies to `bool` plus a built-in rune-based fallback — one fewer API dimension, but the core need ("a no-whitespace long blob can still be split") is met. To replicate exact semantics, change `splitByCharacter` to `*string` so nil = default path and a non-nil pointer is the separator.
- **`split_by_character_only=True` strict mode is omitted in s04.** Upstream has a strict branch that raises if any post-split piece exceeds `chunk_token_size`. s04 doesn't ship it because the semantic is "I, the caller, promise my separator produces small enough pieces, and want to error otherwise" — that's clearer as caller-side validation. Good chapter exercise.
- **`chunk.strip()` vs. `strings.TrimSpace`.** Upstream strips each chunk; s04 also calls `strings.TrimSpace` and additionally skips `body == ""` to avoid emitting phantom chunks from runs of pure whitespace. Equivalent behavior.
- **Last-chunk length formula.** Upstream: `min(chunk_token_size, len(tokens) - start)`. s04: `end - start` where `end := min(start+chunkTokenSize, len(tokens))`. Algebraically identical.
- **No `full_doc_id` / `chunk_id` in `results`.** Upstream returns content-only fields here; the docID / chunk_id is appended at the next layer up in `lightrag/lightrag.py:apipeline_process_enqueue_documents`. s04 takes `ContentDocID` as a parameter directly — one fewer indirection, same semantic.

**Want more**: upstream `lightrag/lightrag.py:1500-1700` (`apipeline_process_enqueue_documents`) is the real call site for `chunking_by_token_size`. It's invoked after doc-status flips to PROCESSING and before the embedding batch starts — once per document, then `compute_mdhash_id(content)` mints the per-chunk ID. s09 (entity extraction) consumes those chunks; that's where s04's `ChunkOrderIndex` becomes the sort key for an entity's `source_ids`.

---

**Next chapter preview**: s05 replaces s01's `sync.Map` with a JSON-persisted `JSONKVStore` and adds a key new method, `FilterMissing(ids) []string` — "of these chunk_ids, which are NOT yet stored?" That's the primitive upstream's `apipeline_process_enqueue_documents` uses to decide which chunks need fresh embeddings, and it's the very first store layer s04's chunks land in.
