# Source: https://github.com/HKUDS/LightRAG  (paths below, branch `main`, 2026-05-09)
# License: MIT (Copyright 2025 LightRAG Team)
# Excerpted for learn-lightrag/s04-chunking — annotated, not for execution.
#
# This file collects the upstream slice that s04 condenses into Go:
#
#   1. lightrag/operate.py:102-166  — `chunking_by_token_size`, the entire
#                                      function s04 mirrors as
#                                      `ChunkByTokenSize(...)`.
#
# The annotations point out where s04 kept upstream behavior verbatim,
# where it simplified, and which later session picks up the loose ends.

# ─────────────────────────────────────────────────────────────────────────────
# (1) lightrag/operate.py:102-166 — chunking_by_token_size
# ─────────────────────────────────────────────────────────────────────────────
# Surrounding imports trimmed; this is the exact function body.

def chunking_by_token_size(
    tokenizer: Tokenizer,                                 # → s04 Tokenizer interface
    content: str,                                         # → s04 ChunkByTokenSize(text)
    split_by_character: str | None = None,                # → s04 bool flag (simplified)
    split_by_character_only: bool = False,                # → s04 omits (caller-side check)
    chunk_overlap_token_size: int = 100,                  # → s04 overlapTokenSize
    chunk_token_size: int = 1200,                         # → s04 chunkTokenSize
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)                    # encode whole doc once
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)    # → s04 chunkByCharacter rune split
        new_chunks = []
        if split_by_character_only:                       # strict mode — s04 omits
            # ... (omitted: per-piece length check that raises on overflow)
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                # ... (omitted: ChunkTokenLimitExceededError raise branch)
                new_chunks.append((len(_tokens), chunk))
        else:
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
        # Default path — what s04's chunkByTokens mirrors line-by-line.
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


# Reading notes (chunking_by_token_size):
#
# - The default `else:` branch is THE algorithm. Encode once, slide a window
#   of size `chunk_token_size` with step `chunk_token_size -
#   chunk_overlap_token_size`, decode each window, attach `chunk_order_index`.
#   s04's `chunkByTokens` is a 30-line Go translation; the only behavior
#   change is an extra `break` on `end == len(tokens)` to prevent a phantom
#   final iteration after the clamp.
#
# - The `if split_by_character:` branch handles non-English / non-whitespace
#   docs by first splitting on a domain separator, then windowing the
#   over-length pieces. s04 simplifies this into a rune-budget fallback
#   (`chunkByCharacter`): callers pass `splitByCharacter=true` to bypass the
#   tokenizer's "one giant token" failure mode for inputs like minified
#   JSON or CJK without explicit separators. Faithful upstream parity would
#   take `splitByCharacter` as `*string` (nil = default path; non-nil =
#   separator); flagged as a chapter exercise.
#
# - `min(chunk_token_size, len(tokens) - start)` is the last-chunk length
#   formula. s04 expresses the same thing as `end - start` after clamping
#   `end = min(start+chunkTokenSize, len(tokens))` — algebraically identical,
#   slightly more idiomatic in Go.
#
# - `chunk.strip()` is reproduced as `strings.TrimSpace(...)` in s04. The
#   strip step is load-bearing: cl100k_base's decode of a window often
#   starts with a leading space (BPE preserves the inter-token space), and
#   shipping that to downstream embedders inflates token counts.
#
# - `results` returns dicts with three keys: `tokens`, `content`,
#   `chunk_order_index`. There's no `full_doc_id` / `chunk_id` here —
#   those are added one layer up in `apipeline_process_enqueue_documents`.
#   s04's `Chunk` struct adds `ContentDocID` as a constructor parameter
#   so callers don't have to wrap.

# ─────────────────────────────────────────────────────────────────────────────
# Reading map (where to go next)
# ─────────────────────────────────────────────────────────────────────────────
# After s04 you've seen the chunking algorithm. From here:
#
#   s05 → lightrag/kg/json_kv_impl.py — the KV store that holds the chunks
#         after this function returns. s04's chunk objects (with `tokens`,
#         `content`, `chunk_order_index`) become KV values keyed by
#         chunk_id; s05 introduces `FilterMissing(ids)` which is what
#         decides whether s04's output needs re-embedding on a re-run.
#
#   s09 → lightrag/operate.py:~2883 (extract_entities) — every chunk that
#         s04 emits gets fed once into the LLM extractor. The
#         `chunk_order_index` field set here becomes the sort key for an
#         entity's `source_ids` list, which downstream affects how the
#         summary in s10 picks the "oldest mentioning chunks" for the LLM
#         prompt. So chunk ordering really matters and gets locked here.
#
#   s_full → step 5 of the 16-step end-to-end trace IS this function;
#         re-read it once you finish s11 to see where this chapter's
#         output lands and how it propagates through extraction → graph
#         storage → query-time context assembly.
