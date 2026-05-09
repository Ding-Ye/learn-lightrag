# =============================================================================
# Upstream reading for s10 — Map-reduce description summarization
# =============================================================================
#
# Source:  https://github.com/HKUDS/LightRAG (commit on 2026-05-09 main)
# Files:   lightrag/operate.py
#            _handle_entity_relation_summary @ L167-303 (decision tree)
#            _summarize_descriptions @ L304-385 (LLM call, JSONL render)
#          lightrag/prompt.py
#            PROMPTS["summarize_entity_descriptions"] @ L185-218
#
# License: MIT (HKUDS, 2025) — same as this repo.  Excerpts are verbatim
#          unless prefixed with "[s10: omitted ...]" or "[s10: summary]".
#
# This file is what our agents/s10-summarization/{summarize.go, merge.go,
# summarize_prompt.go} ports.  Read this first, then open the three Go
# files side by side.  Three correspondences are load-bearing:
#
#   1. THRESHOLD-FIRST DECISION.  If len(descriptions) < threshold AND
#      total_tokens < budget, just join with separator — no LLM call.
#      This is the cheap-path that keeps small entities free of summary
#      drift while still bounding cost on big entities.
#
#   2. MAP-REDUCE WHEN OVER LIMIT.  If we exceed either threshold, split
#      descriptions into context-size windows, LLM-summarize each window,
#      recurse on the partials.  Single-description chunks pass through
#      unchanged (an upstream optimization).
#
#   3. JSONL INPUT FORMAT.  Descriptions are wrapped as
#      `{"Description": "..."}` and joined with newlines so the LLM sees
#      a clean list — not a blob with paragraph breaks of unknown
#      provenance.  Our Go side reuses the same shape via json.Marshal +
#      strings.Builder.
#
# Tests in our summarize_test.go assert all three: short-list skip-LLM,
# long-list invoke-LLM, recursion-until-fits, empty-input no-op.

# -----------------------------------------------------------------------------
# Verbatim excerpt — _handle_entity_relation_summary head + decision
# (lightrag/operate.py:167-227, ~50 LOC):
# -----------------------------------------------------------------------------

# async def _handle_entity_relation_summary(
#     description_type: str,
#     entity_or_relation_name: str,
#     description_list: list[str],
#     separator: str,
#     global_config: dict,
#     llm_response_cache: BaseKVStorage | None = None,
# ) -> tuple[str, bool]:
#     """Handle entity relation description summary using map-reduce approach.
#
#     Decision tree:
#       1. If total tokens < summary_context_size and len < force_llm_summary_on_merge → no LLM
#       2. If total tokens < summary_max_tokens, summarize with LLM directly
#       3. Otherwise, split descriptions into chunks fitting context_size
#       4. Summarize each chunk, then recursively process the summaries
#       5. Continue until final fits or num descriptions < threshold
#     """
#     if not description_list:
#         return "", False
#     if len(description_list) == 1:
#         return description_list[0], False
#
#     tokenizer = global_config["tokenizer"]
#     summary_context_size      = global_config["summary_context_size"]
#     summary_max_tokens        = global_config["summary_max_tokens"]
#     force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]
#
#     current_list = description_list[:]
#     llm_was_used = False
#
#     while True:
#         total_tokens = sum(len(tokenizer.encode(d)) for d in current_list)
#
#         if total_tokens <= summary_context_size or len(current_list) <= 2:
#             if (len(current_list) < force_llm_summary_on_merge
#                 and total_tokens < summary_max_tokens):
#                 # no LLM needed, just join
#                 return separator.join(current_list), llm_was_used
#             else:
#                 final_summary = await _summarize_descriptions(...)
#                 return final_summary, True

# -----------------------------------------------------------------------------
# Verbatim excerpt — chunking phase (operate.py:240-285, ~45 LOC):
# -----------------------------------------------------------------------------

#         # Need to split into chunks - Map phase
#         # Ensure each chunk has minimum 2 descriptions to guarantee progress
#         chunks = []
#         current_chunk = []
#         current_tokens = 0
#         for desc in current_list:
#             desc_tokens = len(tokenizer.encode(desc))
#             if current_tokens + desc_tokens > summary_context_size and current_chunk:
#                 if len(current_chunk) == 1:
#                     # force one more so chunk has 2
#                     current_chunk.append(desc)
#                     chunks.append(current_chunk)
#                     current_chunk = []
#                     current_tokens = 0
#                 else:
#                     chunks.append(current_chunk)
#                     current_chunk = [desc]
#                     current_tokens = desc_tokens
#             else:
#                 current_chunk.append(desc)
#                 current_tokens += desc_tokens
#         if current_chunk:
#             chunks.append(current_chunk)
#
#         # Reduce phase: summarize each group
#         new_summaries = []
#         for chunk in chunks:
#             if len(chunk) == 1:
#                 # single-description chunk: pass-through, no LLM
#                 new_summaries.append(chunk[0])
#             else:
#                 summary = await _summarize_descriptions(...)
#                 new_summaries.append(summary)
#                 llm_was_used = True
#         current_list = new_summaries  # loop and re-evaluate

# -----------------------------------------------------------------------------
# Verbatim excerpt — _summarize_descriptions (operate.py:304-360, ~55 LOC):
# -----------------------------------------------------------------------------

# async def _summarize_descriptions(
#     description_type: str,
#     description_name: str,
#     description_list: list[str],
#     global_config: dict,
#     llm_response_cache: BaseKVStorage | None = None,
# ) -> str:
#     use_llm_func = global_config["llm_model_func"]
#     use_llm_func = partial(use_llm_func, _priority=8)
#     language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
#     summary_length_recommended = global_config["summary_length_recommended"]
#     prompt_template = PROMPTS["summarize_entity_descriptions"]
#
#     tokenizer = global_config["tokenizer"]
#     summary_context_size = global_config["summary_context_size"]
#
#     # Wrap each description as a JSON object and truncate to fit context size
#     json_descriptions = [{"Description": d} for d in description_list]
#     truncated = truncate_list_by_token_size(
#         json_descriptions,
#         key=lambda x: json.dumps(x, ensure_ascii=False),
#         max_token_size=summary_context_size,
#         tokenizer=tokenizer,
#     )
#     joined_descriptions = "\n".join(json.dumps(d, ensure_ascii=False) for d in truncated)
#
#     context_base = dict(
#         description_type=description_type,
#         description_name=description_name,
#         description_list=joined_descriptions,
#         summary_length=summary_length_recommended,
#         language=language,
#     )
#     use_prompt = prompt_template.format(**context_base)
#     summary, _ = await use_llm_func_with_cache(
#         use_prompt, use_llm_func,
#         llm_response_cache=llm_response_cache, cache_type="summary",
#     )
#     # post-check: warn if summary exceeds embedding_token_limit
#     return summary

# -----------------------------------------------------------------------------
# Reading map — what to read AFTER s10
# -----------------------------------------------------------------------------
#
# Upstream files that consume the summarized descriptions s10 produces:
#
#   lightrag/operate.py:1623-1947  (_merge_nodes_then_upsert)
#   lightrag/operate.py:1948-2300  (_merge_edges_then_upsert)
#       These call _handle_entity_relation_summary BEFORE upserting into
#       graph_storage and vdb_entities.  Our Go MergeEntities /
#       MergeRelationships in merge.go shadow this flow at the pure-merge
#       layer (the actual graph upsert is the caller's job).
#
#   lightrag/operate.py:3164-3410  (kg_query → local mode)
#       Local mode does vdb_entities.query(keyword_embed) — and the
#       embeddings are computed over the SUMMARIZED descriptions
#       produced by s10.  Without summarization, vdb_entities would
#       contain 47 near-duplicate vectors per entity.
#       → Maps to s11 (Dual-level retrieval).
#
# Reading map back to s09:
#
#   s09's Extractor produces per-chunk Entity / Relationship records.
#   Each record carries SourceIDs = [chunk_id_for_this_chunk] and one
#   raw description snippet.  s10 is the consumer: groups by name,
#   summarizes the description fragments, returns one merged record.
#   The merge logic mirrors s09's TestExtractionMergesSourceIDsAcrossChunks
#   helper (mergeEntitiesByName) but adds the LLM step on the description
#   field.
#
# What s10 deliberately omits (and why):
#
#   - Cache integration.  Upstream's _summarize_descriptions threads the
#     llm_response_cache through with cache_type="summary".  Our Go port
#     leaves caching as a callback exercise — the merge layer doesn't own
#     a cache; the caller can wrap Provider.Complete with a memoizing
#     decorator if desired (s09's cache.go is the template).
#
#   - Real tokenizer.  Upstream uses tiktoken (cl100k_base).  Our Go
#     tokenCount() is whitespace-word-count for didactic isolation;
#     the real one lives in s04.  Real tokens > word count for English
#     (~1.3x), so our budget is conservative — fits in word count fits
#     in real tokens too.
#
#   - Embedding-token limit post-check.  Upstream warns if the final
#     summary exceeds embedding_token_limit (operate.py:373-385).  Our
#     port skips this — it's a warning-only path, not control flow.
#
#   - Mode-frequency type election.  Upstream picks the most-frequent
#     entity_type across the merged group (statistics.mode).  Our Go
#     MergeEntities preserves the FIRST-seen Type.  One-line extension
#     exercise.
#
#   - description-length conflict resolution.  Upstream's `summary_length_recommended`
#     is a soft target the prompt enforces; we pass budgetTokens as the
#     same soft hint via {summary_length} in the prompt.
#
# -----------------------------------------------------------------------------
# Glossary one-liners
# -----------------------------------------------------------------------------
#
#   summary_max_tokens          Final-summary soft budget; output should
#                                fit under this.  Default in upstream
#                                config: 500.  Maps to our budgetTokens.
#   summary_context_size        Per-chunk window when map-splitting.
#                                Default: 2000.  Maps to our contextSize.
#   force_llm_summary_on_merge  Threshold; below this AND under budget,
#                                skip LLM and concat.  Default: 6.  Maps
#                                to our countThreshold.
#   summary_length_recommended  Soft prompt-side hint for output length
#                                (LLM is told "must not exceed X tokens").
#                                We pass budgetTokens here too.
#   description_type            "entity" or "relation" — flows into prompt
#                                so the LLM knows what kind of subject
#                                it's summarizing.
#   description_name            Canonical name of the subject (e.g.
#                                "Scrooge" or "Scrooge<->Marley").  Pinned
#                                to start of summary for grounding.
#   cache_type="summary"        Namespace inside upstream's llm_response_cache
#                                separating extract / summarize / query
#                                calls.  Our Go cache (when wired) would
#                                follow the same convention.
