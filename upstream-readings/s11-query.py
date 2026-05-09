# =============================================================================
# Upstream reading for s11 — Dual-level retrieval, four modes
# =============================================================================
#
# Source:  https://github.com/HKUDS/LightRAG (commit on 2026-05-09 main)
# Files:   lightrag/operate.py
#            kg_query                @ L3164-3410 (mode dispatch / synthesis)
#            naive_query             @ L4930-5200 (vector-only baseline)
#            get_keywords_from_query @ L3375-3408
#            extract_keywords_only   @ L3411-3500 (called by the above)
#            _build_query_context    @ L3700-4055 (per-mode context builder)
#          lightrag/prompt.py
#            PROMPTS["keywords_extraction"] @ L374-394 (system prompt)
#
# License: MIT (HKUDS, 2025) — same as this repo.  Excerpts are verbatim
#          unless prefixed with "[s11: omitted ...]" or "[s11: summary]".
#
# This file is what our agents/s11-query-modes/{pipeline.go, keywords.go,
# query_naive.go, query_local.go, query_global.go, query_hybrid.go,
# context_builder.go} ports.  Three correspondences are load-bearing:
#
#   1. ONE KEYWORD CALL FEEDS BOTH MODES.  kg_query asks the LLM for
#      hl_keywords + ll_keywords once, then dispatches to local / global /
#      hybrid.  Hybrid never pays twice for keyword extraction.
#
#   2. CONTEXT IS STRUCTURED, NOT FLAT.  _build_query_context emits THREE
#      sections (entities / relations / chunks), each with its own token
#      budget; the final prompt's MaxTotalTokens applies AFTER all three
#      are composed.  Our context_builder.go does the same.
#
#   3. NAIVE MODE IS FORMALLY THE s01 BASELINE.  naive_query does ONE
#      vector lookup against chunks_vdb and feeds the resulting context to
#      the LLM — no graph, no keywords.  Useful as the floor when comparing
#      hybrid's lift on graph-shaped questions.
#
# Tests in our query_test.go assert all three correspondences.

# -----------------------------------------------------------------------------
# Verbatim excerpt — kg_query mode dispatch (operate.py:3164-3232, ~50 LOC):
# -----------------------------------------------------------------------------

# async def kg_query(
#     query: str,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     relationships_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage,
#     query_param: QueryParam,
#     global_config: dict[str, str],
#     hashing_kv: BaseKVStorage | None = None,
#     system_prompt: str | None = None,
#     chunks_vdb: BaseVectorStorage = None,
# ) -> QueryResult | None:
#     """Execute knowledge graph query and return unified QueryResult."""
#     if not query:
#         return QueryResult(content=PROMPTS["fail_response"])
#
#     # ONE LLM call extracts BOTH high-level and low-level keywords.
#     hl_keywords, ll_keywords = await get_keywords_from_query(
#         query, query_param, global_config, hashing_kv
#     )
#     # Mode-specific empty-keyword warnings, then a global fallback:
#     if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
#         logger.warning("low_level_keywords is empty")
#     if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
#         logger.warning("high_level_keywords is empty")
#     if hl_keywords == [] and ll_keywords == []:
#         if len(query) < 50:
#             ll_keywords = [query]                       # seed with the query
#         else:
#             return QueryResult(content=PROMPTS["fail_response"])
#
#     ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
#     hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""
#
#     # Build the per-mode context window (entities / relations / chunks).
#     context_result = await _build_query_context(
#         query, ll_keywords_str, hl_keywords_str,
#         knowledge_graph_inst, entities_vdb, relationships_vdb,
#         text_chunks_db, query_param, chunks_vdb,
#     )
#     if context_result is None:
#         return None  # nothing to retrieve on
#
#     # [s11: omitted ~140 LOC of cache + only_need_context / only_need_prompt
#     # short-circuits + streaming response handling — orthogonal to the four
#     # modes themselves.  See the full source file for those branches.]
#
#     sys_prompt = PROMPTS["rag_response"].format(
#         response_type=query_param.response_type or "Multiple Paragraphs",
#         user_prompt=query_param.user_prompt or "n/a",
#         context_data=context_result.context,
#     )
#     response = await use_model_func(query, system_prompt=sys_prompt, ...)
#     return QueryResult(content=response, raw_data=context_result.raw_data)

# -----------------------------------------------------------------------------
# Verbatim excerpt — naive_query core (operate.py:4953-5012, ~30 LOC):
# -----------------------------------------------------------------------------

# async def naive_query(
#     query: str,
#     chunks_vdb: BaseVectorStorage,
#     query_param: QueryParam,
#     global_config: dict[str, str],
#     hashing_kv: BaseKVStorage | None = None,
#     system_prompt: str | None = None,
# ) -> QueryResult | None:
#     if not query:
#         return QueryResult(content=PROMPTS["fail_response"])
#     # Single vector lookup against chunks_vdb — NO graph, NO keywords.
#     chunks = await _get_vector_context(query, chunks_vdb, query_param, None)
#     if chunks is None or len(chunks) == 0:
#         return None
#
#     # Compute remaining budget for chunks under MaxTotalTokens after the
#     # system-prompt template + query are accounted for.  buffer_tokens
#     # reserves room for the reference list.
#     max_total_tokens = query_param.max_total_tokens or DEFAULT_MAX_TOTAL_TOKENS
#     pre_sys_prompt = PROMPTS["naive_rag_response"].format(
#         response_type="Multiple Paragraphs", user_prompt="n/a", content_data="",
#     )
#     buffer_tokens = 200
#     available_chunk_tokens = max_total_tokens - (
#         len(tokenizer.encode(pre_sys_prompt))
#         + len(tokenizer.encode(query))
#         + buffer_tokens
#     )
#     processed_chunks = await process_chunks_unified(
#         query=query, unique_chunks=chunks, query_param=query_param,
#         global_config=global_config, source_type="vector",
#         chunk_token_limit=available_chunk_tokens,
#     )
#     # [s11: omitted ~150 LOC of reference-list generation, raw_data
#     # assembly, cache write, and streaming response handling.]
#
#     sys_prompt = PROMPTS["naive_rag_response"].format(
#         response_type=query_param.response_type or "Multiple Paragraphs",
#         user_prompt=query_param.user_prompt or "n/a",
#         content_data="\n".join(c["content"] for c in processed_chunks),
#     )
#     response = await use_model_func(query, system_prompt=sys_prompt, ...)
#     return QueryResult(content=response, raw_data=raw_data)

# -----------------------------------------------------------------------------
# Reading map — where the rest is documented:
# -----------------------------------------------------------------------------
#
#   - The full kg_query body (cache + streaming + only_need_*)
#       → s_full integration chapter pulls this thread end-to-end.
#   - The keyword-extraction prompt and parsing logic
#       → see PROMPTS["keywords_extraction"] (prompt.py:374)
#         our keywords.go has a faithful (not verbatim) summary.
#   - _build_query_context per-mode branches
#       → operate.py:3700-4055; we don't excerpt here because each branch is
#         500+ LOC and the structure (3 sections, 3 budgets) is what the
#         port carries forward.  Our context_builder.go owns the structure.
#   - process_chunks_unified
#       → operate.py:4500-4700; rerank, dedup, token-trim.  We omit the
#         reranker (rerank=False is documented as a Phase G exercise).
