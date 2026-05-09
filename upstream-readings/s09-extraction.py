# =============================================================================
# Upstream reading for s09 — Entity / relation extraction with gleaning
# =============================================================================
#
# Source:  https://github.com/HKUDS/LightRAG (commit on 2026-05-09 main)
# Files:   lightrag/operate.py (extract_entities @ L2883-3170,
#          _process_extraction_result @ L937-1063,
#          _handle_single_entity_extraction @ L386-470)
#          lightrag/prompt.py (entity_extraction_system_prompt @ L11-61,
#          entity_continue_extraction_user_prompt @ L84-100,
#          DEFAULT_TUPLE_DELIMITER / DEFAULT_COMPLETION_DELIMITER @ L8-9)
#
# License: MIT (HKUDS, 2025) — same as this repo.  Excerpts are verbatim
#          unless prefixed with "[s09: omitted ...]" or "[s09: summary]".
#
# This file is what our agents/s09-extraction/{extraction.go, parser.go,
# gleaning.go, cache.go, extraction_prompt.go} ports.  Read this first,
# then open the four Go files side by side.  Three correspondences are
# load-bearing:
#
#   1. DELIMITER CONTRACT (not JSON).  tuple_delimiter "<|#|>" splits fields
#      within a record; "\n" splits records; "<|COMPLETE|>" marks the end of
#      the entire output block.  Entity = 4 fields, Relation = 5 fields.
#      Our parser.go uses strings.Split twice (by \n, then by <|#|>) and
#      drops malformed lines silently — same robustness path as upstream.
#
#   2. GLEANING LOOP.  After the initial extraction, a continuation user
#      prompt is sent with the previous (user, assistant) pair packed into
#      history_messages.  This lets the LLM see what was already extracted
#      and add only what was missed.  Upstream actually does ONE extra
#      round (entity_extract_max_gleaning is 0 or 1, treated as a bool);
#      our Go port supports N rounds for didactic clarity, with early-stop
#      when no new entities appear.
#
#   3. CACHE BY HASH.  llm_response_cache is keyed by hash(prompt_text).
#      Hits skip the LLM entirely.  Our Go port uses sha256(chunk +
#      promptVersion) as the key and a minimal in-memory MemoryKVStore.
#
# Tests in our extraction_test.go assert all three: parsing correctness,
# gleaning accumulation, cache hit on repeat.

# -----------------------------------------------------------------------------
# Verbatim excerpt — system prompt header (lightrag/prompt.py:11-21, ~12 LOC):
# -----------------------------------------------------------------------------

# PROMPTS["DEFAULT_TUPLE_DELIMITER"]      = "<|#|>"
# PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
#
# PROMPTS["entity_extraction_system_prompt"] = """---Role---
# You are a Knowledge Graph Specialist responsible for extracting entities
# and relationships from the input text.
#
# ---Instructions---
# 1.  Entity Extraction & Output:
#     - Identification: Identify clearly defined and meaningful entities ...
#     - Entity Details: For each identified entity, extract entity_name,
#       entity_type (one of {entity_types} or "Other"), entity_description.
#     - Output Format: Output 4 fields delimited by `{tuple_delimiter}` on
#       a single line.  First field MUST be the literal string `entity`.
#       Format: entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description
# ...
# # [s09: upstream continues with sections 2-8 of the system prompt
# # — relationship format spec, delimiter usage protocol, undirected
# # relationship rule, completion signal, language handling.  See full
# # prompt at lightrag/prompt.py:11-61.  Our extraction_prompt.go writes
# # a faithful Go-string summary of all 8 sections, ~30 lines.]
# """

# -----------------------------------------------------------------------------
# Verbatim excerpt — continue prompt (lightrag/prompt.py:84-100, ~17 LOC):
# -----------------------------------------------------------------------------

# PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
# Based on the last extraction task, identify and extract any **missed or
# incorrectly formatted** entities and relationships from the input text.
#
# ---Instructions---
# 1. Strict Adherence to System Format: ... [refers back to system prompt]
# 2. Focus on Corrections/Additions:
#    - Do NOT re-output entities that were correctly extracted last task.
#    - If an entity was missed, extract and output it now.
#    - If a record was truncated/missing fields, re-output corrected version.
# 3. Output Format - Entities: 4 fields delimited by tuple_delimiter, first
#    field MUST be `entity`.
# 4. Output Format - Relationships: 5 fields delimited by tuple_delimiter,
#    first field MUST be `relation`.
# 5. Output Content Only — no introductions or conclusions.
# 6. Completion Signal: Output {completion_delimiter} as the final line.
# 7. Output Language: {language}.
# <Output>
# """

# -----------------------------------------------------------------------------
# extract_entities core path (lightrag/operate.py:2949-3050, ~50 LOC):
# -----------------------------------------------------------------------------

# # Get initial extraction (system prompt is reused across chunks for OpenAI
# # prompt-caching benefit; user prompt has the chunk content)
# entity_extraction_system_prompt = PROMPTS[
#     "entity_extraction_system_prompt"
# ].format(**context_base)
# entity_extraction_user_prompt = PROMPTS[
#     "entity_extraction_user_prompt"
# ].format(**{**context_base, "input_text": content})
# entity_continue_extraction_user_prompt = PROMPTS[
#     "entity_continue_extraction_user_prompt"
# ].format(**{**context_base, "input_text": content})
#
# final_result, timestamp = await use_llm_func_with_cache(
#     entity_extraction_user_prompt,
#     use_llm_func,
#     system_prompt=entity_extraction_system_prompt,
#     llm_response_cache=llm_response_cache,
#     cache_type="extract",
#     chunk_id=chunk_key,
# )
#
# history = pack_user_ass_to_openai_messages(
#     entity_extraction_user_prompt, final_result
# )
#
# # Initial parse (delimiter-based)
# maybe_nodes, maybe_edges = await _process_extraction_result(
#     final_result, chunk_key, timestamp, file_path,
#     tuple_delimiter=context_base["tuple_delimiter"],
#     completion_delimiter=context_base["completion_delimiter"],
# )
#
# # Gleaning round (upstream runs at most 1 extra round)
# if entity_extract_max_gleaning > 0:
#     glean_result, timestamp = await use_llm_func_with_cache(
#         entity_continue_extraction_user_prompt,
#         use_llm_func,
#         system_prompt=entity_extraction_system_prompt,
#         llm_response_cache=llm_response_cache,
#         history_messages=history,
#         cache_type="extract",
#         chunk_id=chunk_key,
#     )
#     glean_nodes, glean_edges = await _process_extraction_result(
#         glean_result, chunk_key, timestamp, file_path,
#         tuple_delimiter=context_base["tuple_delimiter"],
#         completion_delimiter=context_base["completion_delimiter"],
#     )
#     # Merge — new entity: insert.  Existing: keep the longer description.
#     for entity_name, glean_entities in glean_nodes.items():
#         if entity_name in maybe_nodes:
#             if glean_desc_len > original_desc_len:
#                 maybe_nodes[entity_name] = list(glean_entities)
#         else:
#             maybe_nodes[entity_name] = list(glean_entities)

# -----------------------------------------------------------------------------
# Parser core (_process_extraction_result, lightrag/operate.py:937-1010):
# -----------------------------------------------------------------------------

# async def _process_extraction_result(
#     result: str, chunk_key: str, timestamp: int, file_path: str,
#     tuple_delimiter: str = "<|#|>",
#     completion_delimiter: str = "<|COMPLETE|>",
# ) -> tuple[dict, dict]:
#     maybe_nodes = defaultdict(list)
#     maybe_edges = defaultdict(list)
#     if completion_delimiter not in result:
#         logger.warning(f"{chunk_key}: complete delimiter missing")
#
#     # Step 1: split by "\n" + completion_delimiter (case-insensitive)
#     records = split_string_by_multi_markers(
#         result, ["\n", completion_delimiter, completion_delimiter.lower()],
#     )
#
#     # Step 2: fix LLM-side errors (sometimes LLM uses tuple_delimiter as
#     # the record separator instead of \n) — split each record again on
#     # "<|#|>entity<|#|>" / "<|#|>relation<|#|>" markers and re-prefix.
#     # [s09: omitted — ~30 lines of split-by-multi-markers + re-prefix
#     # logic.  See lightrag/operate.py:970-1000.  Our parser.go does NOT
#     # implement this repair path; we keep the parser permissive (skip
#     # malformed lines) instead.]
#
#     # Step 3: per-record dispatch
#     for record in fixed_records:
#         attrs = split_string_by_multi_markers(record, [tuple_delimiter])
#         if attrs[0] == "entity":
#             entity = _handle_single_entity_extraction(attrs, ...)
#             if entity: maybe_nodes[entity["entity_name"]].append(entity)
#         elif attrs[0] == "relation" or attrs[0] == "relationship":
#             rel = _handle_single_relationship_extraction(attrs, ...)
#             if rel: maybe_edges[(rel["src"], rel["tgt"])].append(rel)
#     return maybe_nodes, maybe_edges

# -----------------------------------------------------------------------------
# Reading map — what to read AFTER s09
# -----------------------------------------------------------------------------
#
# Upstream files that consume the entities/relationships s09 extracts:
#
#   lightrag/operate.py:1623-1947  (_merge_nodes_then_upsert)
#   lightrag/operate.py:1948-2300  (_merge_edges_then_upsert)
#       After extraction, when the same entity appears in many chunks,
#       these helpers gather all descriptions, optionally summarize via
#       LLM, then upsert ONE merged record into graph_storage and
#       vdb_entities.  The summarize-or-not heuristic lives at
#       lightrag/operate.py:167-303 (_handle_entity_relation_summary).
#       → Maps to s10 (Map-reduce description summarization).
#
#   lightrag/operate.py:3164-3410  (kg_query → local mode)
#   lightrag/operate.py:3516-4055  (retrieval helpers)
#       Local mode uses the s09-extracted entities as seeds: keyword
#       extraction from the query → vdb_entities.query() → seed entity →
#       graph_storage.get_knowledge_graph() (the BFS we ported in s08) →
#       collect chunks via source_ids → context window.
#       → Maps to s11 (Dual-level retrieval).
#
# What s09 deliberately omits (and why):
#
#   - Token-budget guard for gleaning input.  Upstream computes
#     len(tokenizer.encode(system+history+user)) before each gleaning
#     call and skips the round if it would overflow.  Our Go port skips
#     this guard — MockProvider doesn't care about token counts, and the
#     real OpenAI path can rely on the API to reject over-limit requests.
#     Exercise: add a tokenizer interface and the same guard.
#
#   - description-length-based merge.  Upstream picks the LONGER
#     description when the same entity appears in initial vs gleaning
#     extraction.  Our Go merge is "first writer wins" for didactic
#     simplicity — easy to extend (one if-comparison).
#
#   - cache_type namespace.  Upstream's KV cache stores extract /
#     summarize / query calls in one store, separated by a `cache_type`
#     prefix.  Our Go port only caches extract, so we use a flat key.
#
#   - file_path / timestamp metadata on each entity record.  Upstream
#     keeps these for citation and source-id-limit truncation logic
#     (s10/s11 territory).  We carry only SourceIDs (chunk IDs).
#
#   - The repair path inside _process_extraction_result.  Upstream
#     actively fixes LLM mistakes (tuple_delimiter-as-record-separator,
#     missing prefix).  We rely on permissive parsing — drop malformed
#     lines, keep going.  Trade-off: lower recall on broken outputs vs
#     simpler code.  Add the repair path if you see real LLM emits with
#     >5% malformed lines.
#
# -----------------------------------------------------------------------------
# Glossary one-liners
# -----------------------------------------------------------------------------
#
#   tuple_delimiter            "<|#|>" — splits fields within ONE record.
#                               Used because it's vanishingly unlikely to
#                               appear in natural text or LLM hallucination.
#   completion_delimiter       "<|COMPLETE|>" — marks end-of-block.  We
#                               strip it before splitting; absence triggers
#                               a warning but is not an error.
#   record separator           "\n" — splits records.  NOT a delimiter
#                               constant in the prompt; just a Unix newline.
#                               This is what makes descriptions safe to
#                               include free-form punctuation including <|#.
#   gleaning                   N-round continuation loop after initial
#                               extraction.  Each round uses the previous
#                               (user, assistant) pair as history_messages
#                               so the LLM "remembers" what it already
#                               found and adds only the missing bits.
#   entity_extract_max_gleaning  Upstream config; default 1.  Semantically
#                               a bool ("do one extra round or skip").
#                               Our Go MaxGleaningRounds is N rounds with
#                               early-stop on zero new entities.
#   llm_response_cache         Upstream BaseKVStorage instance keyed by
#                               hash(prompt_text).  Three cache_type
#                               namespaces (extract / summarize / query)
#                               share one store.  Our cache.go keeps it
#                               flat — only extract is cached at this layer.
