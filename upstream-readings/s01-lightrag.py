# Source: https://github.com/HKUDS/LightRAG  (commit on default branch `main`, 2026-05-09)
# License: MIT (Copyright 2025 LightRAG Team)
# Excerpted for learn-lightrag/s01-minimum-loop — annotated, not for execution.
#
# This file collects three small slices of upstream that s01 condenses into
# ~1000 LOC of Go:
#
#   1. examples/lightrag_openai_demo.py — the user-facing quickstart we mirror
#   2. lightrag/lightrag.py:1237  — `ainsert()`        (s01 Pipeline.Insert)
#   3. lightrag/lightrag.py:2622  — `aquery()`         (s01 Pipeline.Query)
#   4. lightrag/lightrag.py:2884  — `aquery_llm()`     (s01 Pipeline.Query, mode='naive' branch)
#
# Each block is followed by reading notes that point out: what we kept, what
# we dropped, and what the code we cite hands off to in later chapters.

# ─────────────────────────────────────────────────────────────────────────────
# (1) examples/lightrag_openai_demo.py — the user-facing quickstart
# ─────────────────────────────────────────────────────────────────────────────
# Trimmed to the load-bearing lines; full file is ~187 lines, mostly logging
# config we deliberately skip in s01 (Go has stdlib log already).

WORKING_DIR = "./dickens"

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,            # → s01 Pipeline (no working_dir; s05 adds JSON persist)
        embedding_func=openai_embed,        # → s01 OpenAIEmbedder + EmbeddingProvider interface
        llm_model_func=gpt_4o_mini_complete,# → s01 OpenAIProvider + Provider interface
    )
    await rag.initialize_storages()         # → s01 stub: storage init is in NewPipeline
    return rag

async def main():
    rag = await initialize_rag()

    # Ingestion: read a text file, hand the whole string to ainsert().
    with open("./book.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())          # → s01 pipe.Insert(ctx, "book", string(text))

    # Four query modes against the same question. s01 only does mode="naive";
    # s11 implements local / global / hybrid.
    print(await rag.aquery(
        "What are the top themes in this story?",
        param=QueryParam(mode="naive")
    ))                                       # → s01 pipe.Query(ctx, q, topK=3)

    await rag.finalize_storages()            # → s01: nothing to finalize (in-mem); s05 fsyncs JSON

# Reading notes (demo):
# - Upstream's `working_dir` becomes a `lightrag-data/` directory that holds 7+
#   JSON files (kv_store_*.json, vdb_chunks.json, graph_chunk_entity_relation.graphml).
#   s01 keeps everything in memory; s05 lands the JSON KV layout, s07 the vector
#   serialization, s08 the graph dump.
# - `gpt_4o_mini_complete` is a thin async wrapper around an OpenAI HTTP call.
#   s01's OpenAIProvider has the same shape minus the streaming hooks.
# - The 4-mode comparison loop ("naive", "local", "global", "hybrid") is what
#   s11 reproduces — s01 only ships mode="naive".

# ─────────────────────────────────────────────────────────────────────────────
# (2) lightrag/lightrag.py:1237  — ainsert()
# ─────────────────────────────────────────────────────────────────────────────

async def ainsert(
    self,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    track_id: str | None = None,
) -> str:
    """Async Insert documents with checkpoint support."""
    # 1. Generate a track_id (UUID-ish string) for progress monitoring.
    #    → s01 skips this; we use docID directly. s03 brings the doc-status
    #      state machine, where track_id becomes load-bearing.
    if track_id is None:
        track_id = generate_track_id("insert")

    # 2. Enqueue: write doc(s) to doc-status KV with status=PENDING, dedup
    #    against existing docs by content hash.
    #    → s01: just chunk + embed inline. No dedup.
    #    → s03 + s05: introduce DocStatusStore + KVStore.FilterMissing.
    await self.apipeline_enqueue_documents(input, ids, file_paths, track_id)

    # 3. Process: chunk → embed → extract → upsert. Bounded by an asyncio
    #    semaphore so 100 docs don't overwhelm the LLM API.
    #    → s01: only chunk + embed; no extraction. s09 adds extraction loop.
    await self.apipeline_process_enqueue_documents(
        split_by_character, split_by_character_only
    )

    return track_id

# Reading notes (ainsert):
# - The `split_by_character` / `split_by_character_only` flags are an
#   alternative chunking path for non-English text. s01 doesn't expose them;
#   s04 introduces both in chunking.go.
# - `apipeline_enqueue_documents` does content-MD5 dedup. The s03 chapter
#   makes this explicit: re-inserting the same text is a no-op (PROCESSED
#   status short-circuits).
# - `apipeline_process_enqueue_documents` runs:
#     for chunk in chunks:
#         embed(chunk) → vdb.upsert
#         extract_entities(chunk) → graph.upsert + entity_vdb.upsert
#         summarize_descriptions(...) when merge threshold exceeded
#   s01 only does the embed+vdb step. Each subsequent chapter peels off one
#   stage of this loop.

# ─────────────────────────────────────────────────────────────────────────────
# (3) lightrag/lightrag.py:2622  — aquery()  (backward-compat wrapper)
# ─────────────────────────────────────────────────────────────────────────────

async def aquery(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    """Backward-compat wrapper around aquery_llm — returns just the answer string."""
    # The real work is in aquery_llm (next slice). aquery() is here only because
    # very old user code expected `result = await rag.aquery(...)` and got
    # a string back, not a dict.
    result = await self.aquery_llm(query, param, system_prompt)
    llm_response = result.get("llm_response", {})
    if llm_response.get("is_streaming"):
        return llm_response.get("response_iterator")    # → s11 / Phase G stream
    return llm_response.get("content", "")              # → s01 returns res.Content as str

# Reading notes (aquery):
# - This wrapper is a perfect example of API evolution: the function used to
#   do the real work and return a string; later it grew a richer return type
#   so it now delegates and unwraps. s01's Pipeline.Query is the simplified
#   dict-aware version (it returns a QueryResult struct directly).
# - Streaming responses (`is_streaming=True`) are not implemented in s01.
#   The `Stream bool` field on s01 CompleteRequest is reserved for s11 + Phase G.

# ─────────────────────────────────────────────────────────────────────────────
# (4) lightrag/lightrag.py:2884  — aquery_llm()  (the real work)
# ─────────────────────────────────────────────────────────────────────────────

async def aquery_llm(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Asynchronous complete query API: structured retrieval + LLM generation."""
    global_config = asdict(self)            # all the constructor params, as dict

    if param.mode in ["local", "global", "hybrid", "mix"]:
        # Graph-aware modes: extract keywords, traverse KG, gather chunks.
        # → s11 implements all four.
        query_result = await kg_query(
            query.strip(),
            self.chunk_entity_relation_graph,    # → s08 GraphStore
            self.entities_vdb,                   # → s07 VectorStore (entities index)
            self.relationships_vdb,              # → s07 VectorStore (relations index)
            self.text_chunks,                    # → s05 KVStore (chunk content)
            param,
            global_config,
            hashing_kv=self.llm_response_cache,  # → s05 KVStore (LLM response cache)
            system_prompt=system_prompt,
            chunks_vdb=self.chunks_vdb,          # → s07 VectorStore (chunks index)
        )
    elif param.mode == "naive":
        # The simple path: vector similarity over chunks only. No graph.
        # → s01 Pipeline.Query implements exactly this.
        query_result = await naive_query(
            query.strip(),
            self.chunks_vdb,
            param,
            global_config,
            hashing_kv=self.llm_response_cache,
            system_prompt=system_prompt,
        )
    elif param.mode == "bypass":
        # Don't retrieve at all; ask the LLM cold. Diagnostic only.
        # → not implemented in s01..s11; mentioned in Appendix A.
        ...

    # Then assemble {"llm_response": {...}, "data": {...}, "metadata": {...}}
    # and return. s01 collapses this to QueryResult{Content, References, Mode}.

# Reading notes (aquery_llm):
# - The four if/elif branches are the dispatch you'll re-implement in s11 as
#   `Pipeline.Query(ctx, q, QueryParam{Mode: ...})`.
# - `naive_query` is what s01's Pipeline.Query mirrors. It does roughly:
#       q_emb = embed(query)
#       hits = chunks_vdb.query(q_emb, top_k)
#       chunks = text_chunks.get_by_ids(hits)
#       context = build_context(chunks, max_total_tokens)
#       answer = llm(system_prompt + context, query)
#   That's exactly the s01 Pipeline.Query body, ~30 lines simpler.
# - `hashing_kv=self.llm_response_cache` is a per-(mode, hash) cache that
#   skips LLM calls on identical queries. s01 omits caching (queries rarely
#   repeat in a quickstart); s09 reintroduces it via a hash-keyed wrapper.

# ─────────────────────────────────────────────────────────────────────────────
# Reading map (where to go next)
# ─────────────────────────────────────────────────────────────────────────────
# After s01 you've seen the loop's shape. Each later chapter deepens one piece:
#
#   s02 → lightrag/llm/openai.py            (Provider gets retry + functional opts)
#   s03 → lightrag/base.py:662-697 +
#         lightrag/kg/json_doc_status_impl.py   (DocStatus state machine)
#   s04 → lightrag/operate.py:102-166         (chunking_by_token_size with overlap)
#   s05 → lightrag/kg/json_kv_impl.py         (KVStore with FilterMissing + persist)
#   s06 → lightrag/llm/openai.py (embed slice)(EmbeddingProvider with batching)
#   s07 → lightrag/kg/nano_vector_db_impl.py  (VectorStore with thresholding + 3 indices)
#   s08 → lightrag/kg/networkx_impl.py        (GraphStore with BFS subgraph)
#   s09 → lightrag/operate.py:~2883-3163      (extract_entities + gleaning)
#   s10 → lightrag/operate.py:167-385         (summarization decision tree)
#   s11 → lightrag/operate.py:3164-3410       (kg_query: local / global / hybrid / mix)
#
# Combined, those 11 sessions reconstruct the full LightRAG insert + query path.
