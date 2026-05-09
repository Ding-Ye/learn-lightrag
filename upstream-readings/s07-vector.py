# =============================================================================
# Upstream reading for s07 — Cosine-similarity vector store
# =============================================================================
#
# Source:  https://github.com/HKUDS/LightRAG (commit on 2026-05-09 main)
# File:    lightrag/kg/nano_vector_db_impl.py
# Lines:   24-150 (class def + upsert + query) + 245-268 (index_done_callback)
#
# License: MIT (HKUDS, 2025) — same as this repo. Excerpt is verbatim with
#          ellipsis comments marked "[s07: omitted ...]".
#
# This is the file our agents/s07-vector-store/cosine_index.go ports. Read
# this first, then open the Go file side-by-side. Three correspondences are
# load-bearing:
#
#   1. THREE indices, ONE class. Upstream instantiates NanoVectorDBStorage
#      three separate times (one each for chunks / entities / relations) with
#      different `namespace` values; the file path becomes
#      `vdb_<namespace>.json`. Our s07/main.go does exactly the same — we
#      `NewCosineIndex(dir, "chunks")` etc., and the dispatch lives in the
#      pipeline, not inside the store.
#
#   2. cosine_better_than_threshold is the per-store threshold. Upstream
#      reads it from `vector_db_storage_cls_kwargs` config; our Go port
#      passes it as a per-Query argument so different callers can pick
#      different thresholds without rebuilding the index. Default values
#      (0.2-0.4) discussed in docs/{zh,en}/s07-vector-store.md.
#
#   3. float16 + zlib + base64 encoding is the upstream's wire-size win.
#      Two reasons we omit it:
#        - readability: a JSON dump with raw float32 arrays is greppable;
#          a base64-of-zlib-of-bytes blob is opaque.
#        - stdlib-only: no extra dependencies needed.
#      Trade-off: our on-disk file is ~3-4× larger. Acceptable at chapter
#      scale (≤ 10K records); production-grade users should re-implement.
#
# Tests in our cosine_index_test.go assert the first two properties (three
# distinct namespaces, threshold filter); the third is left as exercise.

# -----------------------------------------------------------------------------
# Verbatim excerpt (lightrag/kg/nano_vector_db_impl.py:24-150, ~50 LOC):
# -----------------------------------------------------------------------------

# @final
# @dataclass
# class NanoVectorDBStorage(BaseVectorStorage):
#     def __post_init__(self):
#         self._validate_embedding_func()
#         self._client = None
#         self._storage_lock = None
#         self.storage_updated = None
#
#         kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
#         cosine_threshold = kwargs.get("cosine_better_than_threshold")
#         if cosine_threshold is None:
#             raise ValueError(
#                 "cosine_better_than_threshold must be specified in "
#                 "vector_db_storage_cls_kwargs"
#             )
#         self.cosine_better_than_threshold = cosine_threshold
#         # [s07: omitted — workspace path & namespace separator logic]
#         self._client_file_name = os.path.join(
#             workspace_dir, f"vdb_{self.namespace}.json"
#         )
#         self._max_batch_size = self.global_config["embedding_batch_num"]
#         self._client = NanoVectorDB(
#             self.embedding_func.embedding_dim,
#             storage_file=self._client_file_name,
#         )
#
#     async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
#         if not data:
#             return
#         current_time = int(time.time())
#         list_data = [
#             {
#                 "__id__": k,
#                 "__created_at__": current_time,
#                 **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
#             }
#             for k, v in data.items()
#         ]
#         contents = [v["content"] for v in data.values()]
#         batches = [
#             contents[i : i + self._max_batch_size]
#             for i in range(0, len(contents), self._max_batch_size)
#         ]
#         embedding_tasks = [
#             self.embedding_func(batch, context="document") for batch in batches
#         ]
#         embeddings_list = await asyncio.gather(*embedding_tasks)
#         embeddings = np.concatenate(embeddings_list)
#         if len(embeddings) == len(list_data):
#             for i, d in enumerate(list_data):
#                 # Compress vector using Float16 + zlib + Base64
#                 vector_f16 = embeddings[i].astype(np.float16)
#                 compressed_vector = zlib.compress(vector_f16.tobytes())
#                 encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
#                 d["vector"] = encoded_vector
#                 d["__vector__"] = embeddings[i]
#             client = await self._get_client()
#             results = client.upsert(datas=list_data)
#             return results

# -----------------------------------------------------------------------------
# query (lightrag/kg/nano_vector_db_impl.py:124-150, ~25 LOC):
# -----------------------------------------------------------------------------

#     async def query(
#         self, query: str, top_k: int, query_embedding: list[float] = None
#     ) -> list[dict[str, Any]]:
#         if query_embedding is not None:
#             embedding = query_embedding
#         else:
#             embedding = await self.embedding_func(
#                 [query], context="query", _priority=5
#             )
#             embedding = embedding[0]
#         client = await self._get_client()
#         results = client.query(
#             query=embedding,
#             top_k=top_k,
#             better_than_threshold=self.cosine_better_than_threshold,
#         )
#         results = [
#             {
#                 **{k: v for k, v in dp.items() if k != "vector"},
#                 "id": dp["__id__"],
#                 "distance": dp["__metrics__"],
#                 "created_at": dp.get("__created_at__"),
#             }
#             for dp in results
#         ]
#         return results

# -----------------------------------------------------------------------------
# index_done_callback (lightrag/kg/nano_vector_db_impl.py:245-268, abridged):
# -----------------------------------------------------------------------------

#     async def index_done_callback(self) -> bool:
#         """Save data to disk"""
#         async with self._storage_lock:
#             if self.storage_updated.value:
#                 # Storage was updated by another process, reload data
#                 self._client = NanoVectorDB(
#                     self.embedding_func.embedding_dim,
#                     storage_file=self._client_file_name,
#                 )
#                 self.storage_updated.value = False
#                 return False
#         async with self._storage_lock:
#             self._client.save()
#             await set_all_update_flags(self.namespace, workspace=self.workspace)
#             self.storage_updated.value = False
#             return True

# -----------------------------------------------------------------------------
# Reading map — what to read AFTER s07
# -----------------------------------------------------------------------------
#
# Upstream files that produce / consume the records this store holds:
#
#   lightrag/llm/openai.py  (the embedding wrapper)
#       The function that turns text into the [][]float32 we Upsert here.
#       → Maps to s06 (Embedding provider with batching) in this repo. The
#         same EmbeddingProvider feeds all THREE indices — chunks, entities,
#         relations — which is why s07 doesn't own the embedder.
#
#   lightrag/operate.py:~2883-3163  (extract_entities)
#       Where entities and relations get extracted from chunks and pushed
#       into vdb_entities + vdb_relationships (each with its own
#       NanoVectorDBStorage instance).
#       → Maps to s09 (Entity/relation extraction with gleaning). After
#         extraction, the entity description is embedded and Upserted into
#         the entities namespace; same for the relation keywords-string into
#         the relations namespace.
#
#   lightrag/operate.py:3164-3410  (kg_query — local/global/hybrid)
#       Where Query() finally gets called, three times, against three
#       different stores depending on the QueryParam.mode.
#       → Maps to s11 (Dual-level retrieval, four modes). naive mode hits
#         vdb_chunks only; local mode hits vdb_entities; global mode hits
#         vdb_relationships; hybrid mode hits all three.
#
# What s07 deliberately omits (and why):
#
#   - float16 + zlib + base64 encoding.  Upstream packs each vector as
#     base64(zlib(float16_bytes)) for ~3-4× wire-size reduction.  We use
#     raw float32 JSON for code transparency.  Exercise: add a
#     `WithCompression()` option that re-implements this.
#
#   - Cross-process update flags.  Upstream uses get_update_flag /
#     set_all_update_flags to coordinate between Python processes sharing
#     a single working_dir.  Our Go port uses one process and a single
#     sync.RWMutex — no IPC needed.  Exercise: add fcntl-based file locking
#     for multi-process safety.
#
#   - `delete_entity` / `delete_entity_relation` helpers.  Upstream provides
#     entity-name-aware deletes (compute MD5 of name, then delete).  Our
#     port keeps Delete generic on []ids — the caller hashes if needed.
#     The "MD5 of entity_name with prefix=ent-" rule lives in s09.
#
#   - `__created_at__` auto-stamping.  Upstream stamps int(time.time()) on
#     every record so the caller can reason about staleness.  We leave
#     timestamps to the caller's metadata bag — one less hardcoded field.
#
# -----------------------------------------------------------------------------
# Glossary one-liners
# -----------------------------------------------------------------------------
#
#   namespace                      — string key picking which of the three
#                                    indices to use (chunks / entities /
#                                    relations).  Determines `vdb_<ns>.json`.
#   cosine_better_than_threshold   — global threshold (0.2-0.4) below which
#                                    a record is filtered out.  Inclusive.
#                                    → Our Query()'s `threshold` arg.
#   embedding_batch_num            — how many texts go in one embed call.
#                                    Owned by s06's EmbeddingProvider, not
#                                    by s07.
#   index_done_callback            — save snapshot to disk + propagate flags.
#                                    → Our Persist() (no flag handling).
#   meta_fields                    — set of metadata keys allowed on a
#                                    record.  Upstream filters input data by
#                                    this whitelist; our Go port accepts an
#                                    open map[string]any.
#   __id__ / __created_at__        — upstream's reserved record fields.
#                                    Equivalent to our VectorRecord.ID and
#                                    (caller-managed) Metadata["created_at"].
