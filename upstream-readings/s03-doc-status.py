# Source: https://github.com/HKUDS/LightRAG  (paths below, branch `main`, 2026-05-09)
# License: MIT (Copyright 2025 LightRAG Team)
# Excerpted for learn-lightrag/s03-doc-status — annotated, not for execution.
#
# This file collects two upstream slices that s03 condenses into Go:
#
#   1. lightrag/base.py:662-755  — DocStatus enum + DocProcessingStatus
#                                   dataclass + DocStatusStorage abstract class.
#                                   These are the SHAPES; s03 re-types them
#                                   verbatim (with Go conventions) in
#                                   doc_status.go.
#
#   2. lightrag/kg/json_doc_status_impl.py — JsonDocStatusStorage, the
#                                   reference JSON-backed impl. s03's
#                                   doc_status_store.go is the Go port,
#                                   minus upstream's namespace/workspace
#                                   prefix and shared-storage lock layer.
#
# Each block is followed by reading notes that point out: what s03 kept,
# what s03 simplified, and which later chapter picks up the loose ends.

# ─────────────────────────────────────────────────────────────────────────────
# (1) lightrag/base.py:662-706 — the data shapes
# ─────────────────────────────────────────────────────────────────────────────

class DocStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    PREPROCESSED = "preprocessed"      # → s03 SKIPS this (multimodal-only state)
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content_summary: str               # first 100 chars; s03: summarize() helper
    content_length: int
    file_path: str
    status: DocStatus
    created_at: str                    # ISO timestamp; s03 uses time.Time + JSON
    updated_at: str
    track_id: str | None = None
    chunks_count: int | None = None
    chunks_list: list[str] | None = field(default_factory=list)
    error_msg: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    multimodal_processed: bool | None = field(default=None, repr=False)  # → s03 skips

    def __post_init__(self):
        """Handle status conversion based on multimodal_processed field."""
        if self.multimodal_processed is not None:
            if (
                self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED
            ):
                self.status = DocStatus.PREPROCESSED   # → only relevant if multimodal


# Reading notes (data shapes):
#
# - Why s03 drops PREPROCESSED + multimodal_processed: the curriculum's s03
#   teaches the textual ingestion lifecycle only. The upstream PREPROCESSED
#   state means "text done, but vision/audio chunks pending." Adding it here
#   would dilute the four-state teaching point with a fifth state that nothing
#   in s01..s11 actually triggers.
#
# - Field rename rationale: Go convention is exported PascalCase; the JSON tags
#   `"content_summary"` etc. preserve byte-level interop with upstream's
#   `kv_store_doc_status.json` so a learner can mount upstream's working_dir.
#
# - `content_summary` heuristic (`first 100 chars`) is reproduced as `summarize`
#   in doc_status_store.go. Upstream doesn't break on word boundaries and
#   neither do we — the field is for human eyeballing in dashboards, not for
#   semantic search.
#
# - `created_at: str` is a plain string upstream (whoever calls upsert formats
#   the timestamp). s03 uses `time.Time` so JSON marshalling handles RFC3339
#   for free; when reading upstream's files that requires a custom decoder
#   in s05's KV chapter — flagged as a TODO there.

# ─────────────────────────────────────────────────────────────────────────────
# (2) lightrag/kg/json_doc_status_impl.py — JSON impl, load-bearing 30 LOC
# ─────────────────────────────────────────────────────────────────────────────
# Imports trimmed; only the lines our Go port actually mirrors.

@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """
        if not data:
            return
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
            if "chunks_list" not in doc_data:
                doc_data["chunks_list"] = []          # → s03: always init []string{}
            await _cooperative_yield(i)
        async with self._storage_lock:
            self._data.update(data)                   # → s03: docs[id] = rec under mu.Lock()
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()              # → s03: caller-driven Persist()

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )
                # write JSON; sanitization step omitted from this excerpt
                needs_reload = write_json(data_dict, self._file_name)  # → s03 atomic-rename
                # ... (omitted: needs_reload sanitize re-load branch — not in s03 scope)
                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        return await self.get_docs_by_statuses([status])    # → s03: ListByStatus directly

    # ... (omitted: get_status_counts / get_docs_paginated / drop / delete /
    #      get_doc_by_file_path / get_docs_by_track_id — these are dashboard
    #      conveniences, not load-bearing for the state machine itself.
    #      s03 ships ListByStatus only; s05 will add the rest as exercises.)


# Reading notes (JsonDocStatusStorage):
#
# - The upsert flow is "filter then merge" — but the filter step is one layer
#   higher up the pipeline, in `lightrag/lightrag.py:apipeline_enqueue_documents`,
#   which calls `filter_keys` against this same storage to drop already-
#   processed docs before calling upsert. s03 collapses both layers into
#   `Enqueue(docID, content, filePath) (rec, fresh, err)`: if the docID is
#   already present, return (existing, false, nil) without touching the record.
#   Same dedup property, fewer moving parts.
#
# - `_cooperative_yield(i)` is upstream's hand-rolled `await asyncio.sleep(0)`
#   every N items so a long upsert doesn't starve other coroutines. s03 just
#   takes a single sync.RWMutex.Lock() around the whole map mutation; the
#   teaching trade-off here is "one lock at the top is easier to reason about
#   than per-key cooperative yields, at the cost of throughput at scale". s05
#   tightens this with per-key sync.Map[string]*sync.Mutex.
#
# - `write_json` does atomic-rename + JSON sanitization upstream. s03's
#   Persist() does the atomic-rename half (write to .tmp, fsync-implied via
#   os.WriteFile, rename). The sanitization half (replacing NaN/Inf floats
#   with null) is upstream-specific and unnecessary for s03's data model
#   which has no float fields.
#
# - `index_done_callback` is upstream's "fsync at end of batch" hook. s03
#   makes this an explicit Persist() call so test code can verify the on-
#   disk file at a known moment. Production code mirroring upstream's flow
#   would call Persist() at the end of every Insert pipeline.

# ─────────────────────────────────────────────────────────────────────────────
# Reading map (where to go next)
# ─────────────────────────────────────────────────────────────────────────────
# After s03 you've seen the doc-status state machine. From here:
#
#   s05 → lightrag/kg/json_kv_impl.py — the parent abstraction (BaseKVStorage)
#         that JsonDocStatusStorage inherits from. s05's KVStore reuses the
#         atomic-rename + per-key locking pattern shown here, generalized to
#         any-key any-value storage. After s05, s03's DocStatusStore could be
#         re-implemented in 30 lines on top of the s05 KV impl (good exercise).
#
#   s09 → lightrag/operate.py:~2883 (extract_entities) — this is where the
#         resume-on-failure scan actually pays off. The extractor's first
#         step on every restart is `ListByStatus(DocStatusFailed)` plus
#         `ListByStatus(DocStatusProcessing)` (in case a worker was killed
#         mid-extraction); both lists are fed back into the queue without
#         re-chunking the documents. s03's ChunksList field is what lets
#         s09 know which chunk IDs were already produced and skip them.
#
#   s_full → the 16-step end-to-end trace puts doc-status at step 3 (the
#         filter-then-upsert that fronts every Insert call). Re-read the
#         trace once you finish s11 — you'll see how the same six methods
#         on DocStatusStore get called from five different points in the
#         pipeline.
