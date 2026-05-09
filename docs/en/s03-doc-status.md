---
title: "s03 · Document status state machine"
chapter: 3
slug: s03-doc-status
est_read_min: 9
---

# s03 · Document status state machine

> What this teaches: pull the document lifecycle PENDING → PROCESSING → PROCESSED/FAILED out into its own persisted, transition-validated `DocStatusStore`. Combined with MD5 content-addressing for the doc ID, two upstream killer features fall out for free: "re-inserting the same document is a no-op" and "a half-failed run resumes by completing only what's missing". No chunking, no embeddings — just the bookkeeping.

---

## Problem / 问题

s01's minimum loop has no concept of doc status — `Insert(text)` goes straight through chunk → embed → store with no record of having seen the doc. Three concrete pain points fall out:

1. **Re-inserting burns tokens.** Run `go run .` twice on the same `book.txt` and the second run re-embeds all 80 chunks. With real OpenAI on a long doc, every redo is dollars wasted.
2. **No resume on failure.** Embed fails on chunk 47 of N because OpenAI returns 503; Insert errors out and the 46 already-embedded chunks are wasted. On restart there's nothing to tell the new process where the previous one stopped.
3. **Multi-doc pipelines have no progress signal.** A user calls Insert on a folder of 10 docs; some are chunking, some embedding, some done. The UI can't draw a progress bar; CI can't assert "everything that should have completed did".

Upstream LightRAG solves it by attaching a `DocProcessingStatus` record to each doc, persisted in a `JsonDocStatusStorage`. That's the load-bearing implementation behind LightRAG's "resume on failure" pitch. s03 carves the same mechanism into ~250 LOC of Go without dragging chunking / embedding / storage into the explanation.

## Solution / 解决方案

Four pieces:

1. **`DocStatus` is a 4-value string enum:** `pending` / `processing` / `processed` / `failed`. Strings, not ints, so JSON dumps are human-readable and line up with upstream's on-disk format without a translation table.
2. **`DocProcessingStatus` is a plain struct** — 13 fields mirroring the upstream dataclass: `DocID` / `ContentSummary` / `ContentLength` / `FilePath` / `Status` / `CreatedAt` / `UpdatedAt` / `TrackID` / `ChunksCount` / `ChunksList` / `ErrorMsg` / `Metadata`.
3. **`IsValidTransition(from, to)` is the load-bearing rule table:** four legal edges, PROCESSED is terminal, and `FAILED → PROCESSING` is the retry edge — that retry edge is the entire point of resume-on-failure. Any non-listed transition produces a `*transitionError` that wraps the sentinel `ErrInvalidTransition`, so callers can `errors.Is(err, ErrInvalidTransition)`.
4. **`DocStatusStore` is the locked + JSON-persisted store** — `map[string]*DocProcessingStatus` + `sync.RWMutex`. `Persist` writes to a `.tmp` sibling then `os.Rename`s atomically; readers always see the old file or the new file, never half-written. The constructor calls `Load` so an existing JSON file restores state.

`Enqueue` is the dedup gateway: first call inserts a PENDING record and returns `(rec, true, nil)`; second call with the same docID returns `(existing, false, nil)` and **does not overwrite** the existing record. This collapses upstream's two-step "filter_keys then upsert" (in `apipeline_enqueue_documents`) into a single locked operation — same dedup semantics, fewer moving parts.

## How It Works / 工作原理

```
       Enqueue(docID, content, filePath)
                │
                ├─ already present? ──┐
                │                     ▼
                │   return (existing, false, nil)
                │   no side effects
                │
                ▼ fresh insert
        ┌──────────────┐
        │   PENDING    │
        └──────┬───────┘
               │ MarkProcessing
               ▼
        ┌──────────────┐
        │  PROCESSING  │◀─────────┐
        └──┬─────────┬─┘          │ MarkProcessing (retry after FAILED)
           │         │            │
 MarkProcessed    MarkFailed      │
           │         │            │
           ▼         ▼            │
   ┌───────────┐ ┌────────┐      │
   │ PROCESSED │ │ FAILED │──────┘
   │ (terminal)│ └────────┘
   └───────────┘
```

The load-bearing 30 lines (excerpt from [`agents/s03-doc-status/doc_status_store.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s03-doc-status/doc_status_store.go)):

```go
func (s *DocStatusStore) Enqueue(ctx context.Context, docID, content, filePath string) (*DocProcessingStatus, bool, error) {
    if err := ctx.Err(); err != nil {
        return nil, false, err
    }
    s.mu.Lock()
    defer s.mu.Unlock()

    if existing, ok := s.docs[docID]; ok {
        // Re-insert is a no-op — the killer dedup property.
        return existing, false, nil
    }
    now := s.Clock()
    rec := &DocProcessingStatus{
        DocID:          docID,
        ContentSummary: summarize(content),
        ContentLength:  len(content),
        FilePath:       filePath,
        Status:         DocStatusPending,
        CreatedAt:      now,
        UpdatedAt:      now,
        ChunksList:     []string{}, // never nil — upstream's chunks_list factory
    }
    s.docs[docID] = rec
    return rec, true, nil
}

func (s *DocStatusStore) transition(ctx context.Context, docID string, to DocStatus, mutate func(*DocProcessingStatus)) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    rec, ok := s.docs[docID]
    if !ok {
        return fmt.Errorf("doc-status: %s: not found", docID)
    }
    if !IsValidTransition(rec.Status, to) {
        return &transitionError{DocID: docID, From: rec.Status, To: to}
    }
    rec.Status = to
    rec.UpdatedAt = s.Clock()
    if mutate != nil { mutate(rec) }
    return nil
}
```

**4 non-obvious points**:

1. **`Enqueue`'s `(rec, fresh, err)` triple is the design point.** Callers branch on `fresh==true` to decide whether to keep going (chunking + embedding). If `fresh==false` the doc is already known — even if its status is FAILED, we don't reset it; the caller instead does `MarkProcessing(FAILED→PROCESSING)` to retry, and `ChunksList` / `CreatedAt` / `FilePath` are all preserved.
2. **MD5, not sha256.** Upstream uses MD5 for content-addressing (`compute_mdhash_id` in `lightrag/utils.py`). We use MD5 too so docID stays byte-equal with upstream's JSON files. This is a "content address hash", not a "security hash" — collision resistance under deliberate attack is not a relevant property here.
3. **`Persist` uses atomic rename.** `os.WriteFile(tmp, ..., 0o644)` + `os.Rename(tmp, target)`. POSIX guarantees rename is atomic, so readers see either the complete old file or the complete new file — never a half-written one. Same pattern as upstream's `write_json`.
4. **`ChunksList` is always defensive-copied.** `MarkProcessed` / `Get` / `ListByStatus` never return the internal slice directly — they `append([]string(nil), src...)` to clone it. A caller doing `sort.Strings` on the returned slice can't accidentally mutate the in-memory truth. This is a "value-semantics first" Go idiom, and it heads off a class of subtle bugs.

## What Changed / 与 s01 的变化

```diff
 // s01 pipeline.go: Insert has no doc-status concept at all
-func (p *Pipeline) Insert(ctx context.Context, text string) error {
-    chunks := p.chunker.Chunk(text)
-    embs, _ := p.embedder.Embed(ctx, chunkContents(chunks))
-    for i, e := range embs {
-        p.vectorStore.Upsert(VectorRecord{ID: fmt.Sprintf("c-%d", i), Vector: e})
-    }
-    return nil
-}

+// s03: Insert grows a state machine on the front
+func (p *Pipeline) Insert(ctx context.Context, text, filePath string) error {
+    docID := MD5DocID(text)                              // content-addressing
+    rec, fresh, err := p.docStatus.Enqueue(ctx, docID, text, filePath)
+    if err != nil { return err }
+    if !fresh {
+        // Already processed (or processing / failed-awaiting-retry) — return zero-side-effect.
+        log.Printf("doc %s already known: status=%s", docID[:8], rec.Status)
+        return nil
+    }
+
+    // Actually start processing: PROCESSING.
+    if err := p.docStatus.MarkProcessing(ctx, docID); err != nil { return err }
+
+    chunks := p.chunker.Chunk(text)
+    chunkIDs := make([]string, len(chunks))
+    for i := range chunks { chunkIDs[i] = fmt.Sprintf("%s::chunk-%d", docID, i) }
+
+    embs, err := p.embedder.Embed(ctx, chunkContents(chunks))
+    if err != nil {
+        // Failure preserves status + error msg; next boot's ListByStatus(FAILED) finds it.
+        _ = p.docStatus.MarkFailed(ctx, docID, err.Error())
+        return err
+    }
+    for i, e := range embs { p.vectorStore.Upsert(VectorRecord{ID: chunkIDs[i], Vector: e}) }
+
+    // All done: PROCESSED + record ChunksList. Re-Enqueue same content is no-op.
+    return p.docStatus.MarkProcessed(ctx, docID, chunkIDs)
+}
```

The caller (`main.go`) gains exactly one read of `filePath` and one `Persist(ctx)` call after Insert; the next process launch reconstructs `DocStatusStore` and `Load` runs automatically. s03's store does NOT depend on s05's KV interface — when s05 lands, an exercise will show how to re-implement this store in 30 lines on top of it.

## Try It / 动手试一试

```bash
cd agents/s03-doc-status

# Default runs testdata/sample.txt — prints every transition across 3 rounds
go run .

# Output looks like:
# == round 1: fresh ingest ==
#   Enqueue          (none) -> pending (fresh)  chunks=0
#   MarkProcessing   pending -> processing  chunks=0
#   MarkProcessed    processing -> processed  chunks=3
#
# == round 2: re-ingest same content (dedup) ==
#   Enqueue returned fresh=false (no-op) — content already PROCESSED, no double-chunking.
#
# == round 3: simulate failure on a second doc ==
#   doc 5b8a... → FAILED (err="synthetic: provider rate-limit")
#   attempting FAILED -> PROCESSED rejected as expected: doc-status: ...: cannot transition failed -> processed
#
# == resume-on-failure scan ==
#   ListByStatus(FAILED) found 1 doc(s) needing retry:
#     - 5b8a...  err="synthetic: provider rate-limit"  updated=...

# Run the tests (5, all offline, no external deps)
go test -v ./...
# === RUN   TestStatusTransitionsValid          full happy path
# === RUN   TestStatusInvalidTransitionRejected PROCESSED -> PROCESSING must return ErrInvalidTransition
# === RUN   TestStatusPersistAcrossRestart      3 docs persisted, fresh store reconstructed, all round-trip
# === RUN   TestDuplicateInsertIsNoop           second Enqueue does not overwrite status / FilePath
# === RUN   TestFailedDocPreservesErrorMsg      MarkFailed retains ErrorMsg field
```

After running, `./lightrag-data/doc_status.json` contains the persisted state file; the next `go run .` will Load it. Delete the directory to start fresh.

## Upstream Source Reading / 上游源码阅读

The load-bearing 30 lines from `lightrag/kg/json_doc_status_impl.py` (full annotated excerpt at [`upstream-readings/s03-doc-status.py`](../../upstream-readings/s03-doc-status.py)):

```python
@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        workspace_dir = (
            os.path.join(working_dir, self.workspace) if self.workspace else working_dir
        )
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data: return
        for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
            if "chunks_list" not in doc_data:
                doc_data["chunks_list"] = []          # default empty list
            await _cooperative_yield(i)
        async with self._storage_lock:
            self._data.update(data)                   # in-memory update first
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()              # flush at end of batch

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                needs_reload = write_json(data_dict, self._file_name)  # atomic rename
                # ... (omitted: sanitize + reload branch — not in s03 scope)
                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])
```

**Reading notes**:

- **`@dataclass` + `@final` vs. Go struct.** Upstream uses dataclass + final to lock down fields. In Go we use a plain struct + unexported fields (lowercase first letter) to get the same "no external subclassing" property.
- **`workspace` / `namespace` prefix vs. s03's single store.** Upstream supports multiple doc-status stores per process (different workspaces); the file is `kv_store_doc_status.json` plus a prefix. s03 has one store, path is `./lightrag-data/doc_status.json` directly — equivalent simplified behavior.
- **`_cooperative_yield(i)` vs. `sync.RWMutex`.** Upstream yields the asyncio coroutine every N items so a long upsert doesn't starve other tasks. Go's sync.RWMutex is a blocking lock, but the critical section here is just a map assignment — nanoseconds, not a starvation risk. Worth the simplification at this scale; s05's KV chapter switches to per-key sync.Map for finer granularity.
- **`write_json` atomic rename vs. `os.Rename`.** Upstream's `write_json` writes to `.tmp` then `os.replace`s; s03 uses `os.WriteFile(tmp,...)` + `os.Rename(tmp,target)`. Both rely on POSIX guaranteeing rename is atomic.
- **Where is the dedup in `upsert`?** Note upstream's `upsert` does NOT dedup — `self._data.update(data)` is a dict update, which overwrites. The dedup lives one layer up in `lightrag/lightrag.py:apipeline_enqueue_documents`, which calls `filter_keys` first. s03 collapses these two steps inside `Enqueue`: the existence check and the insert happen under the same lock — one fewer round trip for the caller.
- **`get_docs_by_status` → `get_docs_by_statuses([status])`.** Upstream treats single-status as a special case of multi-status. s03 ships `ListByStatus(status)` directly; if s09 needs multi-status it'll be added then. YAGNI.

**Want more**: upstream's `JsonDocStatusStorage` also has `get_status_counts` / `get_docs_paginated` / `get_doc_by_file_path` / `get_docs_by_track_id` — dashboard conveniences that are list/filter ops on `_data`. s03 ships only `ListByStatus`; the rest are exercises. s05's KV chapter sets up the full `KVStore` interface to host them. s09 (entity extraction) is where `ListByStatus(FAILED) + ListByStatus(PROCESSING)` becomes the resume-on-failure scan that this layer was built for — re-read this chapter once you reach s09.

---

**Next chapter preview**: s04 replaces s01's "split on newlines" with a real 1200/100 sliding window in tokens (via `pkoukk/tiktoken-go`), porting upstream's `chunking_by_token_size` to Go. Chunk IDs stay `<docID>::chunk-<index>` — and that docID is the same MD5 content address from this chapter, so the two pieces snap together naturally.
