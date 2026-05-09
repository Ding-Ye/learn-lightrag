package main

import "context"

// Namespace is a string alias used as the on-disk filename prefix.  Upstream
// LightRAG groups KV records by `namespace` (e.g. `text_chunks`, `full_docs`,
// `llm_response_cache`) — each namespace maps to its own JSON file under the
// working dir.  s05 mirrors that one-file-per-namespace shape so the directory
// listing matches what a learner sees in upstream's `./rag_storage/`.
type Namespace = string

// KVStore is the abstract interface every KV backend must satisfy.  It mirrors
// upstream `BaseKVStorage` (lightrag/base.py:308) condensed to the subset s09
// will actually call: get / get-by-ids / upsert / filter-missing / delete /
// persist.  The async cooperative-yield from upstream is gone — Go callers
// pass `context.Context` for cancellation instead.
//
// FilterMissing is the centerpiece.  Upstream calls it `filter_keys` and uses
// it as the dedup primitive: "of these N candidate IDs, which are not yet in
// the store?"  s09's extraction pipeline will call this to skip chunks whose
// entities have already been pulled out, avoiding redundant LLM calls.
type KVStore interface {
	// Get fetches a single record.  Returns (record, true, nil) on hit,
	// (nil, false, nil) on miss; any I/O error surfaces as the third value.
	Get(ctx context.Context, id string) (map[string]any, bool, error)

	// GetByIDs fetches a batch.  The returned map only contains hits — IDs
	// not in the store are simply omitted (mirrors upstream's "skip None"
	// semantic).
	GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error)

	// Upsert writes records under their IDs.  Existing IDs are replaced.
	// Persistence is deferred to Persist() to match upstream's
	// `index_done_callback` batching behavior.
	Upsert(ctx context.Context, items map[string]map[string]any) error

	// FilterMissing returns the subset of `ids` NOT present in the store.
	// Empty input returns an empty slice, never nil.  Order is unspecified.
	FilterMissing(ctx context.Context, ids []string) ([]string, error)

	// Delete removes the given IDs.  Missing IDs are silently ignored.
	Delete(ctx context.Context, ids []string) error

	// Persist flushes the in-memory map to durable storage.  After a clean
	// return, the on-disk file reflects everything Upsert/Delete have done
	// since the last Persist.
	Persist(ctx context.Context) error
}
