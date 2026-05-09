package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// defaultStorePath is the on-disk JSON file that mirrors upstream's
// `kv_store_doc_status.json` under `./rag_storage/`.  We default to
// `./lightrag-data/doc_status.json` in keeping with the curriculum's
// "Storage on disk mirrors upstream `./rag_storage/` layout under
// `./lightrag-data/`" rule.
const defaultStorePath = "./lightrag-data/doc_status.json"

// DocStatusStore is the in-memory + JSON-persisted state machine that fronts
// upstream's `JsonDocStatusStorage` (lightrag/kg/json_doc_status_impl.py).
//
// Differences from upstream worth noting:
//   - We don't carry a namespace/workspace prefix (s03 only has one store).
//   - We persist on Persist() rather than every upsert — upstream calls
//     index_done_callback() at the end of every batch.  s03 makes Persist()
//     an explicit caller responsibility so test setups don't have to mock fs.
//   - Lock granularity: one RWMutex over the whole map (vs upstream's
//     per-namespace shared lock).  Adequate for a teaching session; s05's
//     full KV impl will tighten this with per-key sync.Map.
type DocStatusStore struct {
	path string

	mu sync.RWMutex
	// docs is the source of truth in-memory.  Reads (Get / ListByStatus)
	// take RLock; writes (Enqueue, Mark*) take the write lock.
	docs map[string]*DocProcessingStatus

	// Clock is overridable for tests so timestamps are deterministic.
	// Production code lets it default to time.Now (set in NewDocStatusStore).
	Clock func() time.Time
}

// NewDocStatusStore constructs an empty store and synchronously Loads from
// disk if the JSON file at `path` already exists.  Pass empty path to use
// the default.  An empty string return value is never valid — caller MUST
// check the returned error.
func NewDocStatusStore(path string) (*DocStatusStore, error) {
	if path == "" {
		path = defaultStorePath
	}
	s := &DocStatusStore{
		path:  path,
		docs:  make(map[string]*DocProcessingStatus),
		Clock: time.Now,
	}
	if err := s.Load(context.Background()); err != nil {
		return nil, fmt.Errorf("doc-status: load: %w", err)
	}
	return s, nil
}

// Enqueue is the entry point for ingestion.  It mirrors the upstream
// "filter then upsert" two-step:
//   - If a doc with this contentDocID already exists, we DO NOT overwrite —
//     this is the resume-on-failure idempotency property.  The (existing,
//     false, nil) tuple tells the caller "this doc is already known".
//   - Otherwise we insert a new PENDING record with timestamps and return
//     (new, true, nil) so the caller proceeds to chunk + embed.
//
// `content` is the raw doc text; `filePath` is metadata for debugging.
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
		ChunksList:     []string{}, // never nil — upstream's chunks_list default factory
	}
	s.docs[docID] = rec
	return rec, true, nil
}

// MarkProcessing transitions PENDING -> PROCESSING.  Called when a worker
// picks up an enqueued doc and starts chunking/embedding.
func (s *DocStatusStore) MarkProcessing(ctx context.Context, docID string) error {
	return s.transition(ctx, docID, DocStatusProcessing, nil)
}

// MarkProcessed transitions PROCESSING -> PROCESSED and records the chunk
// IDs that were produced.  This is the only success terminal state.
func (s *DocStatusStore) MarkProcessed(ctx context.Context, docID string, chunksList []string) error {
	return s.transition(ctx, docID, DocStatusProcessed, func(rec *DocProcessingStatus) {
		// Defensive copy — caller may mutate the slice afterwards.
		rec.ChunksList = append([]string(nil), chunksList...)
		rec.ChunksCount = len(chunksList)
	})
}

// MarkFailed transitions any non-PROCESSED state -> FAILED, preserving the
// caller-provided error message for resume-on-failure log output.
func (s *DocStatusStore) MarkFailed(ctx context.Context, docID, errMsg string) error {
	return s.transition(ctx, docID, DocStatusFailed, func(rec *DocProcessingStatus) {
		rec.ErrorMsg = errMsg
	})
}

// transition is the shared write path.  Validates the edge via
// IsValidTransition, applies the optional mutation, bumps UpdatedAt.
func (s *DocStatusStore) transition(ctx context.Context, docID string, to DocStatus, mutate func(*DocProcessingStatus)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
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
	if mutate != nil {
		mutate(rec)
	}
	return nil
}

// Get returns a snapshot of the record for docID.  The returned pointer
// references a defensive copy so callers can't mutate the in-memory state.
func (s *DocStatusStore) Get(ctx context.Context, docID string) (*DocProcessingStatus, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	rec, ok := s.docs[docID]
	if !ok {
		return nil, false, nil
	}
	cp := *rec
	cp.ChunksList = append([]string(nil), rec.ChunksList...)
	return &cp, true, nil
}

// ListByStatus returns all records currently in the given state.  This is
// the resume-on-failure scan: at startup, callers do
// `ListByStatus(DocStatusFailed)` (and PROCESSING, in case of mid-run kill)
// to find work that needs to be retried.
func (s *DocStatusStore) ListByStatus(ctx context.Context, status DocStatus) ([]*DocProcessingStatus, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*DocProcessingStatus, 0)
	for _, rec := range s.docs {
		if rec.Status == status {
			cp := *rec
			cp.ChunksList = append([]string(nil), rec.ChunksList...)
			out = append(out, &cp)
		}
	}
	return out, nil
}

// Persist writes the in-memory map to disk via atomic-rename: write to a
// `.tmp` sibling, fsync, then rename(2) over the target.  This guarantees
// readers either see the old file or the new file — never a half-written
// file.  Mirrors upstream's `write_json` in lightrag/utils.py which uses
// the same pattern.
func (s *DocStatusStore) Persist(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.RLock()
	// Snapshot under lock, encode outside lock to keep critical section short.
	snapshot := make(map[string]*DocProcessingStatus, len(s.docs))
	for k, v := range s.docs {
		cp := *v
		cp.ChunksList = append([]string(nil), v.ChunksList...)
		snapshot[k] = &cp
	}
	s.mu.RUnlock()

	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("doc-status: mkdir: %w", err)
	}
	buf, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("doc-status: marshal: %w", err)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return fmt.Errorf("doc-status: write tmp: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		// Best-effort cleanup of the orphan tmp file.
		_ = os.Remove(tmp)
		return fmt.Errorf("doc-status: rename: %w", err)
	}
	return nil
}

// Load replaces the in-memory state with the contents of the JSON file at
// s.path.  A missing file is NOT an error — it just means we're starting
// fresh.  Any decode error is surfaced so a corrupt file fails fast at
// startup rather than silently dropping records.
func (s *DocStatusStore) Load(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	buf, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("doc-status: read: %w", err)
	}
	loaded := make(map[string]*DocProcessingStatus)
	if err := json.Unmarshal(buf, &loaded); err != nil {
		return fmt.Errorf("doc-status: unmarshal: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs = loaded
	for _, rec := range s.docs {
		if rec.ChunksList == nil {
			rec.ChunksList = []string{}
		}
	}
	return nil
}

// summarize is the upstream `content_summary` heuristic — first 100 chars.
// We don't try to break on word boundaries (upstream doesn't either).
func summarize(content string) string {
	const limit = 100
	if len(content) <= limit {
		return content
	}
	return content[:limit]
}
