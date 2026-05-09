package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

// cache.go provides a minimal in-memory KV store implementing the local
// KVStore interface.  Mirrors the subset of upstream's BaseKVStorage
// (lightrag/base.py:308) that the extraction cache actually uses:
//
//   - Get(ctx, id)        -> (value, ok, err)
//   - Upsert(ctx, items)  -> error
//   - FilterMissing(ctx, ids) -> ([]missingIDs, error)
//
// The cache is keyed by sha256(chunk.Content || promptVersion).  Hits
// avoid the LLM call entirely — the cached raw text is re-parsed.  This
// is what lets us iteratively change the parser/gleaning logic without
// burning quota on already-extracted chunks.
//
// We re-declare the KVStore interface locally — this session is isolated,
// no cross-session imports.  The shape is identical to the catalog in
// .learn/plan.md (only Get + Upsert + FilterMissing here; the full KVStore
// in s05 also has GetByIDs / Delete / Persist).

// KVStore is the minimal interface s09 needs from a key-value store.
// Same shape as upstream BaseKVStorage but trimmed to the methods the
// extraction cache uses.
type KVStore interface {
	// Get returns (value, true, nil) on hit; (zero, false, nil) on miss.
	// Errors only on context cancellation or backing-store failure.
	Get(ctx context.Context, id string) (map[string]any, bool, error)

	// Upsert inserts/overwrites multiple records atomically (per-key).
	Upsert(ctx context.Context, items map[string]map[string]any) error

	// FilterMissing returns the subset of ids NOT in the store.  Used by
	// pipelines to decide "which chunks need extracting?".
	FilterMissing(ctx context.Context, ids []string) ([]string, error)
}

// MemoryKVStore is a goroutine-safe in-memory KV store.  Good enough for
// the s09 demo and tests; in a real pipeline this would be backed by the
// persistent JSON store from s05.
type MemoryKVStore struct {
	mu    sync.RWMutex
	store map[string]map[string]any
}

// NewMemoryKVStore returns an empty store ready for use.
func NewMemoryKVStore() *MemoryKVStore {
	return &MemoryKVStore{store: make(map[string]map[string]any)}
}

// Get implements KVStore.  Returns a shallow copy of the stored map so
// callers can't mutate the cache via the returned value.
func (m *MemoryKVStore) Get(ctx context.Context, id string) (map[string]any, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.store[id]
	if !ok {
		return nil, false, nil
	}
	out := make(map[string]any, len(v))
	for k, val := range v {
		out[k] = val
	}
	return out, true, nil
}

// Upsert implements KVStore.  Overwrites existing keys; copies maps in
// so callers can't mutate stored values via their input.
func (m *MemoryKVStore) Upsert(ctx context.Context, items map[string]map[string]any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for id, v := range items {
		stored := make(map[string]any, len(v))
		for k, val := range v {
			stored[k] = val
		}
		m.store[id] = stored
	}
	return nil
}

// FilterMissing implements KVStore.
func (m *MemoryKVStore) FilterMissing(ctx context.Context, ids []string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	missing := make([]string, 0, len(ids))
	for _, id := range ids {
		if _, ok := m.store[id]; !ok {
			missing = append(missing, id)
		}
	}
	return missing, nil
}

// cacheKey returns sha256(chunkContent || promptVersion) hex-encoded.
// This is the canonical key used by Extractor when consulting Cache.
//
// The promptVersion suffix means: if you change either the system prompt
// or the continuation prompt, bump promptVersion in extraction_prompt.go
// and old cache entries automatically become misses (no manual flush).
func cacheKey(chunkContent string) string {
	h := sha256.New()
	h.Write([]byte(chunkContent))
	h.Write([]byte("|"))
	h.Write([]byte(promptVersion))
	return "extract:" + hex.EncodeToString(h.Sum(nil))
}
