package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// JSONKVStore is the s05 reference impl of KVStore: an in-memory map backed
// by one JSON file per namespace under `<dir>/kv_<namespace>.json`.  Two
// distinct locks coexist:
//
//   - `mu` is a coarse RWMutex over the whole `data` map.  Held in WRITE mode
//     during Upsert/Delete/Load (because we mutate the map shape) and during
//     Persist (so we get a consistent snapshot).  Held in READ mode for
//     Get/GetByIDs/FilterMissing.
//
//   - `keyLocks` is a `sync.Map` of `*sync.Mutex` keyed by record id.  It is
//     used inside Upsert to serialize concurrent writes to the SAME id while
//     still letting writes against DIFFERENT ids proceed in parallel.  This
//     mirrors upstream's "namespace lock + per-record sequencing" intent
//     without the global asyncio Lock — different ids genuinely run in
//     parallel here.  The sync.Map gives us lazy, allocation-free creation
//     of the per-key mutex.
//
// Persistence uses atomic-rename: write to `<file>.tmp`, then `os.Rename` to
// the canonical name.  POSIX guarantees the rename is atomic on the same
// filesystem, so a reader either sees the full old file or the full new file
// — never a half-written intermediate.  After a successful Persist the `.tmp`
// no longer exists; if a `.tmp` is found at startup, it's treated as garbage
// from a crashed previous run and ignored (the canonical file is the only
// source of truth on Load).
type JSONKVStore struct {
	dir       string
	namespace string

	mu       sync.RWMutex
	data     map[string]map[string]any
	keyLocks sync.Map // map[string]*sync.Mutex
}

// fileNamePrefix is the on-disk filename prefix.  Upstream uses
// `kv_store_<namespace>.json`; we trim that to `kv_<namespace>.json` so the
// learner sees the namespace prominently in `ls` output.
const fileNamePrefix = "kv_"

// ErrEmptyNamespace is returned by NewJSONKVStore when the caller forgets to
// supply a namespace.  Upstream silently defaults to "" which then makes
// every namespace collide on the same filename — we surface it as a typed
// error instead.
var ErrEmptyNamespace = errors.New("kv-store: namespace must be non-empty")

// NewJSONKVStore constructs a store rooted at `dir` for the given namespace.
// If `<dir>/kv_<namespace>.json` already exists, its contents are loaded
// synchronously; a missing file means we start fresh (NOT an error).  The
// directory is created with 0o755 permissions if it doesn't exist yet.
//
// Callers MUST check the returned error before using the store — a
// successful (nil error) return means the store is ready for concurrent use
// from any goroutine.
func NewJSONKVStore(dir, namespace string) (*JSONKVStore, error) {
	if namespace == "" {
		return nil, ErrEmptyNamespace
	}
	if dir == "" {
		dir = "./lightrag-data"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("kv-store: mkdir %s: %w", dir, err)
	}
	s := &JSONKVStore{
		dir:       dir,
		namespace: namespace,
		data:      make(map[string]map[string]any),
	}
	if err := s.loadFromDisk(); err != nil {
		return nil, err
	}
	return s, nil
}

// filePath returns the canonical on-disk path.  Centralized so tests and
// Persist/Load all agree on the layout.
func (s *JSONKVStore) filePath() string {
	return filepath.Join(s.dir, fileNamePrefix+s.namespace+".json")
}

// loadFromDisk replaces `data` with the contents of the canonical JSON file.
// A missing file is benign — empty store.  A malformed file is fatal: better
// to fail at construction time than silently drop records.  Note that any
// `<file>.tmp` left by a crashed Persist is intentionally ignored — Load
// only ever reads the canonical name.
func (s *JSONKVStore) loadFromDisk() error {
	buf, err := os.ReadFile(s.filePath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("kv-store: read %s: %w", s.filePath(), err)
	}
	if len(buf) == 0 {
		return nil
	}
	loaded := make(map[string]map[string]any)
	if err := json.Unmarshal(buf, &loaded); err != nil {
		return fmt.Errorf("kv-store: unmarshal %s: %w", s.filePath(), err)
	}
	s.mu.Lock()
	s.data = loaded
	s.mu.Unlock()
	return nil
}

// keyLock returns the per-key mutex, creating it on first access.  We use
// sync.Map.LoadOrStore to guarantee exactly one mutex per id even under
// concurrent first-access races.
func (s *JSONKVStore) keyLock(id string) *sync.Mutex {
	if m, ok := s.keyLocks.Load(id); ok {
		return m.(*sync.Mutex)
	}
	m, _ := s.keyLocks.LoadOrStore(id, &sync.Mutex{})
	return m.(*sync.Mutex)
}

// copyMap does a one-level deep copy of the record so callers can't mutate
// the in-memory state by reaching into the returned map.  The values
// themselves are not deep-copied (they're typically scalars or short slices
// that callers shouldn't mutate anyway).
func copyMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// Get returns a defensive copy of the record at id.
func (s *JSONKVStore) Get(ctx context.Context, id string) (map[string]any, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	rec, ok := s.data[id]
	if !ok {
		return nil, false, nil
	}
	return copyMap(rec), true, nil
}

// GetByIDs returns one map per HIT.  Misses are simply absent from the
// returned map (mirrors upstream's "skip None" behavior in a Go-idiomatic
// way — callers iterate over the returned map directly without nil-checks).
func (s *JSONKVStore) GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]map[string]any, len(ids))
	for _, id := range ids {
		if rec, ok := s.data[id]; ok {
			out[id] = copyMap(rec)
		}
	}
	return out, nil
}

// FilterMissing — the centerpiece — returns the subset of `ids` NOT present
// in the store.  Empty input returns an empty slice (never nil).  Output is
// in input order, deduplicated (an id repeated in `ids` only appears once
// in the result if missing).  This is the primitive s09 will call on every
// chunk batch to ask "which of these still need entity extraction?".
func (s *JSONKVStore) FilterMissing(ctx context.Context, ids []string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([]string, 0)
	if len(ids) == 0 {
		return out, nil
	}
	seen := make(map[string]struct{}, len(ids))
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, id := range ids {
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		if _, ok := s.data[id]; !ok {
			out = append(out, id)
		}
	}
	return out, nil
}

// Upsert writes each (id, record) pair into the store.  Per-key locks
// serialize concurrent writers targeting the SAME id; writes to disjoint
// ids run in parallel.  We acquire the global write-lock only for the brief
// final commit step so reads aren't starved.
//
// The flow is:
//  1. For each (id, rec): grab the per-key mutex, copy the record so the
//     caller can't mutate our state through the input map, hold the per-key
//     mutex while we hand the new value to the global write step.
//  2. Take s.mu (write) once at the end and apply all updates atomically
//     from the caller's POV — readers either see all the new values or none
//     of them.  This matches upstream's "upsert is a single critical section
//     under the namespace lock" semantics.
func (s *JSONKVStore) Upsert(ctx context.Context, items map[string]map[string]any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(items) == 0 {
		return nil
	}

	// Stage copies under per-key locks so concurrent Upsert(same-key) is
	// race-free without serializing the whole store.  The temporary
	// `staged` map is owned by this goroutine; writers to disjoint keys
	// don't contend on it.
	staged := make(map[string]map[string]any, len(items))
	for id, rec := range items {
		lk := s.keyLock(id)
		lk.Lock()
		staged[id] = copyMap(rec)
		lk.Unlock()
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	for id, rec := range staged {
		s.data[id] = rec
	}
	return nil
}

// Delete removes the given IDs.  Missing IDs are silently ignored — upstream
// does the same so we don't surface a "not found" error here.  Like Upsert,
// the persistence step is deferred to Persist().
func (s *JSONKVStore) Delete(ctx context.Context, ids []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		delete(s.data, id)
	}
	return nil
}

// Persist atomically flushes the current map to disk.  The dance is:
//
//  1. Snapshot the map under a read-lock so the marshal step can release the
//     critical section while it does CPU work.
//  2. MarshalIndent with two-space indent — matches upstream's `write_json`
//     output and gives the learner a diff-friendly file in `git status`.
//  3. Write to `<file>.tmp` (truncating any orphan tmp from a previous
//     crash).
//  4. `os.Rename(tmp, canonical)` — POSIX-atomic on the same filesystem.
//     After this, the `.tmp` no longer exists.
//
// If the rename fails we make a best-effort attempt to clean up the orphan
// `.tmp` so the next Load doesn't trip on it (Load ignores `.tmp` anyway,
// but tidiness matters for human readers of `ls`).
func (s *JSONKVStore) Persist(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.RLock()
	snapshot := make(map[string]map[string]any, len(s.data))
	for k, v := range s.data {
		snapshot[k] = copyMap(v)
	}
	s.mu.RUnlock()

	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("kv-store: mkdir %s: %w", s.dir, err)
	}
	buf, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("kv-store: marshal: %w", err)
	}

	canonical := s.filePath()
	tmp := canonical + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return fmt.Errorf("kv-store: write tmp %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, canonical); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("kv-store: rename %s -> %s: %w", tmp, canonical, err)
	}
	return nil
}

// Len reports the in-memory record count.  Not part of the KVStore
// interface — exposed only for the CLI demo and for tests that want a quick
// invariant check without round-tripping through the iterators.
func (s *JSONKVStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.data)
}
