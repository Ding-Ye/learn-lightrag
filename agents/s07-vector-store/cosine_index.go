package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// CosineIndex is the s07 reference impl of VectorStore: an in-memory slice of
// VectorRecord scored by cosine similarity, with one JSON snapshot per
// namespace at `<dir>/vdb_<namespace>.json`.
//
// Storage is intentionally simple — the records live in `recs` (slice) and
// `idIdx` (id → recs index) so:
//   - Upsert can dedup by ID in O(1).
//   - Query can iterate `recs` in one pass.
//   - Delete can splice out a record without rescanning the whole slice.
//
// `dim` is set on the FIRST Upsert (or after Load) and validated on every
// subsequent insertion: a vector with the wrong length is rejected so silent
// dimension drift never corrupts the index.
//
// Locking: a single sync.RWMutex coarse-locks the index.  Reads (Query,
// snapshot for Persist) take RLock; writes (Upsert, Delete, Load) take Lock.
// LightRAG's upstream uses a per-namespace asyncio lock; we collapse to one
// RWMutex because Go's stdlib already lets multiple readers in.
type CosineIndex struct {
	dir       string
	namespace string

	mu    sync.RWMutex
	recs  []VectorRecord
	idIdx map[string]int // id → index into recs; rebuilt on every mutation
	dim   int            // 0 means "not yet seen any record"
}

// ErrEmptyNamespace is returned if the caller forgets to supply a namespace.
// Upstream silently defaults to "" which collides every namespace onto one
// file; we surface a typed error instead.
var ErrEmptyNamespace = errors.New("vector-store: namespace must be non-empty")

// ErrDimensionMismatch is returned on Upsert when a record's Vector length
// does not match the index dimension established by the first record (or by
// Load).  Same dim mismatch error shape as upstream's
// "vector dimensions don't match" runtime check.
type ErrDimensionMismatch struct {
	Expected int
	Got      int
	ID       string
}

func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("vector-store: dimension mismatch for id %q: expected %d, got %d",
		e.ID, e.Expected, e.Got)
}

// fileNamePrefix mirrors upstream's `vdb_<namespace>.json` naming.  Centralized
// so Persist and Load can never disagree on the filesystem layout.
const fileNamePrefix = "vdb_"

// NewCosineIndex constructs an index rooted at `dir` for the given namespace.
// If `<dir>/vdb_<namespace>.json` already exists, its contents are loaded
// synchronously; a missing file means we start fresh and is NOT an error.
//
// Callers MUST check the returned error before using the index — a successful
// (nil) return means the index is ready for concurrent use.
func NewCosineIndex(dir, namespace string) (*CosineIndex, error) {
	if namespace == "" {
		return nil, ErrEmptyNamespace
	}
	if dir == "" {
		dir = "./lightrag-data"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("vector-store: mkdir %s: %w", dir, err)
	}
	idx := &CosineIndex{
		dir:       dir,
		namespace: namespace,
		recs:      make([]VectorRecord, 0),
		idIdx:     make(map[string]int),
	}
	if err := idx.loadFromDisk(); err != nil {
		return nil, err
	}
	return idx, nil
}

// filePath returns the canonical on-disk path.  Use this everywhere — never
// build it from string concatenation.
func (s *CosineIndex) filePath() string {
	return filepath.Join(s.dir, fileNamePrefix+s.namespace+".json")
}

// Len reports the in-memory record count.  Useful for the CLI demo and tests
// that want a quick invariant check without iterating the index.
func (s *CosineIndex) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.recs)
}

// Dim reports the index dimension (0 if no record has been inserted yet).
// Set on first Upsert and validated on every subsequent one.
func (s *CosineIndex) Dim() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.dim
}

// rebuildIndex regenerates idIdx from recs.  Called after Delete (which
// shrinks recs) and after Load (which deserializes recs from disk).  Cheap:
// O(N) and only happens on full mutations, not per-Upsert.
func (s *CosineIndex) rebuildIndex() {
	s.idIdx = make(map[string]int, len(s.recs))
	for i, r := range s.recs {
		s.idIdx[r.ID] = i
	}
}

// copyMetadata does a one-level deep copy of the metadata bag so callers
// can't mutate the in-memory state by reaching into a returned VectorHit.
// The values themselves aren't deep-copied — they're typically scalars or
// short strings.
func copyMetadata(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// copyVector returns a defensive copy of the float32 slice.  Without this,
// a caller mutating `record.Vector` after Upsert would corrupt the index.
func copyVector(v []float32) []float32 {
	if v == nil {
		return nil
	}
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

// Upsert writes records into the index, deduplicating by ID.  Validates that
// every record's Vector length matches the index dim (set on first record).
// An empty input is a no-op.
//
// Semantic: if an ID already exists, the new Vector + Metadata REPLACE the
// old.  We don't merge metadata — upstream doesn't either.
func (s *CosineIndex) Upsert(ctx context.Context, records []VectorRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(records) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// First record establishes the dimension if not yet set.
	if s.dim == 0 {
		s.dim = len(records[0].Vector)
		if s.dim == 0 {
			return fmt.Errorf("vector-store: first record %q has zero-length vector", records[0].ID)
		}
	}
	for _, rec := range records {
		if len(rec.Vector) != s.dim {
			return &ErrDimensionMismatch{Expected: s.dim, Got: len(rec.Vector), ID: rec.ID}
		}
		if rec.ID == "" {
			return errors.New("vector-store: record ID must be non-empty")
		}
		newRec := VectorRecord{
			ID:       rec.ID,
			Vector:   copyVector(rec.Vector),
			Metadata: copyMetadata(rec.Metadata),
		}
		if i, ok := s.idIdx[rec.ID]; ok {
			s.recs[i] = newRec
			continue
		}
		s.idIdx[rec.ID] = len(s.recs)
		s.recs = append(s.recs, newRec)
	}
	return nil
}

// Query computes cosine similarity between `query` and every record's vector,
// keeps those with score >= threshold, sorts descending, and returns up to
// topK hits.  An empty index yields []VectorHit{} (never nil) and no error.
//
// Cosine math: cos(a, b) = (a · b) / (|a| * |b|).  We compute `|query|` once
// outside the loop; per-record `|rec|` is computed inline.  If either vector
// has zero norm we score 0 (skipping div-by-zero).
//
// Cost: O(N * dim) per query.  Fine for ≤ 50K vectors which is the typical
// LightRAG load; bigger workloads should swap in HNSW or IVF — discussed in
// the docs but deliberately not implemented here.
func (s *CosineIndex) Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		return []VectorHit{}, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.recs) == 0 {
		return []VectorHit{}, nil
	}
	if s.dim != 0 && len(query) != s.dim {
		return nil, &ErrDimensionMismatch{Expected: s.dim, Got: len(query), ID: "<query>"}
	}

	qNorm := vectorNorm(query)
	hits := make([]VectorHit, 0, len(s.recs))
	for i := range s.recs {
		rec := &s.recs[i]
		score := cosineSimilarity(query, rec.Vector, qNorm)
		if score < threshold {
			continue
		}
		hits = append(hits, VectorHit{
			ID:       rec.ID,
			Score:    score,
			Metadata: copyMetadata(rec.Metadata),
		})
	}

	sort.Slice(hits, func(i, j int) bool {
		// Descending by score; tie-break by ID for deterministic ordering so
		// tests and the CLI demo print stable output.
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].ID < hits[j].ID
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}
	return hits, nil
}

// vectorNorm returns sqrt(sum(v[i]^2)).  Used by Query to compute |query|
// once and amortize the cost across N records.
func vectorNorm(v []float32) float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	return float32(math.Sqrt(sumSq))
}

// cosineSimilarity computes (a · b) / (|a| * |b|).  `aNorm` is supplied so
// the caller (Query) doesn't recompute it per-record.  Returns 0 if either
// vector has zero norm.
func cosineSimilarity(a, b []float32, aNorm float32) float32 {
	var dot float64
	var bSumSq float64
	for i := range a {
		af := float64(a[i])
		bf := float64(b[i])
		dot += af * bf
		bSumSq += bf * bf
	}
	bNorm := float32(math.Sqrt(bSumSq))
	if aNorm == 0 || bNorm == 0 {
		return 0
	}
	return float32(dot / (float64(aNorm) * float64(bNorm)))
}

// Delete removes the given IDs from the index.  Missing IDs are silently
// ignored — upstream does the same so we don't surface a "not found" error.
// After deletion, idIdx is rebuilt from the surviving recs.
func (s *CosineIndex) Delete(ctx context.Context, ids []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	toDrop := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		toDrop[id] = struct{}{}
	}
	kept := s.recs[:0]
	for _, r := range s.recs {
		if _, drop := toDrop[r.ID]; drop {
			continue
		}
		kept = append(kept, r)
	}
	// kept reuses the underlying array; clear stale tail entries so GC can
	// collect the dropped Vector/Metadata allocations promptly.
	for i := len(kept); i < len(s.recs); i++ {
		s.recs[i] = VectorRecord{}
	}
	s.recs = kept
	s.rebuildIndex()
	return nil
}

// vdbFile is the on-disk JSON envelope.  Mirrors the shape upstream's
// NanoVectorDB writes (an "embedding_dim" header + a list of records) but
// uses plain float32 arrays instead of the upstream's float16+zlib+base64
// compression — appendix exercise.
type vdbFile struct {
	EmbeddingDim int                   `json:"embedding_dim"`
	Namespace    string                `json:"namespace"`
	Data         []vdbRecordOnDisk     `json:"data"`
}

type vdbRecordOnDisk struct {
	ID       string         `json:"__id__"`
	Vector   []float32      `json:"vector"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// loadFromDisk reads `<dir>/vdb_<namespace>.json` and replaces the in-memory
// state.  A missing file is benign — empty index.  A malformed file is
// fatal: better to fail at construction time than silently start fresh.
func (s *CosineIndex) loadFromDisk() error {
	buf, err := os.ReadFile(s.filePath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("vector-store: read %s: %w", s.filePath(), err)
	}
	if len(buf) == 0 {
		return nil
	}
	var v vdbFile
	if err := json.Unmarshal(buf, &v); err != nil {
		return fmt.Errorf("vector-store: unmarshal %s: %w", s.filePath(), err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.dim = v.EmbeddingDim
	s.recs = make([]VectorRecord, 0, len(v.Data))
	for _, d := range v.Data {
		s.recs = append(s.recs, VectorRecord{
			ID:       d.ID,
			Vector:   d.Vector,
			Metadata: d.Metadata,
		})
	}
	s.rebuildIndex()
	return nil
}

// Persist atomically writes the index to disk via the same write-tmp + rename
// dance that s05's KV store uses: marshal to `<file>.tmp`, then rename to the
// canonical path.  POSIX guarantees rename atomicity on the same filesystem,
// so a reader either sees the full old file or the full new file — never an
// intermediate.
func (s *CosineIndex) Persist(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.RLock()
	envelope := vdbFile{
		EmbeddingDim: s.dim,
		Namespace:    s.namespace,
		Data:         make([]vdbRecordOnDisk, 0, len(s.recs)),
	}
	for _, r := range s.recs {
		envelope.Data = append(envelope.Data, vdbRecordOnDisk{
			ID:       r.ID,
			Vector:   copyVector(r.Vector),
			Metadata: copyMetadata(r.Metadata),
		})
	}
	s.mu.RUnlock()

	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("vector-store: mkdir %s: %w", s.dir, err)
	}
	buf, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		return fmt.Errorf("vector-store: marshal: %w", err)
	}
	canonical := s.filePath()
	tmp := canonical + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return fmt.Errorf("vector-store: write tmp %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, canonical); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("vector-store: rename %s -> %s: %w", tmp, canonical, err)
	}
	return nil
}
