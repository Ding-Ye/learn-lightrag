package main

// VectorStore mirrors upstream BaseVectorStorage (lightrag/base.py:189).
// In s01 we ship a minimal in-memory cosine-similarity index — O(N·d) per
// query, no thresholding, no persistence. s07 turns this into the real
// nano_vector_db_impl-equivalent.

import (
	"context"
	"errors"
	"math"
	"sort"
	"sync"
)

// VectorRecord is the canonical (id, vector, metadata) tuple persisted in
// the vector store.
type VectorRecord struct {
	ID       string
	Vector   []float32
	Metadata map[string]any
}

// VectorHit is a Query result row.
type VectorHit struct {
	ID       string
	Score    float32 // cosine similarity in [-1, 1]
	Metadata map[string]any
}

// VectorStore is the s01..s11 contract. Threshold is in s01 just for shape;
// the in-memory impl honors it (filters out scores < threshold).
type VectorStore interface {
	Upsert(ctx context.Context, records []VectorRecord) error
	Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error)
	Delete(ctx context.Context, ids []string) error
	Persist(ctx context.Context) error
}

// InMemoryVectorStore is a slice-backed cosine index. Safe for concurrent
// readers/writers via a single sync.RWMutex; that's good enough for s01's
// single-goroutine pipeline.
type InMemoryVectorStore struct {
	mu      sync.RWMutex
	records []VectorRecord
}

// NewInMemoryVectorStore returns an empty store.
func NewInMemoryVectorStore() *InMemoryVectorStore {
	return &InMemoryVectorStore{}
}

// Upsert appends or replaces records by ID.
func (s *InMemoryVectorStore) Upsert(ctx context.Context, records []VectorRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range records {
		replaced := false
		for i := range s.records {
			if s.records[i].ID == r.ID {
				s.records[i] = r
				replaced = true
				break
			}
		}
		if !replaced {
			s.records = append(s.records, r)
		}
	}
	return nil
}

// Query runs an O(N·d) cosine scan and returns the top-k hits with
// score >= threshold, sorted descending by score.
func (s *InMemoryVectorStore) Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error) {
	if len(query) == 0 {
		return nil, errors.New("vectorstore: empty query vector")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	hits := make([]VectorHit, 0, len(s.records))
	for _, r := range s.records {
		score := cosineSimilarity(query, r.Vector)
		if score < threshold {
			continue
		}
		hits = append(hits, VectorHit{ID: r.ID, Score: score, Metadata: r.Metadata})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if topK > 0 && len(hits) > topK {
		hits = hits[:topK]
	}
	return hits, nil
}

// Delete removes records by ID.
func (s *InMemoryVectorStore) Delete(ctx context.Context, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	keep := make([]VectorRecord, 0, len(s.records))
	rm := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		rm[id] = struct{}{}
	}
	for _, r := range s.records {
		if _, drop := rm[r.ID]; !drop {
			keep = append(keep, r)
		}
	}
	s.records = keep
	return nil
}

// Persist is a no-op in s01; s07 ships the real JSON snapshot.
func (s *InMemoryVectorStore) Persist(ctx context.Context) error { return nil }

// cosineSimilarity assumes both vectors have the same length; if they don't
// we just truncate to the shorter side. Returns 0 on zero-norm input.
func cosineSimilarity(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		na += ai * ai
		nb += bi * bi
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
