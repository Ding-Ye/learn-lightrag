package main

import (
	"context"
	"math"
	"testing"
)

// TestVectorStoreInterfaceContract is a compile-time guard that *CosineIndex
// satisfies the VectorStore interface.  No runtime work — if this panics we
// have bigger problems.
func TestVectorStoreInterfaceContract(t *testing.T) {
	var _ VectorStore = (*CosineIndex)(nil)
}

// makeIndex creates a fresh CosineIndex rooted under t.TempDir().  Centralized
// so test bodies stay focused on assertions, not directory plumbing.
func makeIndex(t *testing.T, ns Namespace) *CosineIndex {
	t.Helper()
	idx, err := NewCosineIndex(t.TempDir(), ns)
	if err != nil {
		t.Fatalf("NewCosineIndex: %v", err)
	}
	return idx
}

// unit returns the L2-normalized version of v.  Tests use unit vectors so
// cosine = dot product, making the expected scores easy to reason about.
func unit(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	if sumSq == 0 {
		return v
	}
	n := float32(math.Sqrt(sumSq))
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i] / n
	}
	return out
}

// TestCosineRanksMostSimilarFirst — insert 3 records, query with the same
// vector as one of them, assert that record is rank-1 with score ~ 1.0.
func TestCosineRanksMostSimilarFirst(t *testing.T) {
	ctx := context.Background()
	idx := makeIndex(t, NamespaceChunks)

	a := unit([]float32{1, 0, 0, 0})
	b := unit([]float32{0, 1, 0, 0})
	c := unit([]float32{0.7, 0.7, 0, 0})

	if err := idx.Upsert(ctx, []VectorRecord{
		{ID: "a", Vector: a},
		{ID: "b", Vector: b},
		{ID: "c", Vector: c},
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	hits, err := idx.Query(ctx, a, 3, 0.0)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("want 3 hits, got %d", len(hits))
	}
	if hits[0].ID != "a" {
		t.Errorf("rank-1 want a, got %s (full hits=%v)", hits[0].ID, hits)
	}
	if math.Abs(float64(hits[0].Score-1.0)) > 1e-5 {
		t.Errorf("rank-1 score want ~1.0, got %.6f", hits[0].Score)
	}
}

// TestCosineThresholdFiltersLowScores — a high threshold (0.99) should keep
// only near-identical matches and drop "merely similar" ones.
func TestCosineThresholdFiltersLowScores(t *testing.T) {
	ctx := context.Background()
	idx := makeIndex(t, NamespaceEntities)

	near := unit([]float32{1, 0.001, 0, 0})           // ~1.0 with [1,0,0,0]
	mid := unit([]float32{0.7, 0.7, 0, 0})            // ~0.707
	far := unit([]float32{0, 1, 0, 0})                // 0.0

	if err := idx.Upsert(ctx, []VectorRecord{
		{ID: "near", Vector: near},
		{ID: "mid", Vector: mid},
		{ID: "far", Vector: far},
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	hits, err := idx.Query(ctx, unit([]float32{1, 0, 0, 0}), 10, 0.99)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "near" {
		t.Errorf("threshold=0.99 should leave only 'near'; got %v", hits)
	}
}

// TestCosineRespectsTopK — insert 10 records, ask for top-3, assert exactly 3
// returned even though all 10 score above threshold.
func TestCosineRespectsTopK(t *testing.T) {
	ctx := context.Background()
	idx := makeIndex(t, NamespaceRelations)

	recs := make([]VectorRecord, 0, 10)
	for i := 0; i < 10; i++ {
		v := unit([]float32{float32(i + 1), 0.1, 0.1, 0.1})
		recs = append(recs, VectorRecord{ID: idForI(i), Vector: v})
	}
	if err := idx.Upsert(ctx, recs); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	hits, err := idx.Query(ctx, unit([]float32{1, 0, 0, 0}), 3, 0.0)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 3 {
		t.Errorf("topK=3 should yield 3 hits; got %d", len(hits))
	}
}

func idForI(i int) string {
	return string(rune('a'+i)) + "-rec"
}

// TestCosinePersistAcrossRestart — Upsert 5, Persist, construct a NEW
// CosineIndex with the SAME (dir, namespace), Query and assert the same
// hits come back.
func TestCosinePersistAcrossRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	// Phase 1 — fill and persist.
	idx1, err := NewCosineIndex(dir, NamespaceChunks)
	if err != nil {
		t.Fatalf("NewCosineIndex (1): %v", err)
	}
	recs := []VectorRecord{
		{ID: "p1", Vector: unit([]float32{1, 0, 0, 0}), Metadata: map[string]any{"src": "doc1"}},
		{ID: "p2", Vector: unit([]float32{0, 1, 0, 0})},
		{ID: "p3", Vector: unit([]float32{0, 0, 1, 0})},
		{ID: "p4", Vector: unit([]float32{0, 0, 0, 1})},
		{ID: "p5", Vector: unit([]float32{0.5, 0.5, 0.5, 0.5})},
	}
	if err := idx1.Upsert(ctx, recs); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := idx1.Persist(ctx); err != nil {
		t.Fatalf("Persist: %v", err)
	}

	hits1, err := idx1.Query(ctx, unit([]float32{1, 0, 0, 0}), 3, 0.0)
	if err != nil {
		t.Fatalf("Query (1): %v", err)
	}

	// Phase 2 — fresh process, same path, expect identical results.
	idx2, err := NewCosineIndex(dir, NamespaceChunks)
	if err != nil {
		t.Fatalf("NewCosineIndex (2): %v", err)
	}
	if got := idx2.Len(); got != 5 {
		t.Fatalf("after reload Len()=%d, want 5", got)
	}
	hits2, err := idx2.Query(ctx, unit([]float32{1, 0, 0, 0}), 3, 0.0)
	if err != nil {
		t.Fatalf("Query (2): %v", err)
	}
	if len(hits1) != len(hits2) {
		t.Fatalf("hits len mismatch %d vs %d", len(hits1), len(hits2))
	}
	for i := range hits1 {
		if hits1[i].ID != hits2[i].ID {
			t.Errorf("rank %d ID drift: %s vs %s", i, hits1[i].ID, hits2[i].ID)
		}
		if math.Abs(float64(hits1[i].Score-hits2[i].Score)) > 1e-5 {
			t.Errorf("rank %d score drift: %.6f vs %.6f", i, hits1[i].Score, hits2[i].Score)
		}
	}
	// Spot-check the persisted Metadata survived round-trip.
	if v, ok := hits2[0].Metadata["src"]; !ok || v != "doc1" {
		t.Errorf("after reload metadata['src']=%v want 'doc1'", v)
	}
}

// TestCosineHandlesEmptyIndex — Query against an empty index returns
// []VectorHit{} (never nil) with no error.  Also asserts that Delete on an
// empty index is a benign no-op.
func TestCosineHandlesEmptyIndex(t *testing.T) {
	ctx := context.Background()
	idx := makeIndex(t, NamespaceChunks)

	hits, err := idx.Query(ctx, []float32{1, 0, 0, 0}, 5, 0.0)
	if err != nil {
		t.Fatalf("Query on empty: %v", err)
	}
	if hits == nil {
		t.Errorf("hits should be empty slice, got nil")
	}
	if len(hits) != 0 {
		t.Errorf("hits should be empty, got %d", len(hits))
	}
	if err := idx.Delete(ctx, []string{"nonexistent"}); err != nil {
		t.Errorf("Delete on empty index should be no-op: %v", err)
	}
}

// TestCosineMetadataRoundtrip — Upsert with Metadata={"foo": "bar"}, Query,
// assert the hit's Metadata equals the original (and is a defensive copy
// so caller mutations don't poison the index).
func TestCosineMetadataRoundtrip(t *testing.T) {
	ctx := context.Background()
	idx := makeIndex(t, NamespaceEntities)

	v := unit([]float32{1, 2, 3, 4})
	if err := idx.Upsert(ctx, []VectorRecord{{
		ID:       "x",
		Vector:   v,
		Metadata: map[string]any{"foo": "bar", "weight": 0.5},
	}}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	hits, err := idx.Query(ctx, v, 1, 0.0)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("want 1 hit, got %d", len(hits))
	}
	if got := hits[0].Metadata["foo"]; got != "bar" {
		t.Errorf("Metadata['foo']=%v, want 'bar'", got)
	}
	if got, ok := hits[0].Metadata["weight"].(float64); !ok || math.Abs(got-0.5) > 1e-9 {
		// Note: floats from JSON would be float64; from in-memory they're
		// the original Go value — both are compared as float64 here.
		// Value entered was 0.5 (untyped float, defaults to float64).
		t.Errorf("Metadata['weight']=%v, want 0.5", hits[0].Metadata["weight"])
	}

	// Mutate caller's hit map and re-query to confirm we returned a copy.
	hits[0].Metadata["foo"] = "tampered"
	hits2, _ := idx.Query(ctx, v, 1, 0.0)
	if hits2[0].Metadata["foo"] != "bar" {
		t.Errorf("mutation leaked into index: got %v", hits2[0].Metadata["foo"])
	}
}

// TestCosineDimensionMismatchRejected — the second insertion with a wrong-
// length vector must be rejected, leaving the index unchanged.  Not in the
// spec's six-test list but cheap insurance against silent dim drift.
func TestCosineDimensionMismatchRejected(t *testing.T) {
	ctx := context.Background()
	idx := makeIndex(t, NamespaceChunks)
	if err := idx.Upsert(ctx, []VectorRecord{{ID: "a", Vector: unit([]float32{1, 0, 0, 0})}}); err != nil {
		t.Fatalf("Upsert (good): %v", err)
	}
	err := idx.Upsert(ctx, []VectorRecord{{ID: "b", Vector: []float32{1, 0, 0}}})
	if err == nil {
		t.Fatalf("expected dimension-mismatch error, got nil")
	}
	if idx.Len() != 1 {
		t.Errorf("after rejected Upsert, Len=%d, want 1", idx.Len())
	}
}
