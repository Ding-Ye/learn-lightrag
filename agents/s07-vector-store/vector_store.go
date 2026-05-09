package main

import "context"

// Namespace is a stringly-typed handle for the THREE distinct vector indices
// that LightRAG maintains: chunks, entities, relations.  We model it as a
// type alias rather than a Go type so callers can pass plain string literals
// while still keeping the doc-level intent visible in signatures.  The
// canonical values are kept here so a learner has a single grep-target.
type Namespace = string

const (
	NamespaceChunks    Namespace = "chunks"
	NamespaceEntities  Namespace = "entities"
	NamespaceRelations Namespace = "relations"
)

// VectorRecord is the unit of insertion: a stable ID, a dense float32 vector,
// and an open-ended Metadata bag whose contents survive through Query() so a
// caller can attach (e.g.) the chunk's text or the entity's display name and
// pull it back out without a second store lookup.
type VectorRecord struct {
	ID       string
	Vector   []float32
	Metadata map[string]any
}

// VectorHit is what Query() returns: the original ID, the cosine similarity
// score (in `[-1, 1]`; LightRAG conventionally clamps it to `[0, 1]` and
// thresholds the bottom), and the Metadata that was attached at Upsert time.
type VectorHit struct {
	ID       string
	Score    float32
	Metadata map[string]any
}

// VectorStore is the interface every backend implements.  The shape mirrors
// upstream `lightrag/base.py:189 BaseVectorStorage` minus the upstream's
// async-only sugar — every method takes a context.Context for cancellation.
//
// The Query contract: return up to topK hits with Score >= threshold, sorted
// in descending Score order.  An empty store yields an empty slice (not nil)
// and no error.  The threshold is INCLUSIVE — a hit exactly at threshold is
// kept; this matches upstream's `better_than_threshold` semantic.
type VectorStore interface {
	Upsert(ctx context.Context, records []VectorRecord) error
	Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error)
	Delete(ctx context.Context, ids []string) error
	Persist(ctx context.Context) error
}
