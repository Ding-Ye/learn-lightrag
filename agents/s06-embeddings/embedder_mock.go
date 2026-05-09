package main

// MockEmbedder — deterministic sha256 → 1536-d unit vector. Used by every
// CI test in this repo (and in later sessions s07/s11 when CI must run
// offline). Two load-bearing properties:
//
//   1. Determinism: Embed(["hello"]) twice returns byte-identical vectors.
//      Tests assert byte equality across runs, processes, and machines.
//
//   2. Unit norm: sum(v[i]*v[i]) ≈ 1.0 within 1e-5. With unit vectors,
//      cosine similarity collapses to dot product, which makes s07's
//      vector-store tests cleaner (the "true" similarity formula has one
//      fewer normalisation step).
//
// Why sha256 → 1536d? sha256 emits 32 bytes (256 bits). Tile across 1536
// dims (= 48 copies). Each byte b maps to (float32(b) - 127.5) / 127.5,
// yielding values in [-1, 1]. Then L2-normalise the whole vector so the
// length is exactly 1.

import (
	"context"
	"crypto/sha256"
	"math"
)

// MockEmbedder has no fields besides Dim. It's a pure function.
type MockEmbedder struct {
	dim int
}

// NewMockEmbedder returns a 1536-d mock embedder. The dimension matches
// OpenAI's `text-embedding-3-small` so MockEmbedder is a drop-in
// replacement in s07's tests — neither store nor query path notices.
func NewMockEmbedder() *MockEmbedder {
	return &MockEmbedder{dim: 1536}
}

// Dim returns 1536. Constant; mock ignores the model name.
func (m *MockEmbedder) Dim() int { return m.dim }

// Embed maps each text to its sha256 → unit vector. ctx is accepted for
// interface conformance and to honour cancellation if a caller wraps a
// large slice; in practice mock work is microseconds-per-text.
func (m *MockEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = mockVector(t, m.dim)
	}
	return out, nil
}

// mockVector produces a deterministic dim-length unit vector from text.
// The sha256 digest is tiled across the dimension: dim 1536 = 48 × 32 bytes,
// so byte (i % 32) of the digest contributes to dim i. Each byte is
// centered to zero by subtracting 127.5 then divided by 127.5 to land in
// [-1, 1]. Finally we L2-normalise so ‖v‖ = 1.
func mockVector(text string, dim int) []float32 {
	digest := sha256.Sum256([]byte(text))
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		b := digest[i%len(digest)]
		v[i] = (float32(b) - 127.5) / 127.5
	}
	// L2 normalisation. We accumulate in float64 to keep the rounding
	// error tiny enough that the unit-norm test (tolerance 1e-5) passes
	// reliably across architectures.
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	if sumSq == 0 {
		// Should be impossible (sha256 of any input is non-zero) but
		// belt-and-suspenders. Return a unit vector pointing at axis 0.
		v[0] = 1
		return v
	}
	inv := float32(1.0 / math.Sqrt(sumSq))
	for i := range v {
		v[i] *= inv
	}
	return v
}
