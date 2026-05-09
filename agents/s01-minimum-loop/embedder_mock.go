package main

// MockEmbedder turns a text into a deterministic 1536-dim unit vector by
// expanding sha256(text) into floats in [-1, 1] then L2-normalizing.
// Used by every s01 test plus the offline `-provider mock` CLI flag.

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
)

// MockEmbedder satisfies EmbeddingProvider without any network call.
type MockEmbedder struct {
	dim int
}

// NewMockEmbedder returns a 1536-dim embedder (matches text-embedding-3-small).
func NewMockEmbedder() *MockEmbedder { return &MockEmbedder{dim: 1536} }

// Dim returns the fixed embedding dimension.
func (m *MockEmbedder) Dim() int { return m.dim }

// Embed maps each input text to a deterministic unit vector via sha256-stretching.
//
// The trick: hash(text) gives 32 bytes; we expand by hashing (text || counter)
// repeatedly to fill `dim` bytes, then map each byte to [-1, 1], then normalize.
// Same input → same vector → same cosine score → reproducible tests.
func (m *MockEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = hashToUnitVec(t, m.dim)
	}
	return out, nil
}

// hashToUnitVec expands sha256(text || counter) until it has `dim` floats,
// then L2-normalizes. Deterministic for a fixed text+dim pair.
func hashToUnitVec(text string, dim int) []float32 {
	v := make([]float32, dim)
	// Each sha256 round contributes 32 bytes (= 32 floats). We need ceil(dim/32) rounds.
	for round := 0; round*32 < dim; round++ {
		h := sha256.New()
		h.Write([]byte(text))
		var ctr [4]byte
		binary.BigEndian.PutUint32(ctr[:], uint32(round))
		h.Write(ctr[:])
		sum := h.Sum(nil)
		for j := 0; j < 32 && round*32+j < dim; j++ {
			// Map byte (0..255) into [-1, 1].
			v[round*32+j] = (float32(sum[j])/127.5 - 1.0)
		}
	}
	// L2 normalize so cosine == dot.
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	inv := float32(1.0 / norm)
	for i := range v {
		v[i] *= inv
	}
	return v
}
