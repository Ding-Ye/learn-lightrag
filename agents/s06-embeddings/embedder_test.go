package main

// Tests for s06. All seven tests run offline:
//   - OpenAIEmbedder paths use httptest.NewServer to fake `/embeddings`.
//   - MockEmbedder is itself a pure function — tests are deterministic.
//
// CI passes -count=1 to avoid the test-cache so the 429-retry test really
// runs every time (it has wall-clock-sensitive backoff).

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// TestEmbeddingProviderInterfaceContract is a compile-time guard ensuring
// both concrete types satisfy EmbeddingProvider. Removing a method from
// either struct fails to build.
func TestEmbeddingProviderInterfaceContract(t *testing.T) {
	var _ EmbeddingProvider = (*OpenAIEmbedder)(nil)
	var _ EmbeddingProvider = (*MockEmbedder)(nil)
}

// TestEmbedderBatchesAtLimit asserts that BatchSize=10 + 25 inputs results
// in EXACTLY 3 HTTP calls. This is the load-bearing claim of "batching"
// for s06 — if it failed we'd be paying per-call HTTP overhead like s01.
func TestEmbedderBatchesAtLimit(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req embedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode: %v", err)
		}
		// Echo back one canned 1536-d vector per input.
		data := make([]struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}, len(req.Input))
		for i := range req.Input {
			vec := make([]float32, 1536)
			vec[0] = float32(i) + 0.5
			data[i] = struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{Index: i, Embedding: vec}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer server.Close()

	emb := NewOpenAIEmbedder(
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithBatchSize(10),
		WithMaxRetries(0),
	)
	texts := make([]string, 25)
	for i := range texts {
		texts[i] = "text-" + string(rune('a'+i%26))
	}
	vecs, err := emb.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 25 {
		t.Errorf("got %d vectors, want 25", len(vecs))
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("server saw %d calls, want exactly 3 (batches of 10/10/5)", got)
	}
}

// TestEmbedderDimMatchesProvider asserts both impls return Dim()=1536.
// s07's VectorStore reads Dim() at construction; if the two impls
// disagreed, swapping providers would silently corrupt the index.
func TestEmbedderDimMatchesProvider(t *testing.T) {
	openai := NewOpenAIEmbedder(WithAPIKey("k"))
	mock := NewMockEmbedder()
	if d := openai.Dim(); d != 1536 {
		t.Errorf("OpenAIEmbedder.Dim() = %d, want 1536", d)
	}
	if d := mock.Dim(); d != 1536 {
		t.Errorf("MockEmbedder.Dim() = %d, want 1536", d)
	}
}

// TestEmbedderHandlesEmptyInputSlice asserts that Embed(nil) returns a
// non-nil empty slice WITHOUT making any HTTP call. Doc-status pipelines
// invoke Embed for chunk-empty docs (e.g., a doc with only a title) and
// must not pay OpenAI quota for nothing.
func TestEmbedderHandlesEmptyInputSlice(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer server.Close()

	emb := NewOpenAIEmbedder(
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithBatchSize(8),
		WithMaxRetries(0),
	)
	for _, name := range []string{"nil", "empty"} {
		var input []string
		if name == "empty" {
			input = []string{}
		}
		vecs, err := emb.Embed(context.Background(), input)
		if err != nil {
			t.Errorf("[%s] Embed: %v", name, err)
		}
		if vecs == nil {
			t.Errorf("[%s] vecs is nil, want non-nil empty slice", name)
		}
		if len(vecs) != 0 {
			t.Errorf("[%s] vecs len = %d, want 0", name, len(vecs))
		}
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("server saw %d calls, want 0 (empty input must not hit network)", got)
	}

	// Mock embedder must behave the same way.
	mock := NewMockEmbedder()
	v, err := mock.Embed(context.Background(), nil)
	if err != nil {
		t.Errorf("Mock Embed(nil): %v", err)
	}
	if v == nil || len(v) != 0 {
		t.Errorf("Mock Embed(nil) = %v, want non-nil empty slice", v)
	}
}

// TestEmbedderRetriesOn429 returns 429 twice then 200. Asserts the retry
// loop made exactly 3 attempts AND the final result decoded correctly.
// This verifies (a) transient-status detection, (b) backoff did not
// spuriously give up, (c) successful response after retry is not lost.
func TestEmbedderRetriesOn429(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit"}}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"data": [
				{"index": 0, "embedding": [0.1, 0.2, 0.3]}
			]
		}`))
	}))
	defer server.Close()

	emb := NewOpenAIEmbedder(
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithBatchSize(4),
		WithMaxRetries(3),
	)
	vecs, err := emb.Embed(context.Background(), []string{"hi"})
	if err != nil {
		t.Fatalf("Embed after retries: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 3 {
		t.Fatalf("vecs = %v, want one vector of 3 floats", vecs)
	}
	if vecs[0][0] != 0.1 {
		t.Errorf("vecs[0][0] = %v, want 0.1 (final result lost across retries)", vecs[0][0])
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("server saw %d attempts, want 3 (initial + 2 retries)", got)
	}
}

// TestMockEmbedderDeterministic asserts that Embed(["hello"]) twice yields
// byte-identical vectors. This is the property that makes MockEmbedder
// safe in CI — assertions over its output can use exact comparisons.
func TestMockEmbedderDeterministic(t *testing.T) {
	mock := NewMockEmbedder()
	a, err := mock.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("first Embed: %v", err)
	}
	b, err := mock.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("second Embed: %v", err)
	}
	if len(a) != 1 || len(b) != 1 {
		t.Fatalf("each Embed should return 1 vector, got %d / %d", len(a), len(b))
	}
	if len(a[0]) != 1536 || len(b[0]) != 1536 {
		t.Fatalf("vector dim = (%d, %d), want (1536, 1536)", len(a[0]), len(b[0]))
	}
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatalf("non-deterministic at dim %d: %v vs %v", i, a[0][i], b[0][i])
		}
	}

	// Different input must yield a different vector (sanity check).
	c, err := mock.Embed(context.Background(), []string{"world"})
	if err != nil {
		t.Fatalf("third Embed: %v", err)
	}
	allEqual := true
	for i := range a[0] {
		if a[0][i] != c[0][i] {
			allEqual = false
			break
		}
	}
	if allEqual {
		t.Errorf("Embed('hello') == Embed('world') byte-for-byte; sha256 collision unlikely")
	}
}

// TestMockEmbedderUnitNormalized asserts ‖v‖² ≈ 1.0 within 1e-5. Unit
// vectors make cosine similarity equal dot product; s07's tests rely on
// that simplification.
func TestMockEmbedderUnitNormalized(t *testing.T) {
	mock := NewMockEmbedder()
	for _, text := range []string{"a", "hello world", "Eleanor stood at the lighthouse window."} {
		vecs, err := mock.Embed(context.Background(), []string{text})
		if err != nil {
			t.Fatalf("Embed(%q): %v", text, err)
		}
		var sumSq float64
		for _, x := range vecs[0] {
			sumSq += float64(x) * float64(x)
		}
		if math.Abs(sumSq-1.0) > 1e-5 {
			t.Errorf("text=%q: ‖v‖² = %.8f, want ≈1.0 (tolerance 1e-5)", text, sumSq)
		}
	}
}
