package main

import (
	"context"
	"strings"
	"testing"
)

// newTestPipeline returns a pipeline wired with the deterministic mock provider
// + mock embedder so tests run offline and reproducibly.
func newTestPipeline(t *testing.T) (*Pipeline, *MockProvider) {
	t.Helper()
	mp := NewMockProvider()
	pipe := NewPipeline(mp, NewMockEmbedder(), NewInMemoryVectorStore(), NewMemoryKVStore())
	return pipe, mp
}

// TestPipelineEndToEndWithMock — insert 3 short docs, query a phrase that
// uniquely belongs to one of them, assert References include that doc's chunk.
func TestPipelineEndToEndWithMock(t *testing.T) {
	pipe, mock := newTestPipeline(t)
	ctx := context.Background()
	docs := map[string]string{
		"d1": "Paragraph about apples and pies.",
		"d2": "Scrooge is a miser who hates Christmas.",
		"d3": "Lorem ipsum dolor sit amet.",
	}
	for id, body := range docs {
		if err := pipe.Insert(ctx, id, body); err != nil {
			t.Fatalf("Insert(%s): %v", id, err)
		}
	}
	res, err := pipe.Query(ctx, "Tell me about Scrooge", 3)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(res.References) == 0 {
		t.Fatal("expected at least one reference, got 0")
	}
	// d2's only chunk has index 0.
	wantRef := "d2::chunk-0"
	found := false
	for _, ref := range res.References {
		if ref == wantRef {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("references %v missing %q (Scrooge doc)", res.References, wantRef)
	}
	if len(mock.Calls) != 1 {
		t.Fatalf("MockProvider should have been called once; got %d", len(mock.Calls))
	}
	if !strings.Contains(mock.Calls[0].System, "Scrooge") {
		t.Fatalf("system prompt should mention Scrooge:\n%s", mock.Calls[0].System)
	}
}

// TestInsertChunkCount — insert a doc with 3 paragraphs, assert 3 KV records.
func TestInsertChunkCount(t *testing.T) {
	pipe, _ := newTestPipeline(t)
	ctx := context.Background()
	doc := "Paragraph one body.\n\nParagraph two body.\n\nParagraph three body."
	if err := pipe.Insert(ctx, "doc1", doc); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	wantIDs := []string{"doc1::chunk-0", "doc1::chunk-1", "doc1::chunk-2"}
	got, err := pipe.KV.GetByIDs(ctx, wantIDs)
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 chunks in KV; got %d (%v)", len(got), got)
	}
	for _, id := range wantIDs {
		if _, ok := got[id]; !ok {
			t.Fatalf("missing chunk %q in KV", id)
		}
	}
	// Missing-key check: a fourth chunk should not exist.
	missing, err := pipe.KV.FilterMissing(ctx, []string{"doc1::chunk-3"})
	if err != nil {
		t.Fatalf("FilterMissing: %v", err)
	}
	if len(missing) != 1 {
		t.Fatalf("expected 1 missing key; got %d (%v)", len(missing), missing)
	}
}

// TestQueryUsesTopChunk — ingest two paragraphs with very different content;
// query a phrase from paragraph 2; assert top-1 reference is paragraph 2.
func TestQueryUsesTopChunk(t *testing.T) {
	pipe, _ := newTestPipeline(t)
	ctx := context.Background()
	body := "Apples grow on trees in orchards.\n\nThe quantum entanglement of particles defies intuition."
	if err := pipe.Insert(ctx, "doc", body); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	res, err := pipe.Query(ctx, "quantum entanglement of particles", 2)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(res.References) == 0 {
		t.Fatal("no references returned")
	}
	if res.References[0] != "doc::chunk-1" {
		t.Fatalf("top-1 ref should be doc::chunk-1 (paragraph 2); got %q (full=%v)", res.References[0], res.References)
	}
}

// TestProviderInterfaceContract is purely a compile-time check: both concrete
// providers MUST satisfy the Provider interface, and both embedders MUST
// satisfy EmbeddingProvider. If a future commit breaks one of these, the file
// won't compile and `go test` will fail loudly before any assertion runs.
func TestProviderInterfaceContract(t *testing.T) {
	var _ Provider = (*MockProvider)(nil)
	var _ Provider = (*OpenAIProvider)(nil)
	var _ EmbeddingProvider = (*MockEmbedder)(nil)
	var _ EmbeddingProvider = (*OpenAIEmbedder)(nil)
	var _ VectorStore = (*InMemoryVectorStore)(nil)
	var _ KVStore = (*MemoryKVStore)(nil)
}

// TestMockProviderDeterministic — call Complete twice with identical request,
// assert identical Text. Without this guarantee tests would flap.
func TestMockProviderDeterministic(t *testing.T) {
	mp := NewMockProvider()
	req := CompleteRequest{
		System:   "You are a helpful assistant.\nContext:\n[chunk d1::chunk-0]\nApples and pies.\n",
		Messages: []Message{{Role: "user", Content: "What is in the context?"}},
	}
	a, err := mp.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete A: %v", err)
	}
	b, err := mp.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete B: %v", err)
	}
	if a.Text != b.Text {
		t.Fatalf("MockProvider non-deterministic:\nA=%q\nB=%q", a.Text, b.Text)
	}
}
