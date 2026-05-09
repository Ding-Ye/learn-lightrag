package main

// Pipeline ties the five s01 pieces (chunker, embedder, vector store, KV store,
// LLM provider) into a single Insert + Query loop. It's the s01 analog of
// upstream's `LightRAG.ainsert()` (lightrag/lightrag.py:1237) plus
// `LightRAG.aquery()` (lightrag/lightrag.py:2622) collapsed into ~80 LOC.
//
// Reading map (which upstream lines this file shadows):
//   - Insert  ↔ ainsert (L1237) → apipeline_enqueue_documents → chunking → embed → vdb.upsert + kv.upsert
//   - Query   ↔ aquery  (L2622) → naive_query path (mode="naive")

import (
	"context"
	"fmt"
	"strings"
)

// QueryMode mirrors upstream's QueryParam.mode field.
type QueryMode string

const (
	ModeNaive  QueryMode = "naive"
	ModeLocal  QueryMode = "local"
	ModeGlobal QueryMode = "global"
	ModeHybrid QueryMode = "hybrid"
)

// QueryResult is the shape every chapter returns to the caller.
type QueryResult struct {
	Content    string    // final answer text from the LLM
	References []string  // chunk IDs cited in the context window
	Mode       QueryMode // which retrieval strategy ran
}

// Pipeline is s01's tiny RAG. Future chapters add doc-status, real chunking,
// graph retrieval, multiple modes — but the (Provider, Embedder, VDB, KV)
// shape stays.
type Pipeline struct {
	Provider  Provider
	Embedder  EmbeddingProvider
	VDB       VectorStore
	KV        KVStore
	ChunkSize int  // max runes per chunk (s01 stub)
	Verbose   bool // when true, print debug lines via Logf
	Logf      func(format string, a ...any)
}

// NewPipeline builds a default pipeline with sane stubs.
func NewPipeline(p Provider, e EmbeddingProvider, vdb VectorStore, kv KVStore) *Pipeline {
	return &Pipeline{
		Provider:  p,
		Embedder:  e,
		VDB:       vdb,
		KV:        kv,
		ChunkSize: 1200,
		Logf:      func(string, ...any) {},
	}
}

// chunkID is the canonical "<docID>::chunk-<index>" key used both by KV and VDB.
func chunkID(docID string, idx int) string {
	return fmt.Sprintf("%s::chunk-%d", docID, idx)
}

// Insert chunks the doc, embeds each chunk, writes to VDB + KV. This is the
// s01 condensation of upstream's full ingestion pipeline (which also runs
// extraction + summarization in s09/s10).
func (p *Pipeline) Insert(ctx context.Context, docID, text string) error {
	chunks := ChunkByNewlines(docID, text, p.ChunkSize)
	if len(chunks) == 0 {
		return fmt.Errorf("pipeline: doc %q produced 0 chunks (empty text?)", docID)
	}
	bodies := make([]string, len(chunks))
	for i, c := range chunks {
		bodies[i] = c.Content
	}
	vecs, err := p.Embedder.Embed(ctx, bodies)
	if err != nil {
		return fmt.Errorf("pipeline: embed: %w", err)
	}
	if len(vecs) != len(chunks) {
		return fmt.Errorf("pipeline: embedder returned %d vectors for %d chunks", len(vecs), len(chunks))
	}
	records := make([]VectorRecord, len(chunks))
	kvItems := make(map[string]map[string]any, len(chunks))
	for i, c := range chunks {
		id := chunkID(docID, c.ChunkOrderIndex)
		records[i] = VectorRecord{
			ID:     id,
			Vector: vecs[i],
			Metadata: map[string]any{
				"doc_id":            c.ContentDocID,
				"chunk_order_index": c.ChunkOrderIndex,
			},
		}
		kvItems[id] = map[string]any{
			"content":           c.Content,
			"doc_id":            c.ContentDocID,
			"chunk_order_index": c.ChunkOrderIndex,
			"tokens":            c.Tokens,
		}
	}
	if err := p.VDB.Upsert(ctx, records); err != nil {
		return fmt.Errorf("pipeline: vdb upsert: %w", err)
	}
	if err := p.KV.Upsert(ctx, kvItems); err != nil {
		return fmt.Errorf("pipeline: kv upsert: %w", err)
	}
	if p.Verbose {
		p.Logf("[insert] doc=%s chunks=%d", docID, len(chunks))
	}
	return nil
}

// Query embeds the question, scans the VDB, fetches chunk content from KV,
// builds a context-augmented system prompt, and asks the Provider.
//
// References lists the chunk IDs the LLM saw — every chapter from here on
// preserves this contract so the Web doc viewer can render citations.
func (p *Pipeline) Query(ctx context.Context, q string, topK int) (QueryResult, error) {
	if topK <= 0 {
		topK = 3
	}
	qVecs, err := p.Embedder.Embed(ctx, []string{q})
	if err != nil {
		return QueryResult{}, fmt.Errorf("pipeline: embed query: %w", err)
	}
	hits, err := p.VDB.Query(ctx, qVecs[0], topK, -1.0) // -1.0 = no threshold in s01
	if err != nil {
		return QueryResult{}, fmt.Errorf("pipeline: vdb query: %w", err)
	}
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.ID
	}
	contents, err := p.KV.GetByIDs(ctx, ids)
	if err != nil {
		return QueryResult{}, fmt.Errorf("pipeline: kv getbyids: %w", err)
	}
	var ctxBuf strings.Builder
	ctxBuf.WriteString("Context:\n")
	for _, id := range ids {
		row := contents[id]
		body, _ := row["content"].(string)
		fmt.Fprintf(&ctxBuf, "[chunk %s]\n%s\n\n", id, body)
	}
	system := "You are a helpful assistant. Answer using ONLY the context below. " +
		"Cite chunk IDs in [brackets].\n\n" + ctxBuf.String()
	if p.Verbose {
		p.Logf("[query] q=%q topK=%d retrieved=%v", q, topK, ids)
	}
	resp, err := p.Provider.Complete(ctx, CompleteRequest{
		System:      system,
		Messages:    []Message{{Role: "user", Content: q}},
		MaxTokens:   512,
		Temperature: 0.0,
	})
	if err != nil {
		return QueryResult{}, fmt.Errorf("pipeline: provider complete: %w", err)
	}
	return QueryResult{
		Content:    resp.Text,
		References: ids,
		Mode:       ModeNaive,
	}, nil
}
