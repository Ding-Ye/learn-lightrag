package main

import (
	"context"
	"fmt"
)

// query_naive.go is the s01 baseline formalized: vector similarity over
// chunks only, no graph, no keyword extraction.  Mirrors upstream
// naive_query (operate.py:4930-5200) — but with the upstream's caching,
// streaming, and rerank knobs stripped out for didactic clarity.
//
// Steps:
//   1. Embed the query verbatim.
//   2. VDBChunks.Query for the top ChunkTopK hits.
//   3. KV.GetByIDs to fetch chunk contents.
//   4. buildSystemPrompt with empty entities/relations and the chunk rows.
//   5. Provider.Complete → return.

func queryNaive(ctx context.Context, p *Pipeline, q string, param QueryParam) (QueryResult, error) {
	if p.Embedder == nil {
		return QueryResult{}, fmt.Errorf("queryNaive: nil Embedder")
	}
	if p.VDBChunks == nil {
		return QueryResult{}, fmt.Errorf("queryNaive: nil VDBChunks")
	}
	if p.KV == nil {
		return QueryResult{}, fmt.Errorf("queryNaive: nil KV")
	}
	if p.Provider == nil {
		return QueryResult{}, fmt.Errorf("queryNaive: nil Provider")
	}

	// Step 1: embed the query.  One Embed call, one vector out.
	vecs, err := p.Embedder.Embed(ctx, []string{q})
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryNaive: embed: %w", err)
	}
	if len(vecs) != 1 {
		return QueryResult{}, fmt.Errorf("queryNaive: embedder returned %d vectors, want 1", len(vecs))
	}

	// Step 2: vector search over the chunks index.  topK = ChunkTopK; the
	// threshold -1.0 means "no minimum score" (all hits accepted).
	hits, err := p.VDBChunks.Query(ctx, vecs[0], param.ChunkTopK, -1.0)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryNaive: vector query: %w", err)
	}
	if len(hits) == 0 {
		return QueryResult{Mode: ModeNaive, Content: "no relevant chunks retrieved"}, nil
	}

	// Step 3: pull chunk content from KV.  Order preserved by hit order.
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.ID
	}
	rows, err := fetchChunks(ctx, p.KV, ids)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryNaive: kv: %w", err)
	}
	if len(rows) == 0 {
		return QueryResult{Mode: ModeNaive, Content: "no chunk content found"}, nil
	}

	// Step 4: assemble context with empty entity/relation sections.  Naive
	// mode IS the no-graph mode.
	sysPrompt := buildSystemPrompt(nil, nil, rows, q, param)

	// Step 5: ONE LLM call.
	resp, err := p.Provider.Complete(ctx, CompleteRequest{
		System:      sysPrompt,
		Messages:    []Message{{Role: "user", Content: q}},
		Temperature: 0.0,
	})
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryNaive: complete: %w", err)
	}

	refs := make([]string, 0, len(rows))
	for _, r := range rows {
		refs = append(refs, r.ID)
	}
	return QueryResult{
		Content:    resp.Text,
		References: refs,
		Mode:       ModeNaive,
	}, nil
}

// fetchChunks resolves a list of chunk IDs to chunkRow objects via KV
// lookup.  Missing IDs are silently skipped (the caller already has all
// the knowledge to decide whether that's an error).  Order is preserved
// against the input ids slice.
func fetchChunks(ctx context.Context, kv KVStore, ids []string) ([]chunkRow, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	got, err := kv.GetByIDs(ctx, ids)
	if err != nil {
		return nil, err
	}
	out := make([]chunkRow, 0, len(ids))
	for _, id := range ids {
		rec, ok := got[id]
		if !ok {
			continue
		}
		content := chunkContentField(rec)
		if content == "" {
			continue
		}
		out = append(out, chunkRow{ID: id, Content: content})
	}
	return out, nil
}
