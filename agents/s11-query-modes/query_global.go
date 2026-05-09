package main

import (
	"context"
	"fmt"
)

// query_global.go owns the relation-centric retrieval mode.  Steps:
//
//   1. Extract dual-level keywords; use only HIGH-level (themes / concepts).
//   2. Embed each high-level keyword (or fall back to the whole query).
//   3. VDBRelations.Query → top relations.
//   4. For each relation, attach its endpoint entities (via GraphStore).
//   5. Gather chunk SourceIDs from relations + endpoint entities.
//   6. buildSystemPrompt with entities + relations + chunks.
//   7. Provider.Complete → return.
//
// Mirrors upstream operate.py:3850+ _build_global_query_context.  The
// crucial design choice — high-level keywords drive global mode — is
// what makes this orthogonal to local: queries about "themes" or "what
// kinds of conflicts arise" surface relations in global, while "who is
// X" / "where does Y work" surface entities in local.

func queryGlobal(ctx context.Context, p *Pipeline, q string, param QueryParam) (QueryResult, error) {
	if err := assertPipelineForKG(p); err != nil {
		return QueryResult{}, err
	}

	// Step 1: dual-level keyword extraction → take HIGH-level only.
	hl, _, err := extractKeywords(ctx, p.Provider, q)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryGlobal: keywords: %w", err)
	}

	// Step 2: embed each high-level keyword.  Fall back to the whole query
	// if the LLM judged the query too vague to extract themes.
	keys := hl
	if len(keys) == 0 {
		keys = []string{q}
	}
	kvecs, err := p.Embedder.Embed(ctx, keys)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryGlobal: embed keywords: %w", err)
	}

	// Step 3: vector search over the relations index.  Aggregate hits
	// across keyword embeddings and dedup by relation key.
	rawHits := make([]VectorHit, 0, len(kvecs)*param.TopK)
	for _, v := range kvecs {
		hh, err := p.VDBRels.Query(ctx, v, param.TopK, -1.0)
		if err != nil {
			return QueryResult{}, fmt.Errorf("queryGlobal: relation vector query: %w", err)
		}
		rawHits = append(rawHits, hh...)
	}
	hits := dedupHits(rawHits)

	// Step 4: rebuild Relationship objects from metadata + grab endpoint
	// entities.  Each VDBRels record carries src_id / tgt_id / description /
	// keywords / source_ids fields in metadata so we can reconstruct without
	// a separate KV side-store.
	relSet := map[string]Relationship{}
	entSet := map[string]Entity{}
	for _, h := range hits {
		rel := relationFromMeta(h.ID, h.Metadata)
		relSet[edgeKey(rel)] = rel
		for _, endpt := range []string{rel.SrcID, rel.TgtID} {
			if endpt == "" {
				continue
			}
			if e, ok, _ := p.Graph.GetNode(ctx, endpt); ok {
				entSet[e.Name] = e
			}
		}
	}

	// Step 5: collect chunk SourceIDs.  Relations first (this is global
	// mode — relations are the primary axis), then their endpoint entities.
	chunkIDs := []string{}
	seen := map[string]struct{}{}
	for _, r := range relSet {
		for _, id := range r.SourceIDs {
			if _, ok := seen[id]; !ok && id != "" {
				seen[id] = struct{}{}
				chunkIDs = append(chunkIDs, id)
			}
		}
	}
	for _, e := range entSet {
		for _, id := range e.SourceIDs {
			if _, ok := seen[id]; !ok && id != "" {
				seen[id] = struct{}{}
				chunkIDs = append(chunkIDs, id)
			}
		}
	}
	if param.ChunkTopK > 0 && len(chunkIDs) > param.ChunkTopK {
		chunkIDs = chunkIDs[:param.ChunkTopK]
	}
	rows, err := fetchChunks(ctx, p.KV, chunkIDs)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryGlobal: kv: %w", err)
	}

	// Step 6: assemble.
	ents := mapValuesEntity(entSet)
	rels := mapValuesRelationship(relSet)
	sysPrompt := buildSystemPrompt(ents, rels, rows, q, param)

	// Step 7: synthesize.
	resp, err := p.Provider.Complete(ctx, CompleteRequest{
		System:      sysPrompt,
		Messages:    []Message{{Role: "user", Content: q}},
		Temperature: 0.0,
	})
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryGlobal: complete: %w", err)
	}

	return QueryResult{
		Content:    resp.Text,
		References: chunkRowIDs(rows),
		Mode:       ModeGlobal,
	}, nil
}

// relationFromMeta reconstructs a Relationship from a VDBRels hit's
// metadata.  All fields tolerate absence — caller can still use the partial
// shape for context assembly.  source_ids may be a []string or a []any
// depending on how the index was seeded.
func relationFromMeta(id string, m map[string]any) Relationship {
	r := Relationship{
		SrcID:       stringMeta(m, "src_id", ""),
		TgtID:       stringMeta(m, "tgt_id", ""),
		Keywords:    stringMeta(m, "keywords", ""),
		Description: stringMeta(m, "description", ""),
	}
	if v, ok := m["source_ids"]; ok {
		switch xs := v.(type) {
		case []string:
			r.SourceIDs = append(r.SourceIDs, xs...)
		case []any:
			for _, x := range xs {
				if s, ok := x.(string); ok && s != "" {
					r.SourceIDs = append(r.SourceIDs, s)
				}
			}
		}
	}
	if w, ok := m["weight"]; ok {
		switch v := w.(type) {
		case float32:
			r.Weight = v
		case float64:
			r.Weight = float32(v)
		}
	}
	return r
}
