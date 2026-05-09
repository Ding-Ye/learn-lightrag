package main

import (
	"context"
	"fmt"
)

// query_hybrid.go is the headline mode: runs BOTH local and global retrieval
// pipelines in parallel-ish (sequentially in this port for clarity), dedupes
// chunks by ID, applies a SINGLE token budget, then makes ONE final LLM call.
// Mirrors operate.py:3164-3410 kg_query when QueryParam.mode == "hybrid".
//
// The didactic point: hybrid is not "two answers concatenated".  It's two
// retrievals that hand off to ONE synthesis prompt — that's the whole reason
// dedup matters.  If you merge after generation you get inconsistent voice;
// if you merge before generation the LLM reconciles for you.

func queryHybrid(ctx context.Context, p *Pipeline, q string, param QueryParam) (QueryResult, error) {
	if err := assertPipelineForKG(p); err != nil {
		return QueryResult{}, err
	}

	// ONE keyword extraction call serves both halves.  Avoids paying twice.
	hl, ll, err := extractKeywords(ctx, p.Provider, q)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: keywords: %w", err)
	}

	// Local half — entity-centric.  Fall back to whole query if no LL.
	llKeys := ll
	if len(llKeys) == 0 {
		llKeys = []string{q}
	}
	llVecs, err := p.Embedder.Embed(ctx, llKeys)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: embed ll: %w", err)
	}
	entSet, locRelSet, locChunkIDs, err := retrieveLocal(ctx, p, llVecs, param)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: local: %w", err)
	}

	// Global half — relation-centric.  Fall back to whole query if no HL.
	hlKeys := hl
	if len(hlKeys) == 0 {
		hlKeys = []string{q}
	}
	hlVecs, err := p.Embedder.Embed(ctx, hlKeys)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: embed hl: %w", err)
	}
	gloEntSet, gloRelSet, gloChunkIDs, err := retrieveGlobal(ctx, p, hlVecs, param)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: global: %w", err)
	}

	// Merge entity & relation sets (dedup by canonical key).
	for k, v := range gloEntSet {
		entSet[k] = v
	}
	relSet := map[string]Relationship{}
	for k, v := range locRelSet {
		relSet[k] = v
	}
	for k, v := range gloRelSet {
		relSet[k] = v
	}

	// Dedup chunk IDs: local first, then global appends only the unseen.
	// This is the "dedup chunks" contract of the hybrid mode.
	chunkIDs := []string{}
	seen := map[string]struct{}{}
	for _, id := range locChunkIDs {
		if _, ok := seen[id]; !ok && id != "" {
			seen[id] = struct{}{}
			chunkIDs = append(chunkIDs, id)
		}
	}
	for _, id := range gloChunkIDs {
		if _, ok := seen[id]; !ok && id != "" {
			seen[id] = struct{}{}
			chunkIDs = append(chunkIDs, id)
		}
	}
	if param.ChunkTopK > 0 && len(chunkIDs) > param.ChunkTopK {
		chunkIDs = chunkIDs[:param.ChunkTopK]
	}
	rows, err := fetchChunks(ctx, p.KV, chunkIDs)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: kv: %w", err)
	}

	// One synthesis call.  buildSystemPrompt's MaxEntityTokens +
	// MaxRelationTokens budgets keep the entity/relation sections small; the
	// remainder of MaxTotalTokens flows into the chunks block.
	ents := mapValuesEntity(entSet)
	rels := mapValuesRelationship(relSet)
	sysPrompt := buildSystemPrompt(ents, rels, rows, q, param)

	resp, err := p.Provider.Complete(ctx, CompleteRequest{
		System:      sysPrompt,
		Messages:    []Message{{Role: "user", Content: q}},
		Temperature: 0.0,
	})
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryHybrid: complete: %w", err)
	}

	return QueryResult{
		Content:    resp.Text,
		References: chunkRowIDs(rows),
		Mode:       ModeHybrid,
	}, nil
}

// retrieveLocal is the "local half" of hybrid: takes pre-embedded
// low-level keywords, returns entity/relation/chunk-ID sets without
// invoking the LLM (the LLM call is reserved for the final synthesis in
// hybrid).  Mirrors queryLocal steps 3-5.
func retrieveLocal(
	ctx context.Context,
	p *Pipeline,
	llVecs [][]float32,
	param QueryParam,
) (map[string]Entity, map[string]Relationship, []string, error) {
	rawHits := make([]VectorHit, 0, len(llVecs)*param.TopK)
	for _, v := range llVecs {
		hh, err := p.VDBEnts.Query(ctx, v, param.TopK, -1.0)
		if err != nil {
			return nil, nil, nil, err
		}
		rawHits = append(rawHits, hh...)
	}
	hits := dedupHits(rawHits)

	entSet := map[string]Entity{}
	relSet := map[string]Relationship{}
	for _, h := range hits {
		entityID := stringMeta(h.Metadata, "entity_id", h.ID)
		if e, ok, _ := p.Graph.GetNode(ctx, entityID); ok {
			entSet[e.Name] = e
		}
		sg, err := p.Graph.GetSubgraph(ctx, entityID, 1, param.TopK)
		if err != nil {
			return nil, nil, nil, err
		}
		for _, n := range sg.Nodes {
			entSet[n.Name] = n
		}
		for _, e := range sg.Edges {
			relSet[edgeKey(e)] = e
		}
	}

	chunkIDs := []string{}
	seen := map[string]struct{}{}
	for _, e := range entSet {
		for _, id := range e.SourceIDs {
			if _, ok := seen[id]; !ok && id != "" {
				seen[id] = struct{}{}
				chunkIDs = append(chunkIDs, id)
			}
		}
	}
	for _, r := range relSet {
		for _, id := range r.SourceIDs {
			if _, ok := seen[id]; !ok && id != "" {
				seen[id] = struct{}{}
				chunkIDs = append(chunkIDs, id)
			}
		}
	}
	return entSet, relSet, chunkIDs, nil
}

// retrieveGlobal is the "global half" of hybrid: relation-centric.  Same
// non-LLM contract as retrieveLocal.
func retrieveGlobal(
	ctx context.Context,
	p *Pipeline,
	hlVecs [][]float32,
	param QueryParam,
) (map[string]Entity, map[string]Relationship, []string, error) {
	rawHits := make([]VectorHit, 0, len(hlVecs)*param.TopK)
	for _, v := range hlVecs {
		hh, err := p.VDBRels.Query(ctx, v, param.TopK, -1.0)
		if err != nil {
			return nil, nil, nil, err
		}
		rawHits = append(rawHits, hh...)
	}
	hits := dedupHits(rawHits)

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
	return entSet, relSet, chunkIDs, nil
}
