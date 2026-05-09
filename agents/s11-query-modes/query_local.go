package main

import (
	"context"
	"fmt"
	"strings"
)

// query_local.go owns the entity-centric retrieval mode.  Steps:
//
//   1. Extract dual-level keywords; use only LOW-level (entity-ish names).
//   2. Embed each low-level keyword (or fall back to embedding the query if
//      the LLM returned no keywords).
//   3. For each keyword embedding, VDBEntities.Query → top entities.
//   4. For each top entity, GraphStore.GetSubgraph(depth=1) to gather
//      neighbouring entities + the connecting edges.
//   5. Collect SourceIDs from all entities + edges; KV-fetch those chunks.
//   6. buildSystemPrompt with entities + edges + chunks; honour budgets.
//   7. Provider.Complete → return.
//
// Mirrors upstream operate.py:3700+ _build_local_query_context.  The big
// simplifications vs upstream: we don't rerank, don't apply per-entity
// novelty scoring, and don't deduplicate by description.  Test coverage
// for those is parked under the s_full integration chapter.

func queryLocal(ctx context.Context, p *Pipeline, q string, param QueryParam) (QueryResult, error) {
	if err := assertPipelineForKG(p); err != nil {
		return QueryResult{}, err
	}

	// Step 1: dual-level keyword extraction → take low-level only.
	_, ll, err := extractKeywords(ctx, p.Provider, q)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryLocal: keywords: %w", err)
	}
	// Step 2: keyword embeddings.  Empty list → fall back to whole query.
	keys := ll
	if len(keys) == 0 {
		keys = []string{q}
	}
	kvecs, err := p.Embedder.Embed(ctx, keys)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryLocal: embed keywords: %w", err)
	}

	// Step 3: vector search over entity index.  We aggregate hits across
	// all keyword embeddings; dedupHits keeps the highest-score entry per
	// entity ID.
	rawHits := make([]VectorHit, 0, len(kvecs)*param.TopK)
	for _, v := range kvecs {
		hh, err := p.VDBEnts.Query(ctx, v, param.TopK, -1.0)
		if err != nil {
			return QueryResult{}, fmt.Errorf("queryLocal: entity vector query: %w", err)
		}
		rawHits = append(rawHits, hh...)
	}
	hits := dedupHits(rawHits)

	// Step 4: subgraph expansion per top entity.  Combine into one set of
	// nodes + edges; cap at TopK*3 to keep the prompt bounded.
	entSet := map[string]Entity{}
	relSet := map[string]Relationship{}
	for _, h := range hits {
		entityID := stringMeta(h.Metadata, "entity_id", h.ID)
		if e, ok, _ := p.Graph.GetNode(ctx, entityID); ok {
			entSet[e.Name] = e
		}
		sg, err := p.Graph.GetSubgraph(ctx, entityID, 1, param.TopK)
		if err != nil {
			return QueryResult{}, fmt.Errorf("queryLocal: subgraph(%s): %w", entityID, err)
		}
		for _, n := range sg.Nodes {
			entSet[n.Name] = n
		}
		for _, e := range sg.Edges {
			relSet[edgeKey(e)] = e
		}
	}

	// Step 5: gather chunk SourceIDs from entities + edges.  Preserve
	// insertion order (entities first, then edges) — the heuristic matches
	// upstream which gives entity-anchored chunks priority over edge-only.
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
	if param.ChunkTopK > 0 && len(chunkIDs) > param.ChunkTopK {
		chunkIDs = chunkIDs[:param.ChunkTopK]
	}
	rows, err := fetchChunks(ctx, p.KV, chunkIDs)
	if err != nil {
		return QueryResult{}, fmt.Errorf("queryLocal: kv: %w", err)
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
		return QueryResult{}, fmt.Errorf("queryLocal: complete: %w", err)
	}

	refs := chunkRowIDs(rows)
	return QueryResult{
		Content:    resp.Text,
		References: refs,
		Mode:       ModeLocal,
	}, nil
}

// =============================================================================
// helpers shared by local / global / hybrid
// =============================================================================

// assertPipelineForKG validates the storages graph-aware modes need.
func assertPipelineForKG(p *Pipeline) error {
	if p.Provider == nil {
		return fmt.Errorf("nil Provider")
	}
	if p.Embedder == nil {
		return fmt.Errorf("nil Embedder")
	}
	if p.KV == nil {
		return fmt.Errorf("nil KV")
	}
	if p.VDBEnts == nil {
		return fmt.Errorf("nil VDBEnts")
	}
	if p.VDBRels == nil {
		return fmt.Errorf("nil VDBRels")
	}
	if p.Graph == nil {
		return fmt.Errorf("nil Graph")
	}
	return nil
}

// dedupHits collapses repeated hits (same ID, possibly multiple keyword
// queries) into one entry per ID, keeping the highest score.  Order:
// descending score.
func dedupHits(in []VectorHit) []VectorHit {
	best := map[string]VectorHit{}
	for _, h := range in {
		cur, ok := best[h.ID]
		if !ok || h.Score > cur.Score {
			best[h.ID] = h
		}
	}
	out := make([]VectorHit, 0, len(best))
	for _, h := range best {
		out = append(out, h)
	}
	// stable order by score desc
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Score > out[j-1].Score; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// stringMeta reads a string-typed key from a hit's metadata map, falling
// back to def when absent or when the value is not a string.
func stringMeta(m map[string]any, key, def string) string {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	return def
}

// edgeKey is the canonical key for an edge in a dedup map.  Undirected:
// (a,b) and (b,a) hash the same.
func edgeKey(r Relationship) string {
	a, b := r.SrcID, r.TgtID
	if strings.Compare(a, b) > 0 {
		a, b = b, a
	}
	return a + "||" + b
}

// mapValuesEntity returns the values of an Entity-keyed map as a slice.
func mapValuesEntity(m map[string]Entity) []Entity {
	out := make([]Entity, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}

// mapValuesRelationship returns the values of a Relationship-keyed map.
func mapValuesRelationship(m map[string]Relationship) []Relationship {
	out := make([]Relationship, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}

// chunkRowIDs extracts the IDs from chunkRow slices for QueryResult.References.
func chunkRowIDs(rows []chunkRow) []string {
	out := make([]string, len(rows))
	for i, r := range rows {
		out[i] = r.ID
	}
	return out
}
