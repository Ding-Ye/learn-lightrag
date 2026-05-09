package main

import (
	"context"
	"fmt"
	"sort"
)

// merge.go contains the helpers a downstream pipeline calls after s09's
// extractor has produced a slice of per-chunk Entity / Relationship records.
// The job: group by name (or by canonical edge key), call SummarizeDescriptions
// on the combined description fragments, and return ONE record per unique
// name carrying merged SourceIDs.
//
// Mirrors the spirit of upstream operate.py:1623-1947 (_merge_nodes_then_upsert)
// + :1948-2300 (_merge_edges_then_upsert), simplified — the upstream helpers
// also touch graph_storage and the entity vdb; s10 stays at the pure-merge
// layer and returns the result for the caller to upsert.

// MergeEntities groups ents by Name; for each group, calls
// SummarizeDescriptionsForName on the descriptions; returns one Entity per
// Name with merged Description + sorted SourceIDs.
//
// Type and Name are preserved from the FIRST occurrence (upstream picks the
// most-frequent type via mode; we keep first-writer-wins for didactic
// simplicity — extension exercise).  SourceIDs are sorted for deterministic
// output.
func MergeEntities(ctx context.Context, p Provider, ents []Entity) ([]Entity, error) {
	if len(ents) == 0 {
		return nil, nil
	}
	type group struct {
		name         string
		typ          string
		descriptions []string
		sources      map[string]struct{}
		order        int // first-seen index, preserves insertion order
	}
	groups := make(map[string]*group)
	orderCounter := 0
	for _, e := range ents {
		g, ok := groups[e.Name]
		if !ok {
			g = &group{
				name:    e.Name,
				typ:     e.Type,
				sources: make(map[string]struct{}),
				order:   orderCounter,
			}
			orderCounter++
			groups[e.Name] = g
		}
		if e.Description != "" {
			g.descriptions = append(g.descriptions, e.Description)
		}
		for _, sid := range e.SourceIDs {
			g.sources[sid] = struct{}{}
		}
	}

	// Stable iteration order so test output is deterministic.
	names := make([]string, 0, len(groups))
	for n := range groups {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		return groups[names[i]].order < groups[names[j]].order
	})

	out := make([]Entity, 0, len(groups))
	for _, name := range names {
		g := groups[name]
		merged, _, err := SummarizeDescriptionsForName(ctx, p,
			"entity", g.name, g.descriptions, 0, 0, 0)
		if err != nil {
			return nil, fmt.Errorf("MergeEntities[%s]: %w", g.name, err)
		}
		out = append(out, Entity{
			Name:        g.name,
			Type:        g.typ,
			Description: merged,
			SourceIDs:   sortedKeys(g.sources),
		})
	}
	return out, nil
}

// MergeRelationships groups rels by canonical (SrcID, TgtID) (upstream
// treats edges as undirected, so we sort the pair and use min-max as the
// key).  For each group, summarizes descriptions and merges weights/keywords.
//
// Weights are SUMMED (matches upstream operate.py:2059-2073 — "_aggregate_weight").
// Keywords are deduped + comma-joined.  SourceIDs are merged + sorted.
func MergeRelationships(ctx context.Context, p Provider, rels []Relationship) ([]Relationship, error) {
	if len(rels) == 0 {
		return nil, nil
	}
	type group struct {
		src, tgt     string
		descriptions []string
		keywords     []string
		keywordsSeen map[string]struct{}
		weight       float32
		sources      map[string]struct{}
		order        int
	}
	groups := make(map[[2]string]*group)
	orderCounter := 0
	for _, r := range rels {
		key := canonicalEdge(r.SrcID, r.TgtID)
		g, ok := groups[key]
		if !ok {
			g = &group{
				src:          key[0],
				tgt:          key[1],
				keywordsSeen: make(map[string]struct{}),
				sources:      make(map[string]struct{}),
				order:        orderCounter,
			}
			orderCounter++
			groups[key] = g
		}
		if r.Description != "" {
			g.descriptions = append(g.descriptions, r.Description)
		}
		if r.Keywords != "" {
			if _, seen := g.keywordsSeen[r.Keywords]; !seen {
				g.keywordsSeen[r.Keywords] = struct{}{}
				g.keywords = append(g.keywords, r.Keywords)
			}
		}
		g.weight += r.Weight
		for _, sid := range r.SourceIDs {
			g.sources[sid] = struct{}{}
		}
	}

	keys := make([][2]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return groups[keys[i]].order < groups[keys[j]].order
	})

	out := make([]Relationship, 0, len(groups))
	for _, k := range keys {
		g := groups[k]
		merged, _, err := SummarizeDescriptionsForName(ctx, p,
			"relation", relName(g.src, g.tgt), g.descriptions, 0, 0, 0)
		if err != nil {
			return nil, fmt.Errorf("MergeRelationships[%s-%s]: %w", g.src, g.tgt, err)
		}
		out = append(out, Relationship{
			SrcID:       g.src,
			TgtID:       g.tgt,
			Keywords:    joinNonEmpty(g.keywords, ","),
			Description: merged,
			Weight:      g.weight,
			SourceIDs:   sortedKeys(g.sources),
		})
	}
	return out, nil
}

// canonicalEdge returns the (min, max) ordered pair so undirected edges
// (A->B and B->A) hash to the same key.
func canonicalEdge(a, b string) [2]string {
	if a <= b {
		return [2]string{a, b}
	}
	return [2]string{b, a}
}

// relName produces a human-readable name for a relationship for prompt
// grounding ("Scrooge<->Marley").
func relName(src, tgt string) string {
	return src + "<->" + tgt
}

// sortedKeys returns the keys of set sorted ascending.
func sortedKeys(set map[string]struct{}) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// joinNonEmpty joins non-empty parts with sep.  Skips empties; if all
// inputs are empty, returns "".
func joinNonEmpty(parts []string, sep string) string {
	out := ""
	for _, p := range parts {
		if p == "" {
			continue
		}
		if out == "" {
			out = p
		} else {
			out += sep + p
		}
	}
	return out
}
