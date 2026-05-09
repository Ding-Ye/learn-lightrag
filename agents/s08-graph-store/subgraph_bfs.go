package main

import (
	"sort"
)

// getSubgraphBFS implements upstream's `get_knowledge_graph(node_label,
// max_depth, max_nodes)` algorithm: a depth-limited breadth-first traversal
// where each level is processed in DESCENDING ORDER OF NODE DEGREE.  The
// degree-priority is what makes "Light**RAG**" actually useful for retrieval:
// when the budget runs out, you want the most-connected (most-information-
// dense) neighbors, not whichever ones the map iteration happened to yield.
//
// The result Subgraph contains:
//   - Nodes: every visited entity (in BFS-discovery order, NOT sort order;
//     callers can re-sort if they want).
//   - Edges: every relationship whose BOTH endpoints are in Nodes.  This
//     means a truncated traversal includes "partial" edges — the subgraph
//     is still well-formed (no edges dangling into excluded nodes).
//
// Cancellation: the outer GetSubgraph wrapper has already checked ctx.Err();
// the BFS loop is fast enough (≤ maxNodes iterations, each O(deg)) that
// we don't re-check inside.
//
// Diverges from upstream in two intentional ways:
//   - We don't carry the `is_truncated` flag; callers can compare
//     len(Nodes) to maxNodes themselves if they need it.
//   - We don't special-case the "*" wildcard; the s08 contract is "seed by
//     ID".  A wildcard mode is left as exercise (sorts ALL nodes by degree
//     DESC, takes top N — see docs).
func getSubgraphBFS(g *AdjacencyGraph, seed string, maxDepth, maxNodes int) (Subgraph, error) {
	// Empty / non-positive budgets are a degenerate but legal call — return
	// an empty Subgraph rather than nil so callers can range over it without
	// a nil check.  Mirrors upstream's "return empty KnowledgeGraph()" path.
	if seed == "" || maxNodes <= 0 || maxDepth < 0 {
		return Subgraph{Nodes: []Entity{}, Edges: []Relationship{}}, nil
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	// Seed not in graph: empty result, no error (matches upstream's
	// "Node ... not found in graph" warning path).
	if _, ok := g.nodes[seed]; !ok {
		return Subgraph{Nodes: []Entity{}, Edges: []Relationship{}}, nil
	}

	// `frontier` is the current BFS level — every node at the same depth.
	// We process the WHOLE level at once so the degree sort applies across
	// peers, not just within one parent's neighbor list.  This is the
	// difference between LightRAG's BFS and a textbook one.
	type levelEntry struct {
		id     string
		degree int
	}

	visited := make(map[string]struct{})
	visited[seed] = struct{}{}

	// Result accumulator.  Pre-size cap at maxNodes so the underlying array
	// doesn't reallocate during append.
	bfsOrder := make([]string, 0, maxNodes)
	bfsOrder = append(bfsOrder, seed)

	// Initialize the first frontier with just the seed.  Depth 0 is the
	// seed itself; depth 1 is its direct neighbors; depth maxDepth is the
	// last level we EXPAND from.  (A node at depth maxDepth+1 is reached
	// by expanding from depth maxDepth — but no further.)
	frontier := []levelEntry{{id: seed, degree: g.degreeUnlocked(seed)}}

	for depth := 0; depth < maxDepth && len(bfsOrder) < maxNodes && len(frontier) > 0; depth++ {
		// Build the next level from THIS level's neighbors.  We collect
		// all candidates first, then sort the level by degree DESC, then
		// take them in order until the maxNodes budget is exhausted.
		// Why collect-then-sort instead of expand-each-parent-then-sort?
		// Because the "level" is the cross-section of the BFS tree at a
		// given depth; degree priority is supposed to be GLOBAL within
		// that level, not local to one parent's neighbor list.
		nextCandidates := make([]levelEntry, 0)
		seenInLevel := make(map[string]struct{}) // dedup across multiple parents at this depth

		for _, parent := range frontier {
			neighbors := g.neighborsUnlocked(parent.id)
			for _, n := range neighbors {
				if _, already := visited[n]; already {
					continue
				}
				if _, queued := seenInLevel[n]; queued {
					continue
				}
				seenInLevel[n] = struct{}{}
				nextCandidates = append(nextCandidates, levelEntry{
					id:     n,
					degree: g.degreeUnlocked(n),
				})
			}
		}

		// Sort the candidates by degree DESC, breaking ties by ID ASC for
		// determinism.  Without the tie-break the test output would
		// flip-flop on Go's randomized map iteration.
		sort.Slice(nextCandidates, func(i, j int) bool {
			if nextCandidates[i].degree != nextCandidates[j].degree {
				return nextCandidates[i].degree > nextCandidates[j].degree
			}
			return nextCandidates[i].id < nextCandidates[j].id
		})

		// Admit candidates one at a time until the budget runs out.  Each
		// admitted candidate becomes part of the NEXT frontier so its
		// neighbors get expanded one depth deeper.
		nextFrontier := make([]levelEntry, 0, len(nextCandidates))
		for _, cand := range nextCandidates {
			if len(bfsOrder) >= maxNodes {
				break
			}
			visited[cand.id] = struct{}{}
			bfsOrder = append(bfsOrder, cand.id)
			nextFrontier = append(nextFrontier, cand)
		}
		frontier = nextFrontier
	}

	// Materialize the result.  Nodes come out in BFS discovery order so a
	// human reader can trace the traversal; edges are included only when
	// BOTH endpoints survived the cap.
	result := Subgraph{
		Nodes: make([]Entity, 0, len(bfsOrder)),
		Edges: make([]Relationship, 0),
	}
	visitedSet := make(map[string]struct{}, len(bfsOrder))
	for _, id := range bfsOrder {
		e := g.nodes[id]
		// Defensive copy of SourceIDs so callers mutating the slice can't
		// corrupt the in-memory graph.
		result.Nodes = append(result.Nodes, Entity{
			Name:        e.Name,
			Type:        e.Type,
			Description: e.Description,
			SourceIDs:   append([]string(nil), e.SourceIDs...),
		})
		visitedSet[id] = struct{}{}
	}

	// Walk the canonical edge table once and emit every edge whose A and B
	// are both in visitedSet.  This is O(|E|) — bigger than walking
	// adjacency on visited but simpler and we only do it once per query.
	// Upstream uses NetworkX's `subgraph(...).edges()` which is the same
	// shape: a filtered view of the full edge list.
	edgeKeys := make([]edgeKey, 0, len(g.edges))
	for k := range g.edges {
		if _, okA := visitedSet[k.A]; !okA {
			continue
		}
		if _, okB := visitedSet[k.B]; !okB {
			continue
		}
		edgeKeys = append(edgeKeys, k)
	}
	// Stable edge order: A asc, B asc.  Same reasoning as the node
	// tie-break — without this, repeat calls return identical Subgraphs in
	// content but different slice orders, confusing tests and CLI output.
	sort.Slice(edgeKeys, func(i, j int) bool {
		if edgeKeys[i].A != edgeKeys[j].A {
			return edgeKeys[i].A < edgeKeys[j].A
		}
		return edgeKeys[i].B < edgeKeys[j].B
	})
	for _, k := range edgeKeys {
		r := g.edges[k]
		result.Edges = append(result.Edges, Relationship{
			SrcID:       r.SrcID,
			TgtID:       r.TgtID,
			Keywords:    r.Keywords,
			Description: r.Description,
			Weight:      r.Weight,
			SourceIDs:   append([]string(nil), r.SourceIDs...),
		})
	}
	return result, nil
}
