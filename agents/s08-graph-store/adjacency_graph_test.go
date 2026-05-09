package main

import (
	"context"
	"sort"
	"testing"
)

// TestGraphStoreInterfaceContract is a compile-time guard that
// *AdjacencyGraph satisfies the GraphStore interface.  No runtime work — if
// this stops compiling we have a method-set drift.
func TestGraphStoreInterfaceContract(t *testing.T) {
	var _ GraphStore = (*AdjacencyGraph)(nil)
}

// makeGraph creates a fresh AdjacencyGraph rooted under t.TempDir().
// Centralized so test bodies stay focused on assertions.
func makeGraph(t *testing.T) *AdjacencyGraph {
	t.Helper()
	g, err := NewAdjacencyGraph(t.TempDir())
	if err != nil {
		t.Fatalf("NewAdjacencyGraph: %v", err)
	}
	return g
}

// nodeNames returns the names of every node in the Subgraph in BFS-discovery
// order so callers can write order-sensitive assertions.  Used by
// TestSubgraphPrioritizesHighDegree which checks that B (high degree) is
// admitted before A or C.
func nodeNames(sg Subgraph) []string {
	out := make([]string, 0, len(sg.Nodes))
	for _, n := range sg.Nodes {
		out = append(out, n.Name)
	}
	return out
}

// containsAll asserts that `got` (BFS result) contains every name in `want`,
// regardless of order.  Used for tests where the BFS order isn't load-bearing
// — only set membership matters.
func containsAll(t *testing.T, got, want []string) {
	t.Helper()
	gotSet := make(map[string]struct{}, len(got))
	for _, n := range got {
		gotSet[n] = struct{}{}
	}
	missing := []string{}
	for _, n := range want {
		if _, ok := gotSet[n]; !ok {
			missing = append(missing, n)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		t.Errorf("subgraph missing expected nodes %v (got=%v)", missing, got)
	}
}

// containsNone asserts that `got` (BFS result) does NOT contain any name in
// `forbidden`.  Used for depth/maxNodes tests where the assertion is "X
// should be EXCLUDED".
func containsNone(t *testing.T, got, forbidden []string) {
	t.Helper()
	gotSet := make(map[string]struct{}, len(got))
	for _, n := range got {
		gotSet[n] = struct{}{}
	}
	leaked := []string{}
	for _, n := range forbidden {
		if _, ok := gotSet[n]; ok {
			leaked = append(leaked, n)
		}
	}
	if len(leaked) > 0 {
		sort.Strings(leaked)
		t.Errorf("subgraph contains forbidden nodes %v (got=%v)", leaked, got)
	}
}

// TestUpsertNodeIdempotent — UpsertNode("X", ...) twice; GetNode returns
// the second version (not duplicated).  Asserts the upsert semantic by
// comparing Description fields and NodeCount.
func TestUpsertNodeIdempotent(t *testing.T) {
	ctx := context.Background()
	g := makeGraph(t)

	if err := g.UpsertNode(ctx, Entity{Name: "X", Type: "ORG", Description: "first"}); err != nil {
		t.Fatalf("UpsertNode 1: %v", err)
	}
	if err := g.UpsertNode(ctx, Entity{Name: "X", Type: "ORG", Description: "second"}); err != nil {
		t.Fatalf("UpsertNode 2: %v", err)
	}

	if got := g.NodeCount(); got != 1 {
		t.Errorf("NodeCount after two upserts of same id = %d, want 1", got)
	}
	got, ok, err := g.GetNode(ctx, "X")
	if err != nil {
		t.Fatalf("GetNode: %v", err)
	}
	if !ok {
		t.Fatalf("GetNode(X) reports missing")
	}
	if got.Description != "second" {
		t.Errorf("Description after re-upsert = %q, want %q (overwrite, not duplicate)",
			got.Description, "second")
	}
}

// TestUpsertEdgeUndirected — UpsertEdge(a→b) then GetSubgraph(b, 1, 10);
// `a` appears as a neighbor.  Verifies that edges are bidirectional at the
// adjacency layer, not directed pointer-style.
func TestUpsertEdgeUndirected(t *testing.T) {
	ctx := context.Background()
	g := makeGraph(t)

	if err := g.UpsertNode(ctx, Entity{Name: "a"}); err != nil {
		t.Fatalf("upsert a: %v", err)
	}
	if err := g.UpsertNode(ctx, Entity{Name: "b"}); err != nil {
		t.Fatalf("upsert b: %v", err)
	}
	if err := g.UpsertEdge(ctx, Relationship{SrcID: "a", TgtID: "b", Keywords: "k"}); err != nil {
		t.Fatalf("upsert edge: %v", err)
	}

	// Query from b — `a` must appear in the subgraph, proving the edge is
	// reachable in BOTH directions.
	sg, err := g.GetSubgraph(ctx, "b", 1, 10)
	if err != nil {
		t.Fatalf("GetSubgraph(b): %v", err)
	}
	containsAll(t, nodeNames(sg), []string{"a", "b"})

	// And the symmetric: from a, `b` must appear.
	sg2, err := g.GetSubgraph(ctx, "a", 1, 10)
	if err != nil {
		t.Fatalf("GetSubgraph(a): %v", err)
	}
	containsAll(t, nodeNames(sg2), []string{"a", "b"})

	// EdgeCount must be 1 (NOT 2 — undirected collapses both insertions
	// onto the same canonical edgeKey).
	if got := g.EdgeCount(); got != 1 {
		t.Errorf("EdgeCount = %d, want 1 (undirected should not duplicate)", got)
	}
}

// TestSubgraphBFSDepthLimit — chain A-B-C-D-E.  GetSubgraph(A, depth=2,
// maxNodes=10) returns {A, B, C} only — D and E lie at depth 3 and 4.
func TestSubgraphBFSDepthLimit(t *testing.T) {
	ctx := context.Background()
	g := makeGraph(t)

	for _, n := range []string{"A", "B", "C", "D", "E"} {
		if err := g.UpsertNode(ctx, Entity{Name: n}); err != nil {
			t.Fatalf("upsert %s: %v", n, err)
		}
	}
	chain := [][2]string{{"A", "B"}, {"B", "C"}, {"C", "D"}, {"D", "E"}}
	for _, p := range chain {
		if err := g.UpsertEdge(ctx, Relationship{SrcID: p[0], TgtID: p[1]}); err != nil {
			t.Fatalf("edge %s-%s: %v", p[0], p[1], err)
		}
	}

	sg, err := g.GetSubgraph(ctx, "A", 2, 10)
	if err != nil {
		t.Fatalf("GetSubgraph: %v", err)
	}
	got := nodeNames(sg)
	containsAll(t, got, []string{"A", "B", "C"})
	containsNone(t, got, []string{"D", "E"})
	if len(got) != 3 {
		t.Errorf("len(nodes) = %d, want 3 (got=%v)", len(got), got)
	}
}

// TestSubgraphRespectsMaxNodes — hub with 10 spokes; GetSubgraph(hub,
// depth=1, maxNodes=4) returns hub + 3 spokes (4 nodes total).
func TestSubgraphRespectsMaxNodes(t *testing.T) {
	ctx := context.Background()
	g := makeGraph(t)

	if err := g.UpsertNode(ctx, Entity{Name: "hub"}); err != nil {
		t.Fatalf("upsert hub: %v", err)
	}
	for i := 0; i < 10; i++ {
		spoke := spokeID(i)
		if err := g.UpsertNode(ctx, Entity{Name: spoke}); err != nil {
			t.Fatalf("upsert %s: %v", spoke, err)
		}
		if err := g.UpsertEdge(ctx, Relationship{SrcID: "hub", TgtID: spoke}); err != nil {
			t.Fatalf("edge hub-%s: %v", spoke, err)
		}
	}

	sg, err := g.GetSubgraph(ctx, "hub", 1, 4)
	if err != nil {
		t.Fatalf("GetSubgraph: %v", err)
	}
	got := nodeNames(sg)
	if len(got) != 4 {
		t.Errorf("len(nodes) = %d, want 4 (hub + 3 spokes); got=%v", len(got), got)
	}
	containsAll(t, got, []string{"hub"})
	// Edges: only those between admitted nodes.  Each admitted spoke has
	// exactly one edge (to hub), so we expect 3 edges total.
	if len(sg.Edges) != 3 {
		t.Errorf("len(edges) = %d, want 3 (one per admitted spoke); got %v", len(sg.Edges), sg.Edges)
	}
}

func spokeID(i int) string {
	// 'a'..'j' so spokes have a 1-character ID, easy to read in failures.
	return string(rune('a' + i))
}

// TestSubgraphPrioritizesHighDegree — hub-A (deg 1), hub-B (deg 5 because B
// has its own 4 satellite neighbors), hub-C (deg 1).  GetSubgraph(hub,
// depth=1, maxNodes=2) returns {hub, B} — the high-degree neighbor wins
// the only available slot.
func TestSubgraphPrioritizesHighDegree(t *testing.T) {
	ctx := context.Background()
	g := makeGraph(t)

	for _, n := range []string{"hub", "A", "B", "C", "B1", "B2", "B3", "B4"} {
		if err := g.UpsertNode(ctx, Entity{Name: n}); err != nil {
			t.Fatalf("upsert %s: %v", n, err)
		}
	}
	// hub spokes:
	for _, n := range []string{"A", "B", "C"} {
		if err := g.UpsertEdge(ctx, Relationship{SrcID: "hub", TgtID: n}); err != nil {
			t.Fatalf("edge hub-%s: %v", n, err)
		}
	}
	// B's 4 satellite neighbors → deg(B) = 5 (hub + B1..B4).
	for _, n := range []string{"B1", "B2", "B3", "B4"} {
		if err := g.UpsertEdge(ctx, Relationship{SrcID: "B", TgtID: n}); err != nil {
			t.Fatalf("edge B-%s: %v", n, err)
		}
	}

	sg, err := g.GetSubgraph(ctx, "hub", 1, 2)
	if err != nil {
		t.Fatalf("GetSubgraph: %v", err)
	}
	got := nodeNames(sg)
	if len(got) != 2 {
		t.Fatalf("len(nodes) = %d, want 2 (hub + 1 high-deg neighbor); got=%v", len(got), got)
	}
	containsAll(t, got, []string{"hub", "B"})
	containsNone(t, got, []string{"A", "C"})
}

// TestGraphPersistAcrossRestart — build 4-node graph, Persist, construct a
// new instance with the SAME dir, GetSubgraph returns the same nodes/edges.
func TestGraphPersistAcrossRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	// Phase 1: fill + persist.
	g1, err := NewAdjacencyGraph(dir)
	if err != nil {
		t.Fatalf("NewAdjacencyGraph (1): %v", err)
	}
	for _, n := range []string{"P", "Q", "R", "S"} {
		if err := g1.UpsertNode(ctx, Entity{Name: n, Type: "PERSON", Description: "lives in " + n}); err != nil {
			t.Fatalf("upsert %s: %v", n, err)
		}
	}
	for _, p := range [][2]string{{"P", "Q"}, {"Q", "R"}, {"R", "S"}, {"P", "R"}} {
		if err := g1.UpsertEdge(ctx, Relationship{
			SrcID: p[0], TgtID: p[1], Keywords: "test", Weight: 0.42,
		}); err != nil {
			t.Fatalf("edge %s-%s: %v", p[0], p[1], err)
		}
	}
	if err := g1.Persist(ctx); err != nil {
		t.Fatalf("Persist: %v", err)
	}
	sg1, err := g1.GetSubgraph(ctx, "P", 3, 100)
	if err != nil {
		t.Fatalf("GetSubgraph (1): %v", err)
	}

	// Phase 2: fresh instance, same path.
	g2, err := NewAdjacencyGraph(dir)
	if err != nil {
		t.Fatalf("NewAdjacencyGraph (2): %v", err)
	}
	if got, want := g2.NodeCount(), 4; got != want {
		t.Errorf("after reload NodeCount = %d, want %d", got, want)
	}
	if got, want := g2.EdgeCount(), 4; got != want {
		t.Errorf("after reload EdgeCount = %d, want %d", got, want)
	}

	sg2, err := g2.GetSubgraph(ctx, "P", 3, 100)
	if err != nil {
		t.Fatalf("GetSubgraph (2): %v", err)
	}
	if len(sg1.Nodes) != len(sg2.Nodes) {
		t.Fatalf("subgraph node-count drift: pre=%d post=%d", len(sg1.Nodes), len(sg2.Nodes))
	}
	if len(sg1.Edges) != len(sg2.Edges) {
		t.Fatalf("subgraph edge-count drift: pre=%d post=%d", len(sg1.Edges), len(sg2.Edges))
	}
	// Spot-check a node's persisted attributes survived round-trip.
	r, ok, err := g2.GetNode(ctx, "R")
	if err != nil || !ok {
		t.Fatalf("after reload GetNode(R): ok=%v err=%v", ok, err)
	}
	if r.Type != "PERSON" {
		t.Errorf("R.Type after reload = %q, want %q", r.Type, "PERSON")
	}
	if r.Description != "lives in R" {
		t.Errorf("R.Description after reload = %q, want %q", r.Description, "lives in R")
	}
}
