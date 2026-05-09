package main

import (
	"context"
	"flag"
	"fmt"
	"os"
)

// main is the s08 CLI demo: build a 6-node, 7-edge social-network-style
// graph, print every node and edge, then run GetSubgraph("Hub", depth=2,
// maxNodes=5) and print the trimmed result.
//
// The graph shape (Hub-and-spokes + one cross-link):
//
//	          Hub
//	    ┌──────┼──────┐
//	    │      │      │
//	    Alice  Bob    Carol
//	    │             │
//	    Dave         Eve  ── Carol-Eve cross-link
//	                  │
//	                 (Eve also links to Dave: Alice-Dave-Eve-Carol triangle path)
//
// Edges: Hub-Alice, Hub-Bob, Hub-Carol, Alice-Dave, Carol-Eve, Eve-Dave, Bob-Carol
// → 6 spokes/branches + 1 cross-link = 7 edges.
//
// Hub has degree 3 (Alice, Bob, Carol).  Carol has degree 3 (Hub, Bob, Eve).
// Bob has degree 2.  At depth-1 from Hub we visit {Alice, Bob, Carol}; with
// maxNodes=5 we can keep going one level deeper to grab 1 more — degree
// priority means Carol's neighbors win the slot before Alice's.
func main() {
	dir := flag.String("dir", "./lightrag-data", "where to put graph.json")
	flag.Parse()

	ctx := context.Background()
	if err := run(ctx, *dir); err != nil {
		fmt.Fprintf(os.Stderr, "s08 demo failed: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, dir string) error {
	g, err := NewAdjacencyGraph(dir)
	if err != nil {
		return fmt.Errorf("new graph: %w", err)
	}

	// Build a small social network.  Each entity has a Type so the printed
	// output looks more like real KG data than test fixtures.
	people := []Entity{
		{Name: "Hub", Type: "PERSON", Description: "central organizer of the group"},
		{Name: "Alice", Type: "PERSON", Description: "longtime friend of Hub"},
		{Name: "Bob", Type: "PERSON", Description: "Hub's college roommate"},
		{Name: "Carol", Type: "PERSON", Description: "Hub's coworker, Bob's running partner"},
		{Name: "Dave", Type: "PERSON", Description: "Alice's brother, Eve's neighbor"},
		{Name: "Eve", Type: "PERSON", Description: "Carol's sister, Dave's neighbor"},
	}
	for _, p := range people {
		if err := g.UpsertNode(ctx, p); err != nil {
			return fmt.Errorf("upsert %s: %w", p.Name, err)
		}
	}

	edges := []Relationship{
		{SrcID: "Hub", TgtID: "Alice", Keywords: "friend", Weight: 0.9},
		{SrcID: "Hub", TgtID: "Bob", Keywords: "roommate", Weight: 0.8},
		{SrcID: "Hub", TgtID: "Carol", Keywords: "coworker", Weight: 0.7},
		{SrcID: "Alice", TgtID: "Dave", Keywords: "sibling", Weight: 1.0},
		{SrcID: "Carol", TgtID: "Eve", Keywords: "sibling", Weight: 1.0},
		{SrcID: "Eve", TgtID: "Dave", Keywords: "neighbor", Weight: 0.6},
		{SrcID: "Bob", TgtID: "Carol", Keywords: "running-partner", Weight: 0.7},
	}
	for _, e := range edges {
		if err := g.UpsertEdge(ctx, e); err != nil {
			return fmt.Errorf("upsert edge %s-%s: %w", e.SrcID, e.TgtID, err)
		}
	}

	fmt.Printf("=== full graph: %d nodes, %d edges ===\n", g.NodeCount(), g.EdgeCount())
	for _, p := range people {
		entity, ok, err := g.GetNode(ctx, p.Name)
		if err != nil {
			return fmt.Errorf("get %s: %w", p.Name, err)
		}
		if !ok {
			continue
		}
		fmt.Printf("  node  %-7s  type=%s  %q\n", entity.Name, entity.Type, entity.Description)
	}
	for _, e := range edges {
		fmt.Printf("  edge  %-6s -- %-6s  keywords=%-15q weight=%.2f\n",
			e.SrcID, e.TgtID, e.Keywords, e.Weight)
	}

	// Subgraph from Hub, depth=2, maxNodes=5.  Expected: Hub + 3 direct
	// neighbors {Alice, Bob, Carol} at depth 1 (4 so far) + 1 more at
	// depth 2 (degree priority picks Eve from Carol's neighbors over Dave
	// from Alice's neighbors — Eve has degree 2, Dave has degree 2 too,
	// but tie-break by ID asc means Dave wins... let's see at runtime).
	fmt.Printf("\n=== subgraph: GetSubgraph(\"Hub\", depth=2, maxNodes=5) ===\n")
	sg, err := g.GetSubgraph(ctx, "Hub", 2, 5)
	if err != nil {
		return fmt.Errorf("subgraph: %w", err)
	}
	fmt.Printf("got %d nodes, %d edges\n", len(sg.Nodes), len(sg.Edges))
	for _, n := range sg.Nodes {
		fmt.Printf("  node  %s\n", n.Name)
	}
	for _, e := range sg.Edges {
		fmt.Printf("  edge  %s -- %s\n", e.SrcID, e.TgtID)
	}

	if err := g.Persist(ctx); err != nil {
		return fmt.Errorf("persist: %w", err)
	}
	fmt.Printf("\npersisted -> %s/graph.json\n", dir)
	fmt.Println("Re-run with the same -dir to confirm Load round-trips.")
	return nil
}
