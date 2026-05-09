package main

import "context"

// Entity is the in-memory shape of a knowledge-graph node.  It mirrors
// upstream's per-node attribute dict (entity_type / description / source_id)
// in a named struct so a learner reads the field list rather than guessing
// at dict keys.  Set fields via UpsertNode; SourceIDs aggregates the chunk
// IDs that mention this entity (s09 will populate it during extraction).
type Entity struct {
	Name        string   // canonical key; doubles as the node ID in the adjacency map
	Type        string   // PERSON | ORG | EVENT | CONCEPT | ... — free-form for s08
	Description string   // merged description (s10 will rewrite this via LLM summary)
	SourceIDs   []string // chunk IDs that mention this entity
}

// Relationship is the in-memory shape of an undirected edge.  SrcID / TgtID
// are the two endpoint entity Names; the canonical edge key is computed from
// (min, max) so the graph treats `(a, b)` and `(b, a)` as the same edge.
// Weight, Keywords, and Description carry the per-edge attributes that s11
// uses when building a query context.
type Relationship struct {
	SrcID, TgtID string
	Keywords     string
	Description  string
	Weight       float32
	SourceIDs    []string
}

// Subgraph is the result of GetSubgraph: a slice of Nodes plus a slice of
// Edges that connect ONLY those nodes.  Mirrors upstream's KnowledgeGraph
// dataclass minus the is_truncated flag (truncation is implicit when
// len(Nodes) == maxNodes — callers can check that themselves).
//
// Both slices are returned non-nil even when empty; this matches the
// VectorStore contract from s07 and avoids a class of nil-deref bugs.
type Subgraph struct {
	Nodes []Entity
	Edges []Relationship
}

// GraphStore is the interface every backend implements.  Mirrors upstream
// `lightrag/base.py:333 BaseGraphStorage` minus the async-only sugar — every
// method takes a context.Context for cancellation.  s08 ships ONE concrete
// impl (`AdjacencyGraph`) but the seam is here so s09 / s11 can swap in a
// Neo4j or Memgraph backend later without churning the call sites.
//
// Contract notes:
//   - UpsertNode is idempotent: calling it twice with the same Entity.Name
//     replaces the existing node (no duplication).  Same applies to UpsertEdge.
//   - GetNode returns (zero, false, nil) when the ID is unknown — NOT an error.
//   - GetSubgraph traverses undirected edges; an edge inserted as (a → b) is
//     reachable from both a and b.  Seed nodes that don't exist yield an
//     empty Subgraph with nil error (matches upstream's "log warning" path).
//   - Persist atomic-renames the in-memory state to a JSON adjacency dump.
type GraphStore interface {
	UpsertNode(ctx context.Context, e Entity) error
	UpsertEdge(ctx context.Context, r Relationship) error
	GetNode(ctx context.Context, id string) (Entity, bool, error)
	GetSubgraph(ctx context.Context, seed string, maxDepth, maxNodes int) (Subgraph, error)
	Persist(ctx context.Context) error
}
