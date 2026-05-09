package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// AdjacencyGraph is the s08 reference impl of GraphStore: an in-memory
// undirected graph backed by three coordinated maps + a coarse RWMutex.
//
// Why three maps and not one big nested struct?
//   - `nodes` is the entity attribute table — Name → Entity.  Lookup by ID
//     is the dominant access pattern (Get, neighbor-degree sort, edge sanity
//     checks) so we keep nodes flat in O(1)-access shape.
//   - `edges` is the relationship attribute table keyed by canonicalized
//     `edgeKey{A, B}` (always A < B lexicographically) so `(scrooge, marley)`
//     and `(marley, scrooge)` collapse onto the same row — this is the move
//     that makes the graph undirected at the storage layer.
//   - `adj` is the adjacency set: `node-id → {neighbor-id, ...}` represented
//     as a map of empty-struct sets.  It's redundant with `edges` (could be
//     re-derived) but caching it cuts the BFS cost from O(|E|) per neighbor
//     lookup to O(deg(v)).  Mirror of NetworkX's compact adjacency dict.
//
// Locking: a single sync.RWMutex coarse-locks ALL three maps together.  This
// matches upstream's per-namespace asyncio lock — multiple concurrent reads
// (Query, GetSubgraph, Persist snapshot) are fine; any mutation takes Lock.
type AdjacencyGraph struct {
	dir string

	mu    sync.RWMutex
	nodes map[string]Entity
	edges map[edgeKey]Relationship
	adj   map[string]map[string]struct{}
}

// edgeKey is the canonicalized undirected-edge identity.  A is always the
// lexicographically smaller endpoint, B the larger.  newEdgeKey enforces the
// invariant; downstream code can compare edgeKey values directly without
// re-checking which side is "source" vs "target".
type edgeKey struct {
	A, B string
}

// newEdgeKey builds a canonical edgeKey from any two endpoint IDs by sorting
// them lexicographically.  Empty endpoints are rejected to surface the bug
// at the boundary instead of letting "" silently merge into one side.
func newEdgeKey(src, tgt string) (edgeKey, error) {
	if src == "" || tgt == "" {
		return edgeKey{}, fmt.Errorf("graph-store: edge endpoints must be non-empty (src=%q tgt=%q)", src, tgt)
	}
	if src == tgt {
		// Self-loops would force both adj[a] and adj[b] to point at `a`
		// itself, which BFS does NOT handle (would visit `a` twice in one
		// hop).  Reject up-front rather than carry the special case
		// through the rest of the impl.
		return edgeKey{}, fmt.Errorf("graph-store: self-loops are not supported (id=%q)", src)
	}
	if src < tgt {
		return edgeKey{A: src, B: tgt}, nil
	}
	return edgeKey{A: tgt, B: src}, nil
}

// graphFileName mirrors upstream's `graph_<namespace>.graphml` naming, but
// lives in JSON instead of GraphML so we can write it with stdlib only.
// Centralized so Persist and load can never disagree on the path.
const graphFileName = "graph.json"

// ErrEmptyID is returned when callers forget to set Entity.Name.  Surfacing
// it at the boundary beats silently writing an entry under "" that would
// shadow every subsequent empty-name lookup.
var ErrEmptyID = errors.New("graph-store: Entity.Name (or Relationship endpoint) must be non-empty")

// NewAdjacencyGraph constructs a graph rooted at `dir`.  If
// `<dir>/graph.json` already exists, its contents are loaded synchronously;
// missing file means we start fresh and is NOT an error (matches upstream's
// "create new empty graph" path in __post_init__).
func NewAdjacencyGraph(dir string) (*AdjacencyGraph, error) {
	if dir == "" {
		dir = "./lightrag-data"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("graph-store: mkdir %s: %w", dir, err)
	}
	g := &AdjacencyGraph{
		dir:   dir,
		nodes: make(map[string]Entity),
		edges: make(map[edgeKey]Relationship),
		adj:   make(map[string]map[string]struct{}),
	}
	if err := g.loadFromDisk(); err != nil {
		return nil, err
	}
	return g, nil
}

// filePath returns the canonical on-disk path.  Always go through this —
// never build the path from string concatenation at call sites.
func (g *AdjacencyGraph) filePath() string {
	return filepath.Join(g.dir, graphFileName)
}

// NodeCount reports the in-memory node count.  Useful for the CLI demo and
// quick test invariants without iterating the graph.
func (g *AdjacencyGraph) NodeCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodes)
}

// EdgeCount reports the in-memory edge count.  An edge inserted twice (a→b
// then b→a) counts ONCE because edgeKey collapses the two onto one row.
func (g *AdjacencyGraph) EdgeCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.edges)
}

// degreeUnlocked returns the degree of node id WITHOUT acquiring the mutex.
// Caller MUST hold at least an RLock.  Used by the BFS to sort neighbors at
// each level by degree — we want to avoid re-acquiring the lock per neighbor
// in the inner loop.
func (g *AdjacencyGraph) degreeUnlocked(id string) int {
	return len(g.adj[id])
}

// neighborsUnlocked returns the sorted neighbor IDs of `id`.  Sorted is
// load-bearing: BFS uses this order to break ties when two neighbors have
// the same degree, so test output is deterministic across map-iteration
// reorderings.  Caller MUST hold at least an RLock.
func (g *AdjacencyGraph) neighborsUnlocked(id string) []string {
	set := g.adj[id]
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// UpsertNode writes or replaces an Entity by Name.  Idempotent: calling
// twice with the same Name overwrites the existing record (no duplication).
// Description / Type / SourceIDs from the new Entity REPLACE the old; we
// don't merge — upstream doesn't either, the merge step lives one layer up
// in `_merge_nodes_then_upsert` (s10).
func (g *AdjacencyGraph) UpsertNode(ctx context.Context, e Entity) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if e.Name == "" {
		return ErrEmptyID
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.nodes[e.Name] = Entity{
		Name:        e.Name,
		Type:        e.Type,
		Description: e.Description,
		SourceIDs:   append([]string(nil), e.SourceIDs...), // defensive copy
	}
	// Make sure the adj entry exists even for an isolated node (no edges
	// yet).  Without this, a brand-new node has no map entry and the BFS
	// would treat it as "missing" instead of "isolated".
	if _, ok := g.adj[e.Name]; !ok {
		g.adj[e.Name] = make(map[string]struct{})
	}
	return nil
}

// UpsertEdge writes or replaces an undirected edge identified by the
// canonical (min, max) endpoint pair.  Both endpoint nodes are auto-created
// (with empty Entity records) if they don't already exist — this matches
// upstream NetworkX's `add_edge` semantic and keeps the call sites simple.
//
// Per-edge attributes (Keywords / Description / Weight / SourceIDs) replace
// the existing values; the merge logic lives in s10's `_merge_edges_then_upsert`.
func (g *AdjacencyGraph) UpsertEdge(ctx context.Context, r Relationship) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	key, err := newEdgeKey(r.SrcID, r.TgtID)
	if err != nil {
		return err
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	// Auto-create endpoint nodes if missing.  Upsert path matches NetworkX
	// add_edge — call sites stay clean of "ensure-node-then-add-edge" boilerplate.
	for _, id := range []string{r.SrcID, r.TgtID} {
		if _, ok := g.nodes[id]; !ok {
			g.nodes[id] = Entity{Name: id}
		}
		if _, ok := g.adj[id]; !ok {
			g.adj[id] = make(map[string]struct{})
		}
	}

	// Adjacency: each endpoint gets the OTHER as a neighbor.  This is what
	// "undirected" means at the storage layer — both directions resolve to
	// the same edgeKey above and the same neighbor lookup here.
	g.adj[r.SrcID][r.TgtID] = struct{}{}
	g.adj[r.TgtID][r.SrcID] = struct{}{}

	// Store the canonical Relationship under the canonical key.  We
	// preserve the caller's SrcID/TgtID order in the stored value — the
	// canonical (A, B) lives only in the map key, not in the value, so the
	// caller can still tell which side they originally inserted from.
	g.edges[key] = Relationship{
		SrcID:       r.SrcID,
		TgtID:       r.TgtID,
		Keywords:    r.Keywords,
		Description: r.Description,
		Weight:      r.Weight,
		SourceIDs:   append([]string(nil), r.SourceIDs...),
	}
	return nil
}

// GetNode returns the Entity by ID.  The (zero, false, nil) tuple for a
// missing ID is the established Go pattern (see context.Value, sync.Map.Load)
// and lets callers distinguish "not present" from a real error.
func (g *AdjacencyGraph) GetNode(ctx context.Context, id string) (Entity, bool, error) {
	if err := ctx.Err(); err != nil {
		return Entity{}, false, err
	}
	if id == "" {
		return Entity{}, false, ErrEmptyID
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	e, ok := g.nodes[id]
	if !ok {
		return Entity{}, false, nil
	}
	// Defensive copy of SourceIDs so callers mutating the slice can't
	// poison the in-memory state.
	out := Entity{
		Name:        e.Name,
		Type:        e.Type,
		Description: e.Description,
		SourceIDs:   append([]string(nil), e.SourceIDs...),
	}
	return out, true, nil
}

// GetSubgraph runs the BFS from subgraph_bfs.go.  Kept as a thin wrapper so
// the interface contract lives next to the rest of the GraphStore methods.
func (g *AdjacencyGraph) GetSubgraph(ctx context.Context, seed string, maxDepth, maxNodes int) (Subgraph, error) {
	if err := ctx.Err(); err != nil {
		return Subgraph{}, err
	}
	return getSubgraphBFS(g, seed, maxDepth, maxNodes)
}

// graphFile is the on-disk JSON envelope.  Mirrors the shape NetworkX writes
// to GraphML (a `nodes` table + an `edges` table) but uses plain JSON instead
// of XML so the file is greppable and stdlib can write it.  We deliberately
// store edges with their canonical (A, B) endpoints + the original SrcID/TgtID
// so a load doesn't lose the caller's insertion direction.
type graphFile struct {
	Version int                  `json:"version"`
	Nodes   []nodeOnDisk         `json:"nodes"`
	Edges   []relationshipOnDisk `json:"edges"`
}

type nodeOnDisk struct {
	Name        string   `json:"name"`
	Type        string   `json:"type,omitempty"`
	Description string   `json:"description,omitempty"`
	SourceIDs   []string `json:"source_ids,omitempty"`
}

type relationshipOnDisk struct {
	SrcID       string   `json:"src_id"`
	TgtID       string   `json:"tgt_id"`
	Keywords    string   `json:"keywords,omitempty"`
	Description string   `json:"description,omitempty"`
	Weight      float32  `json:"weight,omitempty"`
	SourceIDs   []string `json:"source_ids,omitempty"`
}

// loadFromDisk reads `<dir>/graph.json` and replaces the in-memory state.
// A missing file is benign — empty graph.  A malformed file is fatal: better
// to fail at construction time than silently start fresh and drop user data.
func (g *AdjacencyGraph) loadFromDisk() error {
	buf, err := os.ReadFile(g.filePath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("graph-store: read %s: %w", g.filePath(), err)
	}
	if len(buf) == 0 {
		return nil
	}
	var f graphFile
	if err := json.Unmarshal(buf, &f); err != nil {
		return fmt.Errorf("graph-store: unmarshal %s: %w", g.filePath(), err)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	g.nodes = make(map[string]Entity, len(f.Nodes))
	g.edges = make(map[edgeKey]Relationship, len(f.Edges))
	g.adj = make(map[string]map[string]struct{}, len(f.Nodes))
	for _, n := range f.Nodes {
		if n.Name == "" {
			continue // skip malformed rows defensively
		}
		g.nodes[n.Name] = Entity{
			Name:        n.Name,
			Type:        n.Type,
			Description: n.Description,
			SourceIDs:   append([]string(nil), n.SourceIDs...),
		}
		g.adj[n.Name] = make(map[string]struct{})
	}
	for _, r := range f.Edges {
		key, err := newEdgeKey(r.SrcID, r.TgtID)
		if err != nil {
			continue
		}
		// Auto-create endpoints if a malformed file references nodes the
		// nodes-table didn't declare; better to recover than fail hard.
		for _, id := range []string{r.SrcID, r.TgtID} {
			if _, ok := g.nodes[id]; !ok {
				g.nodes[id] = Entity{Name: id}
			}
			if _, ok := g.adj[id]; !ok {
				g.adj[id] = make(map[string]struct{})
			}
		}
		g.adj[r.SrcID][r.TgtID] = struct{}{}
		g.adj[r.TgtID][r.SrcID] = struct{}{}
		g.edges[key] = Relationship{
			SrcID:       r.SrcID,
			TgtID:       r.TgtID,
			Keywords:    r.Keywords,
			Description: r.Description,
			Weight:      r.Weight,
			SourceIDs:   append([]string(nil), r.SourceIDs...),
		}
	}
	return nil
}

// Persist atomically writes the graph to disk via the same write-tmp +
// rename dance s05's KV store and s07's vector store use.  Snapshot is taken
// under RLock, file write happens after the lock is released so a slow disk
// can't starve concurrent readers.
func (g *AdjacencyGraph) Persist(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	g.mu.RLock()
	envelope := graphFile{
		Version: 1,
		Nodes:   make([]nodeOnDisk, 0, len(g.nodes)),
		Edges:   make([]relationshipOnDisk, 0, len(g.edges)),
	}
	// Sort node IDs and edge keys before writing so the file is byte-stable
	// across runs — makes diff-able / reviewable like the s05/s07 snapshots.
	nodeIDs := make([]string, 0, len(g.nodes))
	for id := range g.nodes {
		nodeIDs = append(nodeIDs, id)
	}
	sort.Strings(nodeIDs)
	for _, id := range nodeIDs {
		n := g.nodes[id]
		envelope.Nodes = append(envelope.Nodes, nodeOnDisk{
			Name:        n.Name,
			Type:        n.Type,
			Description: n.Description,
			SourceIDs:   append([]string(nil), n.SourceIDs...),
		})
	}
	edgeKeys := make([]edgeKey, 0, len(g.edges))
	for k := range g.edges {
		edgeKeys = append(edgeKeys, k)
	}
	sort.Slice(edgeKeys, func(i, j int) bool {
		if edgeKeys[i].A != edgeKeys[j].A {
			return edgeKeys[i].A < edgeKeys[j].A
		}
		return edgeKeys[i].B < edgeKeys[j].B
	})
	for _, k := range edgeKeys {
		r := g.edges[k]
		envelope.Edges = append(envelope.Edges, relationshipOnDisk{
			SrcID:       r.SrcID,
			TgtID:       r.TgtID,
			Keywords:    r.Keywords,
			Description: r.Description,
			Weight:      r.Weight,
			SourceIDs:   append([]string(nil), r.SourceIDs...),
		})
	}
	g.mu.RUnlock()

	if err := os.MkdirAll(g.dir, 0o755); err != nil {
		return fmt.Errorf("graph-store: mkdir %s: %w", g.dir, err)
	}
	buf, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		return fmt.Errorf("graph-store: marshal: %w", err)
	}
	canonical := g.filePath()
	tmp := canonical + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return fmt.Errorf("graph-store: write tmp %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, canonical); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("graph-store: rename %s -> %s: %w", tmp, canonical, err)
	}
	return nil
}
