package main

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
)

// pipeline.go is the s11 spine.  Pipeline wires together the four storages
// and two providers, then exposes Query(ctx, q, param) which dispatches to
// one of four mode functions in query_naive.go / query_local.go /
// query_global.go / query_hybrid.go.  The mode functions compile a context
// window and call Provider.Complete; this file only owns the dispatch.
//
// The in-memory stub stores below (memVectorStore, memKVStore, memGraphStore)
// are minimal re-implementations of s05/s07/s08 so the demo and the seven
// tests run without any cross-session import.  IsolationContract trumps DRY.

// Pipeline holds every dependency the query path needs.  Construct once,
// reuse across queries — the stores keep their own internal state.
type Pipeline struct {
	Provider  Provider
	Embedder  EmbeddingProvider
	VDBChunks VectorStore
	VDBEnts   VectorStore
	VDBRels   VectorStore
	KV        KVStore
	Graph     GraphStore
}

// Query is the public entry point.  It validates QueryParam, dispatches to
// the mode-specific function, and returns the QueryResult.  Defaults match
// what upstream hard-codes when params are zero (see operate.py:3251).
func (p *Pipeline) Query(ctx context.Context, q string, param QueryParam) (QueryResult, error) {
	if p == nil {
		return QueryResult{}, fmt.Errorf("s11: nil Pipeline")
	}
	if q == "" {
		return QueryResult{}, fmt.Errorf("s11: empty query")
	}
	param = applyDefaults(param)

	switch param.Mode {
	case ModeNaive:
		return queryNaive(ctx, p, q, param)
	case ModeLocal:
		return queryLocal(ctx, p, q, param)
	case ModeGlobal:
		return queryGlobal(ctx, p, q, param)
	case ModeHybrid:
		return queryHybrid(ctx, p, q, param)
	default:
		return QueryResult{}, fmt.Errorf("s11: unknown mode %q (want naive / local / global / hybrid)", param.Mode)
	}
}

// applyDefaults fills in zero-valued QueryParam fields with sensible defaults
// matching upstream (operate.py:3251 + base.py:77 QueryParam).
func applyDefaults(p QueryParam) QueryParam {
	if p.Mode == "" {
		p.Mode = ModeHybrid
	}
	if p.TopK <= 0 {
		p.TopK = 10
	}
	if p.ChunkTopK <= 0 {
		p.ChunkTopK = 5
	}
	if p.MaxEntityTokens <= 0 {
		p.MaxEntityTokens = 4000
	}
	if p.MaxRelationTokens <= 0 {
		p.MaxRelationTokens = 4000
	}
	if p.MaxTotalTokens <= 0 {
		p.MaxTotalTokens = 8000
	}
	return p
}

// =============================================================================
// In-memory storage stubs  (minimal local copies of s05 / s07 / s08)
// =============================================================================
//
// These are deliberately small.  They share the same interfaces declared in
// types.go but cut out persistence, locking, and JSON marshalling.  Under
// 100 LOC total — enough for the demo and tests, not a production substitute.

// --- memKVStore --------------------------------------------------------------

type memKVStore struct {
	mu   sync.RWMutex
	data map[string]map[string]any
}

// newMemKV returns an empty in-memory KVStore.
func newMemKV() *memKVStore {
	return &memKVStore{data: map[string]map[string]any{}}
}

func (m *memKVStore) Get(ctx context.Context, id string) (map[string]any, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.data[id]
	return v, ok, nil
}
func (m *memKVStore) GetByIDs(ctx context.Context, ids []string) (map[string]map[string]any, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]map[string]any, len(ids))
	for _, id := range ids {
		if v, ok := m.data[id]; ok {
			out[id] = v
		}
	}
	return out, nil
}
func (m *memKVStore) Upsert(ctx context.Context, items map[string]map[string]any) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for k, v := range items {
		m.data[k] = v
	}
	return nil
}
func (m *memKVStore) FilterMissing(ctx context.Context, ids []string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	missing := make([]string, 0, len(ids))
	for _, id := range ids {
		if _, ok := m.data[id]; !ok {
			missing = append(missing, id)
		}
	}
	return missing, nil
}
func (m *memKVStore) Delete(ctx context.Context, ids []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, id := range ids {
		delete(m.data, id)
	}
	return nil
}
func (m *memKVStore) Persist(ctx context.Context) error { return nil }

// --- memVectorStore ----------------------------------------------------------

type memVectorStore struct {
	mu      sync.RWMutex
	records []VectorRecord
}

// newMemVec returns an empty cosine-similarity vector store.
func newMemVec() *memVectorStore { return &memVectorStore{} }

func (v *memVectorStore) Upsert(ctx context.Context, records []VectorRecord) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	// Replace by ID to keep the index unique-keyed.
	idx := map[string]int{}
	for i, r := range v.records {
		idx[r.ID] = i
	}
	for _, r := range records {
		if pos, ok := idx[r.ID]; ok {
			v.records[pos] = r
			continue
		}
		v.records = append(v.records, r)
	}
	return nil
}

func (v *memVectorStore) Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	hits := make([]VectorHit, 0, len(v.records))
	for _, r := range v.records {
		s := cosine(query, r.Vector)
		if threshold > -1.0 && s < threshold {
			continue
		}
		hits = append(hits, VectorHit{ID: r.ID, Score: s, Metadata: r.Metadata})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if topK > 0 && len(hits) > topK {
		hits = hits[:topK]
	}
	return hits, nil
}

func (v *memVectorStore) Delete(ctx context.Context, ids []string) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	keep := v.records[:0]
	rm := map[string]struct{}{}
	for _, id := range ids {
		rm[id] = struct{}{}
	}
	for _, r := range v.records {
		if _, drop := rm[r.ID]; !drop {
			keep = append(keep, r)
		}
	}
	v.records = keep
	return nil
}
func (v *memVectorStore) Persist(ctx context.Context) error { return nil }

// cosine is the standard cosine similarity between two equal-length vectors.
// Returns 0 on zero-norm or length mismatch.
func cosine(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	for _, x := range a {
		na += float64(x) * float64(x)
	}
	for _, x := range b {
		nb += float64(x) * float64(x)
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// --- memGraphStore -----------------------------------------------------------

type memGraphStore struct {
	mu    sync.RWMutex
	nodes map[string]Entity
	edges []Relationship
}

func newMemGraph() *memGraphStore {
	return &memGraphStore{nodes: map[string]Entity{}}
}

func (g *memGraphStore) UpsertNode(ctx context.Context, e Entity) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.nodes[e.Name] = e
	return nil
}
func (g *memGraphStore) UpsertEdge(ctx context.Context, r Relationship) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	for i, existing := range g.edges {
		if existing.SrcID == r.SrcID && existing.TgtID == r.TgtID {
			g.edges[i] = r
			return nil
		}
	}
	g.edges = append(g.edges, r)
	return nil
}
func (g *memGraphStore) GetNode(ctx context.Context, id string) (Entity, bool, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	e, ok := g.nodes[id]
	return e, ok, nil
}

// GetSubgraph does a tiny BFS up to maxDepth, capping at maxNodes total
// nodes (including the seed).  Mirrors s08's behaviour but without the
// degree-priority frontier.
func (g *memGraphStore) GetSubgraph(ctx context.Context, seed string, maxDepth, maxNodes int) (Subgraph, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if maxDepth <= 0 {
		maxDepth = 1
	}
	if maxNodes <= 0 {
		maxNodes = 16
	}
	seen := map[string]struct{}{}
	out := Subgraph{}
	if e, ok := g.nodes[seed]; ok {
		seen[seed] = struct{}{}
		out.Nodes = append(out.Nodes, e)
	}
	frontier := []string{seed}
	for depth := 0; depth < maxDepth && len(out.Nodes) < maxNodes; depth++ {
		next := []string{}
		for _, n := range frontier {
			for _, edge := range g.edges {
				var other string
				switch n {
				case edge.SrcID:
					other = edge.TgtID
				case edge.TgtID:
					other = edge.SrcID
				default:
					continue
				}
				out.Edges = append(out.Edges, edge)
				if _, ok := seen[other]; ok {
					continue
				}
				if e, ok := g.nodes[other]; ok && len(out.Nodes) < maxNodes {
					seen[other] = struct{}{}
					out.Nodes = append(out.Nodes, e)
					next = append(next, other)
				}
			}
		}
		frontier = next
	}
	return out, nil
}
func (g *memGraphStore) Persist(ctx context.Context) error { return nil }

// --- compile-time interface assertions ---------------------------------------

var (
	_ KVStore     = (*memKVStore)(nil)
	_ VectorStore = (*memVectorStore)(nil)
	_ GraphStore  = (*memGraphStore)(nil)
)
