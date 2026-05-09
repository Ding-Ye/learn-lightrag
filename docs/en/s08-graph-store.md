---
title: "s08 · Adjacency graph store and subgraph BFS"
chapter: 8
slug: s08-graph-store
est_read_min: 10
---

# s08 · Adjacency graph store and subgraph BFS

> What this teaches: implement an in-memory undirected `AdjacencyGraph` — nodes in `map[string]Entity`, edges in a canonicalized `map[edgeKey]Relationship` (where `edgeKey` orders endpoints `(min, max)` so `(a, b)` and `(b, a)` collapse onto one row). `GetSubgraph(seed, maxDepth, maxNodes)` is BFS with **degree-based priority** — the most-connected neighbors at each level are visited first, mirroring upstream NetworkX's behavior. Atomic-rename JSON snapshot for persistence. Tracks `lightrag/kg/networkx_impl.py`.

---

## Problem / 问题

s07 left us with three independent vector indices (chunks / entities / relations), but the entities inside have no idea the others exist — they're just isolated bags of vectors. Ask "what's Scrooge's relationship with Marley" and vector retrieval can find chunks that mention each name separately, but it can't answer "they were business partners in the same story" because that fact is distributed across multiple chunks and requires **structured** connections to assemble.

This is LightRAG's central bet: **a real "GraphRAG" must have a graph, not just a pile of chunks**. Concretely:

1. **No nodes = no entity-centric retrieval.** s11's local mode wants to do "find everything related to this entity," which requires storing "this is an entity" as a structural record. s07's `vdbEntities` stores entity-description **vectors**, not entity **structure**.
2. **No edges = no relationship signal.** s11's global mode wants to do "find the relation network around this theme," which requires walking `(src, tgt, weight)` triples directly. Vector similarity alone over `vdbRelations` can't answer "how many hops between A and B."
3. **No subgraph = no context window.** The final LLM prompt has to splice in "entity X's neighbors + neighbors-of-neighbors" within a token budget. That's a depth-limited, importance-truncated BFS — not "top-k by similarity in one big bag."
4. **NetworkX isn't a Go library.** Upstream gets `nx.Graph()` for free — undirected graph + GraphML persistence + `nx.subgraph()` in three lines. We don't have that luxury and must grow it from `map[string]Entity` ourselves — fortunately the core algorithm is ~150 lines.

s08 fixes all four by porting `lightrag/kg/networkx_impl.py`, especially the "degree-priority BFS" inside `get_knowledge_graph()`.

## Solution / 解决方案

Four moves:

1. **`GraphStore` interface frozen from plan.md.** Five methods (`UpsertNode`, `UpsertEdge`, `GetNode`, `GetSubgraph`, `Persist`), each takes `context.Context`. `Entity` is `Name + Type + Description + SourceIDs`; `Relationship` is `SrcID + TgtID + Keywords + Description + Weight + SourceIDs`; `Subgraph` is `[]Entity + []Relationship`.
2. **`AdjacencyGraph` is "three coordinated maps + one RWMutex."** `nodes: map[string]Entity` is the attribute table for nodes (O(1) by ID); `edges: map[edgeKey]Relationship` is the attribute table for edges, where `edgeKey{A, B}` is always `A < B` so `(a, b)` and `(b, a)` land on the same row (undirectedness becomes explicit at the storage layer); `adj: map[string]map[string]struct{}` caches adjacency so BFS expands neighbors in O(deg(v)) instead of O(|E|).
3. **`GetSubgraph` is "level-by-level BFS with degree-DESC sort within each level."** Not Dijkstra, not DFS — process the entire current level (every node at the same depth) as one batch: enumerate all unvisited neighbors → sort the candidate pool DESC by degree → admit one by one until the maxNodes budget runs out → admitted candidates form the next level's frontier. `maxDepth` caps how many expansions we run; the returned `Subgraph.Edges` includes only edges whose BOTH endpoints survived the cap (truncated subgraphs are still well-formed).
4. **`Persist(ctx)` is the same atomic-rename dance as s05/s07.** Marshal in-memory state to `<file>.tmp`, then `os.Rename` to `<dir>/graph.json`. POSIX guarantees rename atomicity on the same filesystem. `NewAdjacencyGraph` auto-loads from disk if the file exists; missing file means we start fresh — NOT an error (matches upstream's "create new empty graph" branch in `__post_init__`).

## How It Works / 工作原理

```
                       AdjacencyGraph (in memory)
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │   nodes  map[string]Entity        — Name → Entity attrs     │
   │   edges  map[edgeKey]Relationship — (A,B) → Relationship    │
   │   adj    map[string]map[string]   — node → set(neighbors)   │
   │   mu     sync.RWMutex                                       │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
            │                  │                  │
            ▼                  ▼                  ▼
       UpsertNode         UpsertEdge          GetSubgraph
            │                  │                  │
            ▼                  ▼                  ▼
       O(1)             O(1) +               BFS with
       overwrite        canonical key        degree priority

   BFS layer-by-layer expansion:

       depth 0:  [seed]
                  │
                  ▼  expand neighbors, sort by degree DESC
       depth 1:  [N1 (deg 5), N2 (deg 3), N3 (deg 1)]
                  │
                  ▼  expand each, dedup, sort by degree DESC
       depth 2:  [M1 (deg 4), M2 (deg 2), ...]
                  │
                  ▼  stop when len(visited) == maxNodes
                     OR depth == maxDepth
```

**Why degree priority instead of distance / weight?** LightRAG's bet is **degree = information density** — an entity that appears in 47 edges is usually a hub ("Scrooge"), and one hop away from a hub there are far more story-relevant nodes than one hop away from a niche entity ("the lamp post on Cornhill Street"). When the context budget is tight, you want the hubs first.

Load-bearing 30 lines from [`agents/s08-graph-store/subgraph_bfs.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s08-graph-store/subgraph_bfs.go), the BFS main loop:

```go
for depth := 0; depth < maxDepth && len(bfsOrder) < maxNodes && len(frontier) > 0; depth++ {
    nextCandidates := make([]levelEntry, 0)
    seenInLevel := make(map[string]struct{})

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

    sort.Slice(nextCandidates, func(i, j int) bool {
        if nextCandidates[i].degree != nextCandidates[j].degree {
            return nextCandidates[i].degree > nextCandidates[j].degree
        }
        return nextCandidates[i].id < nextCandidates[j].id
    })

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
```

**Four non-obvious points:**

1. **The "level" is a cross-section; degree priority is global within it.** Naive BFS sorts neighbors per parent, which only guarantees "of Alice's neighbors, take the highest-degree N1" — but if Bob's neighbors include an even higher-degree M1, naive misses it. We collect-then-sort: pool the entire next level (every parent's neighbors merged) and rank globally by degree. This is exactly upstream's `current_level_nodes.sort(...)` semantic.
2. **Tie-break by ID for determinism.** Go's `sort` isn't stable and map iteration order is randomized; without a deterministic tie-break, the BFS order flips run-to-run and tests / CLI output become unreviewable. `sort.Slice` falls back to `id < id` ASC when degrees are equal — same trick as s07's `VectorHit` tie-break.
3. **Truncated subgraphs include "partial edges."** When maxNodes stops the expansion, the resulting `Edges` slice includes ONLY edges where BOTH endpoints are in `Nodes` — dangling edges into excluded nodes are dropped. The subgraph is still a well-formed graph, so s11's context builder doesn't need a "filter dangling edges" pass. Matches upstream's `nx.subgraph(bfs_nodes).edges()` semantic.
4. **The `adj` table is a redundant but necessary cache.** You could derive adjacency from the `edges` table on the fly — but every BFS neighbor lookup would scan the full edge table at O(|E|). Maintaining `adj` makes `degreeUnlocked / neighborsUnlocked` O(1) / O(deg). The cost is two `adj` writes per UpsertEdge (both directions), trading one extra memory write for one fewer scan — easy win.

## What Changed / 与 s07 的变化

s07 solved "find the most-similar chunk," but chunks have no relationships among themselves. s08 adds the layer above chunks — the graph — without touching s07's interface (s11 will use both vector indices and the graph from the same query mode).

| Dimension | s07 (vector retrieval) | s08 (graph layer) |
|---|---|---|
| Data shape | bags of independent vectors | nodes + undirected edges |
| Primary op | `Query(vec, topK, threshold)` | `GetSubgraph(seed, maxDepth, maxNodes)` |
| Ranking signal | cosine similarity | degree (node connectivity) |
| Retrieval grain | a single record | an entire **subgraph** (nodes + edges) |
| Truncation | topK | maxDepth + maxNodes (two-axis) |
| Tie-break | same score → ID ASC | same degree → ID ASC (same idea) |
| Persistence path | `vdb_<namespace>.json` | `graph.json` |
| Used by query mode | naive (s11) | local / global / hybrid (s11) |

**Why does the graph need to live alongside vectors?** Because they answer different questions:

- Vectors answer: "which span of text in the docs looks like my query?" — shallow semantic match.
- The graph answers: "how do these entities connect to each other?" — deep structural reasoning.

LightRAG's hybrid mode uses both: pick "seed" entities by vector ("the keywords in the user's query are most similar to these entity descriptions"), then expand by graph ("from those entities, one hop away we reach these others"), then stitch the expanded subgraph + the original chunks the entities mentioned into the LLM context window. s08 is the **structural-reasoning** half of that pipeline.

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s08-graph-store

# 1. Build the 6-node hub-and-spokes demo, run subgraph query, persist to ./lightrag-data/
go run . -dir ./lightrag-data

# 2. Re-run to confirm load round-trip (output should be identical)
go run . -dir ./lightrag-data

# 3. Inspect what got written
ls -la ./lightrag-data/
cat ./lightrag-data/graph.json | head -40

# 4. Run all 7 tests (no network, all use t.TempDir)
go test -v ./...
```

Demo output (excerpt):

```
=== full graph: 6 nodes, 7 edges ===
  node  Hub      type=PERSON  "central organizer of the group"
  node  Alice    type=PERSON  "longtime friend of Hub"
  ...
  edge  Hub    -- Alice   keywords="friend"
  edge  Carol  -- Eve     keywords="sibling"
  edge  Bob    -- Carol   keywords="running-partner"

=== subgraph: GetSubgraph("Hub", depth=2, maxNodes=5) ===
got 5 nodes, 5 edges
  node  Hub
  node  Carol     ← deg 3 (Hub, Bob, Eve), wins level 1 by degree
  node  Alice     ← deg 2, ID ASC < Bob
  node  Bob       ← deg 2
  node  Dave      ← level 2: Alice's neighbor (Eve also deg 2 but Dave < Eve)
```

Test matrix:

| Test | Asserts |
|---|---|
| `TestGraphStoreInterfaceContract` | compile-time guard: `*AdjacencyGraph` satisfies `GraphStore` |
| `TestUpsertNodeIdempotent` | second upsert with same Name overwrites, doesn't duplicate |
| `TestUpsertEdgeUndirected` | after `a→b`, BFS from `b` finds `a`, AND `EdgeCount==1` (no double-count) |
| `TestSubgraphBFSDepthLimit` | chain A-B-C-D-E, `depth=2` reaches C only — D and E excluded |
| `TestSubgraphRespectsMaxNodes` | hub + 10 spokes, `maxNodes=4` → hub + 3 spokes + 3 edges |
| `TestSubgraphPrioritizesHighDegree` | hub-A/B/C, B has 4 satellite neighbors → `maxNodes=2` picks hub + B |
| `TestGraphPersistAcrossRestart` | 4-node 4-edge graph, Persist → new instance same path → identical subgraph |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/kg/networkx_impl.py:334-455 (excerpt; full file ~600 lines)
# get_knowledge_graph's BFS main loop — this is the source for s08 GetSubgraph.

async def get_knowledge_graph(
    self,
    node_label: str,
    max_depth: int = 3,
    max_nodes: int = None,
) -> KnowledgeGraph:
    if max_nodes is None:
        max_nodes = self.global_config.get("max_graph_nodes", 1000)
    graph = await self._get_graph()
    result = KnowledgeGraph()

    if node_label == "*":
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_nodes) > max_nodes:
            result.is_truncated = True
        limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        subgraph = graph.subgraph(limited_nodes)
    else:
        if node_label not in graph:
            return KnowledgeGraph()  # empty

        # Modified BFS: prioritize high-degree nodes at the same depth
        bfs_nodes = []
        visited = set()
        queue = deque([(node_label, 0, graph.degree(node_label))])
        has_unexplored_neighbors = False

        while queue and len(bfs_nodes) < max_nodes:
            current_depth = queue[0][1]
            current_level_nodes = []
            while queue and queue[0][1] == current_depth:
                current_level_nodes.append(queue.popleft())
            current_level_nodes.sort(key=lambda x: x[2], reverse=True)

            for current_node, depth, degree in current_level_nodes:
                if current_node not in visited:
                    visited.add(current_node)
                    bfs_nodes.append(current_node)
                    if depth < max_depth:
                        neighbors = list(graph.neighbors(current_node))
                        unvisited_neighbors = [n for n in neighbors if n not in visited]
                        for neighbor in unvisited_neighbors:
                            queue.append((neighbor, depth + 1, graph.degree(neighbor)))
                    else:
                        neighbors = list(graph.neighbors(current_node))
                        if [n for n in neighbors if n not in visited]:
                            has_unexplored_neighbors = True
                if len(bfs_nodes) >= max_nodes:
                    break

        subgraph = graph.subgraph(bfs_nodes)
    # ... result construction (subgraph → KnowledgeGraphNode/Edge lists) elided
```

**How to read it:**

- `deque[(node_label, 0, graph.degree(node_label))]` packs `(node_id, depth, degree)` triples into a single deque — upstream uses one deque to handle BOTH "by depth" and "by degree." We split it into two data structures on the Go side: `frontier` is the current level (a slice), `nextCandidates` is the next-level candidate pool (another slice), with an explicit sort between them. Same semantic, easier to read.
- `current_level_nodes.sort(key=lambda x: x[2], reverse=True)` is "degree DESC within a level" — the move that turns "vector RAG" into "**Light**RAG." Because `current_level_nodes` is a snapshot of the entire level, the sort is global within the level (every parent's neighbors are in there).
- `has_unexplored_neighbors = True` is upstream's truncation signal: if any node at depth==max_depth still has unvisited neighbors, the real graph is bigger than what we returned. We don't propagate this signal in Go — callers can compare `len(Subgraph.Nodes)` to maxNodes directly if they need it.
- `graph.subgraph(bfs_nodes)` is NetworkX's "induced subgraph": pick a node set, automatically retain every edge where both endpoints are in the set. Our Go side runs an explicit `for k := range g.edges` loop at the end of `subgraph_bfs.go` — same "both endpoints in visitedSet" filter.

A more annotated version with reading-map: [`upstream-readings/s08-graph.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s08-graph.py).

The next chapter, [s09 entity/relation extraction](s09-extraction.md), uses the LLM to extract entities and relationships from chunks, then calls s08's `UpsertNode` / `UpsertEdge` to populate the graph. After that, s11's local/global modes will call `GetSubgraph` to traverse it back into a context window. s08 is the storage foundation under that whole chain.
