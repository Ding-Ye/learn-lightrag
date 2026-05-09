# s08 · Adjacency graph store + BFS / 邻接图存储与子图

Undirected `AdjacencyGraph` backed by `map[string]Entity` for nodes and a
canonicalized `map[edgeKey]Relationship` for edges (where `edgeKey` orders
the endpoints `(min, max)` so `(a, b)` and `(b, a)` collapse onto one row).
`GetSubgraph(seed, maxDepth, maxNodes)` is BFS with **degree-based
priority** — visit the most-connected neighbors first, mirroring upstream
NetworkX's behavior. Atomic-rename JSON snapshot for persistence.

s07 shipped vector retrieval; without a graph layer there is no
"Light**RAG**" — only a flat vector index, indistinguishable from textbook
RAG. s08 adds the graph layer that s09's extraction will populate and s11's
local/global query modes will traverse.

无向 `AdjacencyGraph`：`map[string]Entity` 存节点，`map[edgeKey]Relationship`
存边（`edgeKey` 按 `(min, max)` 规范化，让 `(a, b)` 和 `(b, a)` 落到同一行）。
`GetSubgraph(seed, maxDepth, maxNodes)` 是「按度数优先」的 BFS——优先访问
连接度高的邻居，对应上游 NetworkX 的行为。原子重命名的 JSON 快照做持久化。

s07 给了向量检索；缺了图这层就只是「Light + 向量 RAG」，跟教科书 RAG 没差别。
s08 把图加上来，s09 的实体抽取会往里写，s11 的 local/global 模式会遍历它。

## Run / 运行

```bash
cd /Users/yeding/learn-lightrag/agents/s08-graph-store

# CLI demo: build a 6-node hub-and-spokes graph, run GetSubgraph, persist
go run . -dir ./lightrag-data

# Re-run to see the load round-trip (same output, no recomputation)
go run . -dir ./lightrag-data

# Tests (no network, all use t.TempDir)
go test -v ./...
```

The demo prints the full graph, calls `GetSubgraph("Hub", depth=2, maxNodes=5)`,
and writes `graph.json` under `-dir`.

## Files / 文件

| File | Lines | What |
|---|---|---|
| `graph_store.go` | ~70 | `Entity`, `Relationship`, `Subgraph` structs + `GraphStore` interface |
| `adjacency_graph.go` | ~330 | In-memory impl: 3 coordinated maps + RWMutex + JSON Persist/Load |
| `subgraph_bfs.go` | ~150 | BFS with degree priority, level-by-level frontier, partial-edge filtering |
| `main.go` | ~110 | Hub-and-spokes social network demo, calls GetSubgraph + Persist |
| `adjacency_graph_test.go` | ~250 | 6 spec tests + interface contract |

## Tests / 测试清单

- `TestGraphStoreInterfaceContract` — compile-time guard
- `TestUpsertNodeIdempotent` — same Name twice = overwrite, not duplicate
- `TestUpsertEdgeUndirected` — `a→b` is reachable from both `a` and `b`
- `TestSubgraphBFSDepthLimit` — chain A-B-C-D-E, depth=2 gets {A,B,C} only
- `TestSubgraphRespectsMaxNodes` — 10 spokes, maxNodes=4 → hub + 3 spokes
- `TestSubgraphPrioritizesHighDegree` — high-degree neighbor wins the slot
- `TestGraphPersistAcrossRestart` — Persist → new instance → same subgraph

## Reading order / 阅读顺序

1. `docs/{zh,en}/s08-graph-store.md` — mental model + ASCII + diff vs s07
2. `graph_store.go` — types and interface (read signatures first)
3. `adjacency_graph.go` — `NewAdjacencyGraph` → `UpsertNode` → `UpsertEdge` → `Persist`
4. `subgraph_bfs.go` — the BFS, level-by-level
5. `main.go` — wiring it all together
6. `upstream-readings/s08-graph.py` — upstream Python on the side

`s09-extraction` writes entities into this graph from LLM extraction;
`s11-query-modes` calls `GetSubgraph` for the local/global retrieval paths.
