# =============================================================================
# Upstream reading for s08 — Adjacency graph store + subgraph BFS
# =============================================================================
#
# Source:  https://github.com/HKUDS/LightRAG (commit on 2026-05-09 main)
# File:    lightrag/kg/networkx_impl.py
# Lines:   24-68 (class def + persistence) + 135-155 (upsert) + 334-455 (BFS)
#
# License: MIT (HKUDS, 2025) — same as this repo. Excerpt is verbatim with
#          ellipsis comments marked "[s08: omitted ...]".
#
# This is the file our agents/s08-graph-store/{adjacency_graph.go,
# subgraph_bfs.go} ports.  Read this first, then open the two Go files
# side-by-side.  Three correspondences are load-bearing:
#
#   1. UNDIRECTED edges with auto-created endpoints.  Upstream uses
#      networkx's `nx.Graph()` (undirected) and calls `add_edge(src, tgt)`
#      directly — networkx auto-creates missing nodes.  Our Go port stores
#      edges keyed by `edgeKey{A, B}` where A < B lexicographically; that's
#      what makes (a, b) and (b, a) collapse onto one row.  UpsertEdge also
#      auto-creates missing endpoint nodes — same NetworkX semantic.
#
#   2. DEGREE-PRIORITY BFS at each level.  This is the algorithmic core and
#      what makes "Light**RAG**" actually work for retrieval.  The full
#      get_knowledge_graph() function is excerpted below; the load-bearing
#      line is `current_level_nodes.sort(key=lambda x: x[2], reverse=True)`.
#      Our Go port mirrors this exactly: collect-then-sort the entire next
#      level, admit candidates one at a time until maxNodes runs out.
#
#   3. GraphML on disk vs JSON on disk.  Upstream writes networkx's GraphML
#      format (XML) for cross-tool compat with Gephi / Cytoscape.  We write
#      a flat JSON (nodes table + edges table) for stdlib-only and grep-
#      ability.  Trade-off: cannot re-open in Gephi without conversion.
#      Exercise: add a `WithGraphMLFormat()` option.
#
# Tests in our adjacency_graph_test.go assert all three: undirected
# semantics, degree-priority on truncation, and persistence round-trip.

# -----------------------------------------------------------------------------
# Verbatim excerpt (lightrag/kg/networkx_impl.py:24-68, ~45 LOC):
# -----------------------------------------------------------------------------

# @final
# @dataclass
# class NetworkXStorage(BaseGraphStorage):
#     @staticmethod
#     def load_nx_graph(file_name) -> nx.Graph:
#         if os.path.exists(file_name):
#             return nx.read_graphml(file_name)
#         return None
#
#     @staticmethod
#     def write_nx_graph(graph: nx.Graph, file_name, workspace="_"):
#         logger.info(
#             f"[{workspace}] Writing graph with {graph.number_of_nodes()} nodes, "
#             f"{graph.number_of_edges()} edges"
#         )
#         nx.write_graphml(graph, file_name)
#
#     def __post_init__(self):
#         working_dir = self.global_config["working_dir"]
#         if self.workspace:
#             workspace_dir = os.path.join(working_dir, self.workspace)
#         else:
#             workspace_dir = working_dir
#             self.workspace = ""
#         os.makedirs(workspace_dir, exist_ok=True)
#         self._graphml_xml_file = os.path.join(
#             workspace_dir, f"graph_{self.namespace}.graphml"
#         )
#         self._storage_lock = None
#         self.storage_updated = None
#         self._graph = None
#         preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
#         self._graph = preloaded_graph or nx.Graph()
#
#     async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
#         graph = await self._get_graph()
#         graph.add_node(node_id, **node_data)
#
#     async def upsert_edge(
#         self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
#     ) -> None:
#         graph = await self._get_graph()
#         graph.add_edge(source_node_id, target_node_id, **edge_data)

# -----------------------------------------------------------------------------
# get_knowledge_graph BFS (lightrag/kg/networkx_impl.py:334-455, ~50 LOC):
# -----------------------------------------------------------------------------

# async def get_knowledge_graph(
#     self,
#     node_label: str,
#     max_depth: int = 3,
#     max_nodes: int = None,
# ) -> KnowledgeGraph:
#     if max_nodes is None:
#         max_nodes = self.global_config.get("max_graph_nodes", 1000)
#     graph = await self._get_graph()
#     result = KnowledgeGraph()
#
#     if node_label == "*":
#         degrees = dict(graph.degree())
#         sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
#         if len(sorted_nodes) > max_nodes:
#             result.is_truncated = True
#         limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
#         subgraph = graph.subgraph(limited_nodes)
#     else:
#         if node_label not in graph:
#             return KnowledgeGraph()  # empty
#
#         # Modified BFS: prioritize high-degree nodes at the same depth
#         bfs_nodes = []
#         visited = set()
#         queue = deque([(node_label, 0, graph.degree(node_label))])
#         has_unexplored_neighbors = False
#
#         while queue and len(bfs_nodes) < max_nodes:
#             current_depth = queue[0][1]
#             current_level_nodes = []
#             while queue and queue[0][1] == current_depth:
#                 current_level_nodes.append(queue.popleft())
#             current_level_nodes.sort(key=lambda x: x[2], reverse=True)
#
#             for current_node, depth, degree in current_level_nodes:
#                 if current_node not in visited:
#                     visited.add(current_node)
#                     bfs_nodes.append(current_node)
#                     if depth < max_depth:
#                         neighbors = list(graph.neighbors(current_node))
#                         unvisited = [n for n in neighbors if n not in visited]
#                         for neighbor in unvisited:
#                             queue.append((neighbor, depth + 1, graph.degree(neighbor)))
#                     else:
#                         if [n for n in graph.neighbors(current_node) if n not in visited]:
#                             has_unexplored_neighbors = True
#                 if len(bfs_nodes) >= max_nodes:
#                     break
#
#         subgraph = graph.subgraph(bfs_nodes)
#     # [s08: omitted — KnowledgeGraphNode/Edge result construction]
#     return result

# -----------------------------------------------------------------------------
# Reading map — what to read AFTER s08
# -----------------------------------------------------------------------------
#
# Upstream files that produce / consume the graph this store holds:
#
#   lightrag/operate.py:~2883-3163  (extract_entities)
#       Where the graph actually gets POPULATED.  After the LLM extracts
#       entities and relationships from each chunk, this code calls
#       `graph_storage.upsert_node(...)` and `graph_storage.upsert_edge(...)`
#       — the same two methods s08 ports.  Source IDs aggregate so each
#       node/edge knows which chunks mention it.
#       → Maps to s09 (Entity/relation extraction with gleaning).  Watch for
#         `_merge_nodes_then_upsert` and `_merge_edges_then_upsert` — they're
#         the helpers that gather descriptions from many chunks before the
#         single upsert call.
#
#   lightrag/operate.py:167-385  (summarization)
#       When an entity is mentioned in many chunks, descriptions are merged
#       (LLM-summarized if over a threshold) BEFORE upsert_node so the graph
#       carries one good description per entity, not a concatenated mess.
#       → Maps to s10 (Map-reduce description summarization).  The
#         `_handle_entity_relation_summary` decision tree (skip-LLM
#         heuristic vs LLM call) is the centerpiece.
#
#   lightrag/operate.py:3164-3410 + :3516-4055  (kg_query → local/global)
#       Where get_knowledge_graph() actually gets called.  Local mode uses
#       the seed entity → 1-hop subgraph; global mode walks relation-centric
#       subgraphs.  Both go through the same BFS we ported.
#       → Maps to s11 (Dual-level retrieval).  Watch for the ContextBuilder
#         logic — it's where Subgraph + chunks + summaries get spliced into
#         a token-budgeted prompt.
#
# What s08 deliberately omits (and why):
#
#   - GraphML persistence.  Upstream uses `nx.write_graphml(graph, path)` so
#     the resulting file opens in Gephi / Cytoscape for visual debugging.
#     We use plain JSON for stdlib-only + grep-ability.  Exercise: add a
#     `WithGraphMLFormat()` option using encoding/xml.
#
#   - Cross-process update flags.  Upstream uses get_update_flag /
#     set_all_update_flags to coordinate between Python processes sharing a
#     working_dir.  Our Go port uses one process and a single sync.RWMutex
#     — no IPC needed.  Exercise: add fcntl-based file locking for
#     multi-process safety.
#
#   - The "*" wildcard seed.  Upstream's get_knowledge_graph("*", ...)
#     returns the top-N nodes by degree without any BFS at all (used by the
#     web UI's "show me the whole graph" button).  We left it out — drop in
#     a 5-line branch if you want it: sort all g.nodes by g.degreeUnlocked
#     DESC, take top maxNodes.  See the `if node_label == "*":` block above.
#
#   - is_truncated flag.  Upstream returns a KnowledgeGraph with a bool flag
#     so the caller knows the result was capped.  We left it out — callers
#     can compare `len(Subgraph.Nodes)` to maxNodes themselves.
#
#   - Per-edge directionality.  Upstream's nx.Graph IS undirected; ours is
#     too.  But neither stores "which side did the LLM extract first" once
#     the canonical edgeKey is computed.  If you need that, add a
#     SrcID/TgtID pair to the stored Relationship — we already do (preserve
#     the caller's insertion direction in the value, not the key).
#
# -----------------------------------------------------------------------------
# Glossary one-liners
# -----------------------------------------------------------------------------
#
#   undirected graph              — an edge between A and B is reachable
#                                   from both endpoints.  In storage terms:
#                                   one canonical (min, max) row, not two.
#   degree                        — number of edges incident on a node.
#                                   Used as the priority signal in BFS.
#   degree-priority BFS           — at each depth level, sort the next-layer
#                                   candidates by degree DESC before
#                                   admitting them up to the maxNodes cap.
#   max_depth                     — how many BFS expansions to run.  depth=0
#                                   is the seed, depth=1 is its neighbors,
#                                   depth=N is the last layer to expand.
#   max_nodes                     — total node budget for the result.
#                                   Truncates LATE (post-sort) to keep the
#                                   highest-degree nodes within budget.
#   induced subgraph              — given a node set, the subgraph that
#                                   includes those nodes plus EVERY edge
#                                   whose endpoints are both in the set.
#                                   → Our Subgraph.Edges semantic.
#   GraphML                       — XML-based graph file format networkx
#                                   uses by default.  We write JSON instead.
#   namespace                     — string key for the graph file path.
#                                   Upstream: `graph_<namespace>.graphml`.
#                                   Ours: `graph.json` (single namespace per
#                                   AdjacencyGraph instance — simpler).
