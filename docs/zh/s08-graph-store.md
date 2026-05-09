---
title: "s08 · 邻接图存储与子图"
chapter: 8
slug: s08-graph-store
est_read_min: 10
---

# s08 · 邻接图存储与子图

> 这一节做什么：在内存里实现一个无向 `AdjacencyGraph`——节点用 `map[string]Entity`，边用按 `(min, max)` 规范化过的 `map[edgeKey]Relationship` 存。`GetSubgraph(seed, maxDepth, maxNodes)` 是「按度数优先」的 BFS——同一层里连接度高的邻居先访问，跟上游 NetworkX 实现的行为对齐。原子重命名的 JSON 快照做持久化。对照上游 `lightrag/kg/networkx_impl.py`。

---

## Problem / 问题

s07 已经有了三个独立的向量索引（chunks / entities / relations），但里面的 entity 之间彼此完全不知道对方的存在——它们只是一袋袋孤立向量。这时候问"Scrooge 跟 Marley 的关系是什么"，向量检索能找到分别提到 Scrooge 和 Marley 的 chunk，但是答不出"它们俩在同一个故事里互为合伙人"——因为这个事实分布在多个 chunk 里，需要**结构化**的连接才能拼出来。

这就是 LightRAG 的核心赌注：**真正的"GraphRAG"必须有一个图，而不只是有一堆 chunk**。具体来说：

1. **没有节点 = 没有 entity-centric 检索。**s11 的 local 模式要做"找跟这个 entity 有关的所有东西"，需要先存得下"这是个 entity"这件事。s07 的 `vdbEntities` 存的是 entity 描述的**向量**，不是 entity 本身的**结构**。
2. **没有边 = 没有 relationship 信号。**s11 的 global 模式要做"找跟这个主题相关的关系网"，需要直接遍历 `(src, tgt, weight)` 三元组。光靠 `vdbRelations` 的向量相似度没法回答"A 跟 B 之间隔了几跳"。
3. **没有子图 = 没法构造上下文窗口。**最后给 LLM 的 prompt 里要塞"实体 X 的邻居 + 邻居的邻居"，长度被 token 预算限制。这必须是个能按深度展开、按重要性截断的 BFS——不是"top-k 最相似的随便一袋"。
4. **NetworkX 不是 Go 库。**上游用 `nx.Graph()` 一行搞定无向图 + GraphML 持久化 + `nx.subgraph()`。我们没这个奢侈，要从 `map[string]Entity` 重新长一个出来——好在原理就 ~150 行。

s08 一次性把这四件事修掉——对照实现 `lightrag/kg/networkx_impl.py`，特别是 `get_knowledge_graph()` 的「按度数优先的 BFS」。

## Solution / 解决方案

四步：

1. **`GraphStore` 接口按 plan.md 锁定。** 五个方法（`UpsertNode` / `UpsertEdge` / `GetNode` / `GetSubgraph` / `Persist`），每个都收 `context.Context`。`Entity` 是 `Name + Type + Description + SourceIDs`；`Relationship` 是 `SrcID + TgtID + Keywords + Description + Weight + SourceIDs`；`Subgraph` 是 `[]Entity + []Relationship`。
2. **`AdjacencyGraph` 是「三个协调的 map + 一个 RWMutex」。** `nodes: map[string]Entity` 存节点属性，O(1) 按 ID 查；`edges: map[edgeKey]Relationship` 存边属性，`edgeKey{A, B}` 永远 `A < B` 所以 `(a, b)` 和 `(b, a)` 落同一行（无向语义在存储层就显式了）；`adj: map[string]map[string]struct{}` 存邻接集，BFS 在这个表上展开避免 O(|E|) 扫边。
3. **`GetSubgraph` 是「层级 BFS + 层内按度数 DESC 排序」。** 不是「单源最短路径」也不是「DFS」——而是把每一层（同一深度的所有节点）当作一个整体处理：枚举本层所有未访问的邻居 → 按度数降序排序 → 逐个塞进结果直到 maxNodes 用完 → 这批节点构成下一层 frontier 继续展开。`maxDepth` 限制扩展轮数；返回的 `Subgraph.Edges` 只包含两端都在 `Nodes` 里的边（被截断的子图依然合法）。
4. **`Persist(ctx)` 跟 s05/s07 是同一个原子重命名套路。** 写到 `<file>.tmp` → `os.Rename` 到 `<dir>/graph.json`。POSIX 保证同文件系统下 rename 原子。`NewAdjacencyGraph` 在文件存在时自动 load，文件不存在就从空开始（**不**报错，对应上游 `__post_init__` 里"create new empty graph"的分支）。

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

**为什么是「按度数」而不是「按距离」或「按 Weight」？** LightRAG 的赌注是：**度数高 = 信息密度高**——一个 entity 出现在 47 条边里，往往说明它是个 hub（"Scrooge"），周围一跳能拿到的故事节点远比一个偏门 entity（"the lamp post on Cornhill Street"）多。当上下文 budget 紧的时候要先要 hub。

[`agents/s08-graph-store/subgraph_bfs.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s08-graph-store/subgraph_bfs.go) 中 BFS 主循环的关键 30 行：

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

**四个不太显眼的点：**

1. **「层」是横切，度数排序是层内全局。** 朴素 BFS 在每个 parent 内部排序，那就只能保证 "Alice 的邻居里挑 deg 最高的 N1"——但如果 Bob 的邻居里有 deg 更高的 M1，朴素法漏掉它。我们 collect-then-sort：把整层（所有 parent 的邻居加起来）当一个候选池，全局排度数。这才是上游 `current_level_nodes.sort(...)` 的语义。
2. **同度数按 ID 字典序破平。** Go 的 `sort` 不稳定，map 迭代顺序更是随机的——没有确定性 tie-break，每次跑出的 BFS 顺序都会变，测试和 CLI 输出都没法对。`sort.Slice` 在度数相等时按 `id < id` 升序，跟 s07 的 `VectorHit` tie-break 是同一思路。
3. **截断子图也带「部分边」。** maxNodes 截断时停掉再展开，**最后输出的 `Edges` 只包含两端都在 `Nodes` 里的边**——丢弃掉指向被截断节点的"悬空边"。子图依然是个合法图，s11 的 context builder 不需要做"过滤悬空边"的特殊处理。这跟上游 `nx.subgraph(bfs_nodes).edges()` 的语义一致。
4. **`adj` 表是冗余但必要的缓存。** 完全可以从 `edges` 表反推 adjacency——但每次 BFS 取邻居就要扫整张 `edges` 表，O(|E|)。维护一份 `adj` 让 `degreeUnlocked / neighborsUnlocked` 都是 O(1) / O(deg)。代价就是 UpsertEdge 要写两次 `adj`（双向），多写一次内存换一次扫边——划算。

## What Changed / 与 s07 的变化

s07 解决了"找最像的 chunk"，但 chunk 之间没有关系。s08 把"chunk 之外"的图层加进来，同时不动 s07 的接口（s11 时三向量索引和图会一起被 query mode 用到）。

| 维度 | s07（向量检索） | s08（图层） |
|---|---|---|
| 数据形态 | 一袋袋向量（独立记录） | 节点 + 无向边的图 |
| 主要操作 | `Query(vec, topK, threshold)` | `GetSubgraph(seed, maxDepth, maxNodes)` |
| 排序信号 | 余弦相似度 | 度数（节点连接度） |
| 检索粒度 | 单条记录 | 一整个**子图**（节点 + 边 + 邻接关系） |
| 截断策略 | topK 截 | maxDepth + maxNodes 双重截 |
| Tie-break | 同分按 ID 升序 | 同度数按 ID 升序（同一思路） |
| 持久化路径 | `vdb_<namespace>.json` | `graph.json` |
| 适用 query 模式 | naive（s11） | local / global / hybrid（s11） |

**为什么图要跟向量一起活着？** 因为它们答的是不同的问题：

- 向量答："文档里哪段文字跟我这个查询长得像？"——浅层语义匹配。
- 图答："这些 entity 怎么互相连？"——深层结构推理。

LightRAG 的 hybrid 模式同时用两者：先用向量挑出"种子" entity（"用户问的关键词跟这几个 entity 描述最像"），然后用图扩展（"这几个 entity 一跳能到哪些其他 entity"），最后把扩展开的子图 + 那些 entity 提到的原始 chunk 拼成 context window。s08 是这个流水线里**结构推理**那一半的存储。

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s08-graph-store

# 1. 构建 6 节点 hub-and-spokes 图，跑子图查询，持久化到 ./lightrag-data/
go run . -dir ./lightrag-data

# 2. 再跑一次确认 load 往返（输出应该完全一样）
go run . -dir ./lightrag-data

# 3. 看看写出了什么
ls -la ./lightrag-data/
cat ./lightrag-data/graph.json | head -40

# 4. 跑 7 个测试（不联网，全部用 t.TempDir）
go test -v ./...
```

demo 输出（节选）：

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
  node  Carol     ← deg 3 (Hub, Bob, Eve), 在第 1 层先选
  node  Alice     ← deg 2, ID 升序 < Bob
  node  Bob       ← deg 2
  node  Dave      ← 第 2 层，Alice 的邻居（Carol 的邻居 Eve 也是 deg 2，但 Dave < Eve）
```

测试矩阵：

| 测试 | 断言 |
|---|---|
| `TestGraphStoreInterfaceContract` | 编译期约束：`*AdjacencyGraph` 满足 `GraphStore` |
| `TestUpsertNodeIdempotent` | 同 Name 第二次插入 = 覆盖，不是重复 |
| `TestUpsertEdgeUndirected` | `a→b` 之后从 `b` 也能 BFS 到 `a`，且 `EdgeCount==1`（无向不重复） |
| `TestSubgraphBFSDepthLimit` | 链 A-B-C-D-E，`depth=2` 只到 C，D/E 不进 |
| `TestSubgraphRespectsMaxNodes` | 10 spokes，`maxNodes=4` → hub + 3 个 spoke，加 3 条边 |
| `TestSubgraphPrioritizesHighDegree` | hub-A/B/C，B 自带 4 邻居 → `maxNodes=2` 选 hub + B（高度数） |
| `TestGraphPersistAcrossRestart` | 4 节点 4 边图，Persist → 同路径新实例 → 子图一致 |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/kg/networkx_impl.py:334-455 (节选；完整文件 ~600 行)
# get_knowledge_graph 的 BFS 主循环——这就是 s08 GetSubgraph 对应的源头。

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
    # ... 后续把 subgraph 转成 KnowledgeGraphNode/Edge 列表（省略）
```

**怎么读这段：**

- `deque[(node_label, 0, graph.degree(node_label))]` 把 `(node_id, depth, degree)` 三元组塞队列——上游用一个 deque 同时管「按层」和「按度数」。我们 Go 端拆成两个数据结构：`frontier` 是当前层（一个 slice），`nextCandidates` 是下层候选池（再一个 slice），中间显式 sort 一次。同样的语义，更易读。
- `current_level_nodes.sort(key=lambda x: x[2], reverse=True)` 是「层内按度数 DESC」——这就是「光 RAG」变成「Light**RAG**」的那一步。`current_level_nodes` 是一整层的快照，所以排序是全局的（同一层所有 parent 的邻居都在里面）。
- `has_unexplored_neighbors = True` 是上游的"截断信号"：如果某个 depth==max_depth 的节点还有没访问的邻居，说明真实的图比我们返回的要大。我们 Go 端没传这个信号——调用方需要的话比较 `len(Subgraph.Nodes)` 跟 maxNodes 就行。
- 最后那行 `graph.subgraph(bfs_nodes)` 是 NetworkX 的"诱导子图"：选定一组节点，自动保留所有两端都在选定集里的边。我们 Go 端在 `subgraph_bfs.go` 末尾用一个 `for k := range g.edges` 显式跑一遍——同样的"两端都在 visitedSet 里"过滤。

带注解的版本+阅读地图见 [`upstream-readings/s08-graph.py`](https://github.com/Ding-Ye/galaxy-lightrag/blob/main/upstream-readings/s08-graph.py)。

下一节 [s09 实体关系抽取](s09-extraction.md) 会用 LLM 从 chunk 里抽 entity/relationship，然后调用 s08 的 `UpsertNode` / `UpsertEdge` 把图填起来。再下一节 s11 的 local/global 模式会调 `GetSubgraph` 把图遍历回上下文。s08 是这条链的存储底座。
