---
title: "s07 · 余弦相似向量库"
chapter: 7
slug: s07-vector-store
est_read_min: 9
---

# s07 · 余弦相似向量库

> 这一节做什么：把 s01 里那个「切片 + 线性余弦扫描」的占位实现升级成一个真正的 `VectorStore`。基于 `[]VectorRecord` 的余弦相似度计算，带阈值过滤，原子持久化为 JSON 快照，并且按上游的「三个独立索引」模式同时跑起来（`vdbChunks` / `vdbEntities` / `vdbRelations`，三个实例共用同一个实现）。对照上游 `lightrag/kg/nano_vector_db_impl.py`，但**不**实现 `float16 + zlib + base64` 三件套压缩——会讲，但作为练习留给读者。

---

## Problem / 问题

s01 里的向量库基本就是个能跑的 demo：

```go
// s01/vectorstore.go (简化)
type vstore struct{ recs []record }

func (s *vstore) Query(q []float32, k int) []record {
    // 线性扫描，没阈值，没 metadata，没持久化
    scored := make([]record, 0, len(s.recs))
    for _, r := range s.recs {
        score := dot(q, r.vec) / (norm(q) * norm(r.vec))
        scored = append(scored, record{r.id, r.vec, score})
    }
    sort.Slice(scored, ...)
    return scored[:k]
}
```

只要超过 50 个 chunk，四件事就开始崩：

1. **没阈值。**哪怕*所有*记录都只有 0.05 的得分（也就是没有真正相似的东西），返回的 top-k 仍然是一堆噪声。上游的 `cosine_better_than_threshold`（一般 0.2-0.4）把底部切掉，避免把 LLM context 灌满垃圾。
2. **没持久化。**进程一重启索引就消失了，s01 的 pipeline 每次都要重新 embed 所有 chunk，OpenAI 账单直接 100 倍。
3. **一个袋子装所有东西。**真实的 LightRAG 维护**三个**独立索引——chunks / entities / relations，在不同检索阶段查不同的库。如果只有一个共享的 store，"按关键字找 top-k 实体"还得在事后按 metadata 过滤一遍，又慢又容易出错。
4. **Metadata 不能往返。**`Query` 只返回 ID 的话，要拿 chunk 文本还得去 KV store 二次查询。上游直接把 metadata（chunk 内容、entity 名称等）挂在每条记录上，检索一次往返完事。

s07 一次性修复这四件事——对照实现 `lightrag/kg/nano_vector_db_impl.py:79-150`，也就是 `NanoVectorDBStorage` 类的 `upsert()` 和 `query()` 方法。

## Solution / 解决方案

四步：

1. **`VectorStore` 接口按 plan.md 锁定。** 四个方法（`Upsert` / `Query` / `Delete` / `Persist`），每个方法第一参数都是 `context.Context`。`VectorRecord` 包含 `ID`、`Vector []float32`、`Metadata map[string]any`；`VectorHit` 多一个 `Score float32`。`Namespace` 常量（`chunks` / `entities` / `relations`）让上游"三个索引"的模式在类型层面就显眼。
2. **`CosineIndex` 是「内存里的 `[]VectorRecord` + 一个 id→index 的快查 map」。** Upsert 用 ID 在 O(1) 时间去重；Delete 时整体 splice 然后重建 map。第一条记录确定 `dim`；后续插入维度不一致的会被拒掉，避免静默的维度漂移腐化整个索引。
3. **`Query(query, topK, threshold)` 一次 `O(N · dim)` 扫描 + 降序排序 + topK 截断。** 阈值是「>= 包含」语义——刚好等于阈值的那条会被保留，跟上游 `better_than_threshold` 一致。空索引返回 `[]VectorHit{}`（不是 nil）+ nil error。
4. **`Persist(ctx)` 跟 s05 KV store 是同一个原子重命名套路。** 把内存状态序列化到 `<file>.tmp`，然后 `os.Rename` 到 `<dir>/vdb_<namespace>.json`。POSIX 保证同文件系统下 rename 是原子的，所以读者要么看到完整的旧文件、要么完整的新文件，不会卡在中间状态。`NewCosineIndex` 在文件存在时自动加载，文件不存在就从空开始（**不**报错）。

## How It Works / 工作原理

```
                    pipeline (s11) creates THREE indices
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       vdbChunks            vdbEntities         vdbRelations
       (NS=chunks)          (NS=entities)       (NS=relations)
              │                   │                   │
              ▼                   ▼                   ▼
     ┌─────────────────────────────────────────────────────┐
     │ CosineIndex                                         │
     │   recs   []VectorRecord    (slice; index access)    │
     │   idIdx  map[string]int    (id → recs index)        │
     │   dim    int               (locked on first Upsert) │
     │   mu     sync.RWMutex      (R for Query, W for Up.) │
     └─────────────────────────────────────────────────────┘
              │
              ▼
     Upsert  ──► dedup / dim-check / append    ──► O(1) per record
     Query   ──► loop recs, cosine, threshold,
                 sort desc, slice [:topK]      ──► O(N · dim)
     Persist ──► write tmp + rename atomic     ──► <dir>/vdb_<ns>.json
     Load    ──► auto on NewCosineIndex        ──► same path
```

余弦相似度公式：

```
                    a · b
    cos(a, b) = ──────────────
                  |a| * |b|

    其中  a · b   = sum(a[i] * b[i])
          |a|     = sqrt(sum(a[i]^2))
```

`|query|` 在循环外算一次，`|rec|` 在内联里算——不会每条记录都重新算 `|query|`。如果有任意一边是零向量我们直接返回 0（避开除零），上游会抛异常，我们更倾向于优雅退化——避免单条全零记录把整个 query 500 掉。

[`agents/s07-vector-store/cosine_index.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s07-vector-store/cosine_index.go) 中 `Query` 的关键 30 行：

```go
func (s *CosineIndex) Query(ctx context.Context, query []float32, topK int, threshold float32) ([]VectorHit, error) {
    if err := ctx.Err(); err != nil {
        return nil, err
    }
    if topK <= 0 {
        return []VectorHit{}, nil
    }
    s.mu.RLock()
    defer s.mu.RUnlock()
    if len(s.recs) == 0 {
        return []VectorHit{}, nil
    }
    if s.dim != 0 && len(query) != s.dim {
        return nil, &ErrDimensionMismatch{Expected: s.dim, Got: len(query), ID: "<query>"}
    }
    qNorm := vectorNorm(query)
    hits := make([]VectorHit, 0, len(s.recs))
    for i := range s.recs {
        rec := &s.recs[i]
        score := cosineSimilarity(query, rec.Vector, qNorm)
        if score < threshold {
            continue
        }
        hits = append(hits, VectorHit{ID: rec.ID, Score: score, Metadata: copyMetadata(rec.Metadata)})
    }
    sort.Slice(hits, func(i, j int) bool {
        if hits[i].Score != hits[j].Score {
            return hits[i].Score > hits[j].Score
        }
        return hits[i].ID < hits[j].ID
    })
    if len(hits) > topK {
        hits = hits[:topK]
    }
    return hits, nil
}
```

**四个不太显眼的点：**

1. **`|query|` 算一次而不是每条记录都算。**朴素写法把 `cosine = dot(a,b) / (norm(a)*norm(b))` 整个塞循环里，但 `norm(a)` 跟 `b` 无关。提到外面常数因子能降三成。上游 NanoVectorDB 也是这么干的——`numpy.linalg.norm(query)` 算一次然后复用。
2. **同分时按 ID 字典序破平。** 两条得分完全相同的记录（罕见但有可能在阈值边缘出现）按 ID 字母序排序。没这个步骤的话测试的输出就不可复现——Go 的 `sort` 不是稳定排序，就算稳定也救不了你，因为 Go map 迭代顺序是随机的，插入顺序根本传不下来。`TestCosinePersistAcrossRestart` 要求两个索引实例返回字节级一致的 hits，没确定性 tie-break 根本做不到。
3. **`Metadata` 出去时做防御性拷贝。**调用方写 `hit.Metadata["foo"] = "tampered"` 不能污染索引下次查询的结果。`TestCosineMetadataRoundtrip` 故意改返回的 hit 然后再查一次，断言原始值还在。Upsert 入参也做同样的防御性拷贝（调用方在 Upsert 之后改输入 map 也漏不进来）。
4. **第一条记录锁定维度，后续不一致的直接拒。**上游悄悄等到后面 numpy 矩阵运算时崩；我们在 Upsert 边界就用类型化的 `ErrDimensionMismatch` 报错，把锅指给犯错的那条记录的 ID。`TestCosineDimensionMismatchRejected` 既断言报错也断言索引大小没变（拒掉的记录没漏进去）。

## What Changed / 与 s01 的变化

| 维度 | s01（占位实现） | s07（正式版） |
|---|---|---|
| 存储 | `vectorstore.go` 内嵌 `[]record` | 独立的 `CosineIndex` 结构 + id→index map |
| 索引数 | 一个袋子装所有 | 三个独立实例（chunks / entities / relations） |
| 阈值 | 没有 | `threshold float32` 参数；低于阈值丢弃 |
| 持久化 | 没有 | 原子重命名到 `<dir>/vdb_<namespace>.json` |
| 自动加载 | n/a | `NewCosineIndex` 在文件存在时自动 load |
| Metadata | 没有 | 每条记录一个 `map[string]any`，出去时防御性拷贝 |
| Dim 检查 | `len(vec)` 不一致直接 panic | 类型化 `ErrDimensionMismatch` 在 Upsert 边界拦截 |
| 空索引 | 行为未定义 | `[]VectorHit{}` + nil error |
| Tie-break | sort 不稳定 | 得分降序、ID 升序，确定性 |

三个独立索引是这一节的架构关键——s11 才能在不扫 chunks 和 relations 的前提下问"按关键字找 top-k 实体"。代价是 `Insert` 现在要写两个 store（embed 后写 vdbChunks，s09 抽取后写 vdbEntities / vdbRelations）。

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s07-vector-store

# 1. 构建三个索引并持久化到 ./lightrag-data/
go run . -dir ./lightrag-data

# 2. 再跑一次确认 load 往返（输出应该完全一样）
go run . -dir ./lightrag-data

# 3. 看看写出了什么
ls -la ./lightrag-data/
cat ./lightrag-data/vdb_chunks.json | head -30

# 4. 跑全部 8 个测试（不联网，全部用 t.TempDir）
go test -v ./...
```

demo 输出：

```
=== namespace: chunks ===
count=6  query=chunk-2  top-3:
  1. id=chunk-2        score=1.0000
  2. id=chunk-1        score=0.4068
  3. id=chunk-3        score=0.3132
persisted -> ./lightrag-data/vdb_chunks.json

=== namespace: entities ===
count=5  query=entity-1  top-3:
  1. id=entity-1       score=1.0000
  ...
```

测试矩阵：

| 测试 | 断言 |
|---|---|
| `TestVectorStoreInterfaceContract` | 编译期约束：`*CosineIndex` 满足 `VectorStore` |
| `TestCosineRanksMostSimilarFirst` | 用某条记录的同向量去查 → 该记录排第一，得分 ≈ 1.0 |
| `TestCosineThresholdFiltersLowScores` | `threshold=0.99` 只保留几乎完全一致的匹配 |
| `TestCosineRespectsTopK` | 10 条记录、`topK=3` → 正好 3 个 hit |
| `TestCosinePersistAcrossRestart` | Persist → 用同一路径新建实例 → hits 一致（含 metadata） |
| `TestCosineHandlesEmptyIndex` | 空索引 Query 返回 `[]VectorHit{}`（非 nil），无 error |
| `TestCosineMetadataRoundtrip` | upsert 时 `{"foo":"bar"}` 能 query 拿回，调用方改了 hit 也不会反向漏回索引 |
| `TestCosineDimensionMismatchRejected` | 第二次插入维度不一致被拒，Len 不变 |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/kg/nano_vector_db_impl.py:79-150 (节选；完整文件 ~270 行)
# 两个方法，都是 async，都被 per-namespace 的 asyncio lock 包着。

async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
    if not data:
        return
    current_time = int(time.time())
    list_data = [
        {
            "__id__": k,
            "__created_at__": current_time,
            **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
        }
        for k, v in data.items()
    ]
    contents = [v["content"] for v in data.values()]
    batches = [
        contents[i : i + self._max_batch_size]
        for i in range(0, len(contents), self._max_batch_size)
    ]
    embedding_tasks = [
        self.embedding_func(batch, context="document") for batch in batches
    ]
    embeddings_list = await asyncio.gather(*embedding_tasks)
    embeddings = np.concatenate(embeddings_list)
    if len(embeddings) == len(list_data):
        for i, d in enumerate(list_data):
            # Compress vector using Float16 + zlib + Base64 for storage optimization
            vector_f16 = embeddings[i].astype(np.float16)
            compressed_vector = zlib.compress(vector_f16.tobytes())
            encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
            d["vector"] = encoded_vector
            d["__vector__"] = embeddings[i]
        client = await self._get_client()
        results = client.upsert(datas=list_data)
        return results

async def query(
    self, query: str, top_k: int, query_embedding: list[float] = None
) -> list[dict[str, Any]]:
    if query_embedding is not None:
        embedding = query_embedding
    else:
        embedding = await self.embedding_func([query], context="query", _priority=5)
        embedding = embedding[0]
    client = await self._get_client()
    results = client.query(
        query=embedding,
        top_k=top_k,
        better_than_threshold=self.cosine_better_than_threshold,
    )
    results = [
        {
            **{k: v for k, v in dp.items() if k != "vector"},
            "id": dp["__id__"],
            "distance": dp["__metrics__"],
            "created_at": dp.get("__created_at__"),
        }
        for dp in results
    ]
    return results
```

**怎么读这段：**

- `upsert` 把 embedding **放在了 storage 方法内部**——上游把"嵌入 + 插入"耦合在一起，因为 storage 持有 `embedding_func` 引用。我们在 s07 的拆法是：embedder 留在 s06，`Upsert` 接收已经 embed 好的向量。`[]VectorRecord` 这层 seam 跟上游 `list_data` 的形状一致。
- `vector_f16 + zlib + base64` 三件套是上游的 disk 体积优化（相比裸 float32 JSON 能省 25-50%）。我们为了代码透明把这部分省了；这一节写出的 JSON 文件会大一些，但 `git diff` 起来一目了然。**练习：** 加一个 `WithCompression()` 选项把上游的编码方式重新实现一下。
- `better_than_threshold` 跟我们的 INCLUSIVE 过滤一致——`score >= threshold` 保留。默认值在不同 workspace 下是 0.2-0.4。
- 上游每条记录都打 `__created_at__` 时间戳让调用方判断 staleness；我们把时间戳留给调用方的 metadata bag（少一个硬编码字段）。

带注解的版本+阅读地图见 [`upstream-readings/s07-vector.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s07-vector.py)。

下一节 [s08 邻接图存储](s08-graph-store.md) 引入 LightRAG 真正用的第二层索引——把 entity 当节点、relation 当边。s09 会同时往 s08 的图和 s07 的 `vdbEntities` / `vdbRelations` 写，到那时三索引模式才会被端到端跑起来。
