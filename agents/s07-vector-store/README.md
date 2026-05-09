# s07 · Cosine-similarity vector store / 余弦相似向量库

Real in-memory cosine-similarity index over `[]VectorRecord` with persistent
JSON snapshot, threshold-based filtering, and the upstream pattern of THREE
separate instances per LightRAG (`vdbChunks`, `vdbEntities`, `vdbRelations`).
Mirrors `lightrag/kg/nano_vector_db_impl.py` minus the upstream's
float16+zlib+base64 compression (discussed in the docs as an exercise).

s01 used a slice + linear scan with no thresholding, no metadata, and no
persistence. s07 promotes that placeholder to the full upstream shape.

实在的内存向量索引：基于 `[]VectorRecord` 的余弦相似度匹配 + 持久化 JSON
快照 + 阈值过滤 + 三个独立索引（chunks / entities / relations）。对应上游
`lightrag/kg/nano_vector_db_impl.py`（不实现 float16+zlib+base64 压缩，作
为练习留给读者）。s01 的 placeholder 在这里升级成完整版本。

## Run / 运行

```bash
cd /Users/yeding/learn-lightrag/agents/s07-vector-store

# CLI demo: build three indices, query each, persist to ./lightrag-data/
go run . -dir ./lightrag-data

# Re-run to see the round-trip Load (same output, no recomputation)
go run . -dir ./lightrag-data

# Tests (no network, all use t.TempDir)
go test -v ./...
```

The demo prints a top-3 readout per namespace and writes
`vdb_chunks.json` / `vdb_entities.json` / `vdb_relations.json` under `-dir`.

## Files / 文件

| File | Lines | What |
|---|---|---|
| `vector_store.go` | ~50 | `VectorStore` interface + `VectorRecord` + `VectorHit` + `Namespace` consts |
| `cosine_index.go` | ~330 | In-memory impl: cosine math, dim validation, dedup-on-Upsert, Persist/Load |
| `main.go` | ~110 | Three-namespace demo (chunks/entities/relations) with deterministic seed vectors |
| `cosine_index_test.go` | ~225 | 6 spec tests + interface contract + dim-mismatch guard |

## Tests / 测试清单

- `TestVectorStoreInterfaceContract` — compile-time guard
- `TestCosineRanksMostSimilarFirst`
- `TestCosineThresholdFiltersLowScores`
- `TestCosineRespectsTopK`
- `TestCosinePersistAcrossRestart`
- `TestCosineHandlesEmptyIndex`
- `TestCosineMetadataRoundtrip`
- `TestCosineDimensionMismatchRejected`

## Reading order / 阅读顺序

1. `docs/{zh,en}/s07-vector-store.md` — 心智模型 + ASCII 图 + 与 s01 的 diff
2. `vector_store.go` — 三个类型 + 一个接口（先看签名）
3. `cosine_index.go` — `NewCosineIndex` → `Upsert` → `Query` → `Persist`
4. `main.go` — 三个独立实例如何使用
5. `upstream-readings/s07-vector.py` — 上游 Python 对照阅读

`s06-embeddings` ships the producer of these vectors; `s09-extraction` will
write entities to `vdbEntities`, and `s11-query-modes` reads from all three
during retrieval.
