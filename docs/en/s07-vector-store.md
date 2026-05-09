---
title: "s07 · Cosine-similarity vector store"
chapter: 7
slug: s07-vector-store
est_read_min: 9
---

# s07 · Cosine-similarity vector store

> What this teaches: promote s01's "slice + linear cosine scan" placeholder into a real `VectorStore`. Cosine similarity over `[]VectorRecord`, threshold-based filtering, atomic JSON persistence, and the upstream pattern of THREE separate indices (`vdbChunks`, `vdbEntities`, `vdbRelations`) feeding the same impl. Tracks `lightrag/kg/nano_vector_db_impl.py` minus the upstream's `float16 + zlib + base64` compression — discussed but deliberately not implemented (exercise at the end).

---

## Problem / 问题

s01's vector store was, charitably, a single-pass demo:

```go
// s01/vectorstore.go (simplified)
type vstore struct{ recs []record }

func (s *vstore) Query(q []float32, k int) []record {
    // Linear scan, no threshold, no metadata, no persistence.
    scored := make([]record, 0, len(s.recs))
    for _, r := range s.recs {
        score := dot(q, r.vec) / (norm(q) * norm(r.vec))
        scored = append(scored, record{r.id, r.vec, score})
    }
    sort.Slice(scored, ...)
    return scored[:k]
}
```

Four things break the moment you try to scale past 50 chunks:

1. **No thresholding.** Even when *every* record scores 0.05 — i.e. nothing is genuinely similar — the store returns "top-k" of garbage. Upstream's `cosine_better_than_threshold` (typically 0.2-0.4) cuts the bottom off so retrieval doesn't poison the LLM context with noise.
2. **No persistence.** Restart the process, the index evaporates. s01's pipeline re-embeds every chunk on every run, costing 100× the OpenAI bill it should.
3. **One bag for everything.** Real LightRAG keeps THREE distinct indices — chunks, entities, relations — and queries them at different stages of retrieval. With one shared store the "top-k entities matching this keyword" call has to filter by metadata after-the-fact, which is both slow and error-prone.
4. **No metadata round-trip.** When `Query` returns IDs only, you have to do a second store lookup to fetch the chunk text. Upstream attaches metadata (chunk content, entity name, etc.) directly on each record so retrieval is one round trip.

s07 fixes all four by porting `lightrag/kg/nano_vector_db_impl.py:79-150` — the `upsert()` and `query()` methods of the `NanoVectorDBStorage` class.

## Solution / 解决方案

Four moves:

1. **`VectorStore` interface frozen from plan.md.** Four methods (`Upsert`, `Query`, `Delete`, `Persist`) with `context.Context` first arg. `VectorRecord` carries `ID`, `Vector []float32`, `Metadata map[string]any`; `VectorHit` adds `Score float32`. The `Namespace` constants (`chunks`, `entities`, `relations`) make the upstream three-index pattern visible at the type level.
2. **`CosineIndex` is an in-memory `[]VectorRecord` with a fast `id → index` map.** Upsert dedups by ID in O(1), Delete splices out and rebuilds the map. The first record establishes the `dim`; subsequent inserts with the wrong-length vector are rejected so silent dimension drift never corrupts the index.
3. **`Query(query, topK, threshold)` is one `O(N · dim)` pass with descending sort + topK truncation.** Threshold is INCLUSIVE — a hit exactly at threshold is kept, matching upstream's `better_than_threshold` semantic. Empty index returns `[]VectorHit{}` (never nil) with no error.
4. **`Persist(ctx)` is the same atomic-rename dance as s05's KV store.** Marshal in-memory state to `<file>.tmp`, then `os.Rename` to `<dir>/vdb_<namespace>.json`. POSIX guarantees rename atomicity on the same filesystem, so a reader either sees the full old file or the full new file. `NewCosineIndex` auto-loads from disk if the file exists; missing file means we start fresh (NOT an error).

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

Cosine math:

```
                    a · b
    cos(a, b) = ──────────────
                  |a| * |b|

    where  a · b   = sum(a[i] * b[i])
           |a|     = sqrt(sum(a[i]^2))
```

We compute `|query|` once outside the loop and `|rec|` inline — no full re-norm per record. If either norm is zero we return `0` (skipping div-by-zero); upstream raises, we'd rather degrade gracefully so a single all-zero record can't 500 the whole query.

Load-bearing 30 lines from [`agents/s07-vector-store/cosine_index.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s07-vector-store/cosine_index.go), `Query`:

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

**Four non-obvious points:**

1. **`|query|` is computed once, not per record.** The naive code computes `cosine = dot(a,b) / (norm(a)*norm(b))` inside the loop, but `norm(a)` doesn't depend on `b`. Pulling it out cuts the constant factor by ~33% on typical workloads. Upstream's NanoVectorDB uses `numpy.linalg.norm(query)` once with the same logic.
2. **Tie-break by ID for determinism.** Two records with identical scores (rare but possible at threshold edges) get sorted alphabetically. Without this, test output isn't reproducible — the Go runtime's `sort` is not stable, and even if it were, map iteration order in Go is randomized so insertion order isn't preserved either. The spec test `TestCosinePersistAcrossRestart` asserts byte-identical hits across two index instances, which is only achievable with a deterministic tie-break.
3. **Defensive copy on `Metadata` egress.** A caller mutating `hit.Metadata["foo"] = "tampered"` mustn't poison the index for the next query. `TestCosineMetadataRoundtrip` mutates the returned hit and re-queries to assert the original value comes back unchanged. Same defensive copy applied on Upsert ingress (so caller mutating their input map post-call doesn't leak in either direction).
4. **First record locks the dimension; mismatched vectors are rejected.** Upstream silently lets numpy raise on the eventual matrix op; we surface a typed `ErrDimensionMismatch` at the Upsert boundary so the failure points at the offending record's ID. `TestCosineDimensionMismatchRejected` asserts the rejection AND that the index size is unchanged after the bad insert.

## What Changed / 与 s01 的变化

| Dimension | s01 (placeholder) | s07 (real chapter) |
|---|---|---|
| Storage | inline `[]record` in `vectorstore.go` | dedicated `CosineIndex` struct with id→index map |
| Indices | one bag for everything | three separate instances (chunks / entities / relations) |
| Threshold | none | `threshold float32` arg; below-threshold hits dropped |
| Persistence | none | atomic-rename to `<dir>/vdb_<namespace>.json` |
| Auto-load | n/a | `NewCosineIndex` auto-loads if file exists |
| Metadata | none | `map[string]any` per record, defensively copied on egress |
| Dim check | crash on mismatched len(vec) | typed `ErrDimensionMismatch` at Upsert |
| Empty index | undefined | `[]VectorHit{}` + nil error |
| Tie-break | sort instability | descending score, ascending ID for determinism |

The three-index move is the architectural win: it lets s11 ask "top-k entities matching the query keywords" without scanning chunks or relations. The cost is that `Insert` now has to write to two stores (vdbChunks for the embed; vdbEntities/vdbRelations after extraction in s09).

## Try It / 动手试一试

```bash
cd /Users/yeding/learn-lightrag/agents/s07-vector-store

# 1. Build all three indices, persist to ./lightrag-data/
go run . -dir ./lightrag-data

# 2. Re-run to confirm load round-trip (output should be identical)
go run . -dir ./lightrag-data

# 3. Inspect what got written
ls -la ./lightrag-data/
cat ./lightrag-data/vdb_chunks.json | head -30

# 4. Run all 8 tests (no network, all use t.TempDir)
go test -v ./...
```

The demo output:

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

Test matrix:

| Test | Asserts |
|---|---|
| `TestVectorStoreInterfaceContract` | compile-time guard: `*CosineIndex` satisfies `VectorStore` |
| `TestCosineRanksMostSimilarFirst` | querying with the same vector as a record → that record is rank-1 with score ≈ 1.0 |
| `TestCosineThresholdFiltersLowScores` | `threshold=0.99` keeps only near-identical matches |
| `TestCosineRespectsTopK` | 10 records, `topK=3` → exactly 3 hits |
| `TestCosinePersistAcrossRestart` | Persist → new instance same path → identical hits including metadata |
| `TestCosineHandlesEmptyIndex` | empty `Query` returns `[]VectorHit{}` (not nil), no error |
| `TestCosineMetadataRoundtrip` | upsert with `{"foo":"bar"}` survives Query, caller mutation does NOT leak back |
| `TestCosineDimensionMismatchRejected` | second insert with wrong-length vector rejected, Len unchanged |

## Upstream Source Reading / 上游源码阅读

```python
# lightrag/kg/nano_vector_db_impl.py:79-150 (excerpt; full file ~270 lines)
# Two methods, both async, both wrapped by the per-namespace asyncio lock.

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

**How to read it:**

- `upsert` runs the embedding INSIDE the storage method — upstream couples "embed + insert" because the storage owns the `embedding_func` reference. Our s07 split: the embedder lives in s06, `Upsert` accepts already-embedded vectors. The seam at `[]VectorRecord` is the same shape as upstream's `list_data`.
- The `vector_f16 + zlib + base64` triple is the upstream's storage-size optimization (~25-50% smaller on disk vs raw float32 JSON). We omit it for code transparency; the JSON files this chapter writes are larger but trivially diff-able. **Exercise:** add a `WithCompression()` option that re-implements upstream's encoding.
- `better_than_threshold` is the same INCLUSIVE filter we ship — `score >= threshold` keeps the record. Default is `0.2`-`0.4` depending on workspace.
- Upstream stamps `__created_at__` on every record so callers can reason about staleness; we leave timestamps to the caller's metadata bag (one less hardcoded field).

A more annotated version with reading-map: [`upstream-readings/s07-vector.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s07-vector.py).

The next chapter, [s08 adjacency graph store](s08-graph-store.md), introduces the second indexing layer LightRAG actually uses — entities and relations as nodes/edges. s09 will then write to BOTH the s08 graph and s07's `vdbEntities` / `vdbRelations`, finally exercising the three-index pattern end-to-end.
