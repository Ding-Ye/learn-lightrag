# s05 · 键值存储与过滤 / KV store with filter_keys

> 把 s01 的 `sync.Map` 推进成一个文件落地、命名空间隔离、原子重命名持久化的 `JSONKVStore`。`FilterMissing` 是这一节的核心：给一组 ID，返回还不在 store 里的子集——这就是 s09 抽取流水线问「这 50 个 chunk 哪些还没抽过实体」的那个原语。
> Promote s01's `sync.Map` into a `JSONKVStore` with namespace-scoped JSON persistence, atomic-rename writes, and per-key locking. `FilterMissing` is the centerpiece — given a list of IDs, return the subset NOT in the store. This is the primitive s09's extraction pipeline calls to ask "which of these 50 chunks haven't been extracted yet?".

## 跑起来 / Run

```bash
cd agents/s05-kv-store

# 默认 ./lightrag-data/kv_demo.json
go run .

# 自定义目录与命名空间
go run . -dir ./lightrag-data -namespace mychunks

# 跑一次写入，再跑一次确认 loaded=N（持久化生效）
go run .
go run .

# 7 个测试，含 -race；全离线
go test -race -count=1 -v ./...
```

输出会演示 5 条 Upsert → Get 还原 → 删 2 条 → `FilterMissing(["chunk-0","chunk-1","chunk-3","chunk-9","chunk-42"])` 拿到 `["chunk-1","chunk-3","chunk-9","chunk-42"]` → Persist。第二次跑会从磁盘 load 回 3 条记录。
The output walks 5 Upserts → Get round-trips → Delete 2 → `FilterMissing(["chunk-0","chunk-1","chunk-3","chunk-9","chunk-42"])` returning `["chunk-1","chunk-3","chunk-9","chunk-42"]` → Persist. A second run reloads the 3 surviving records from disk.

## 文件 / Files

| 文件 / file | 作用 / role | LOC |
|---|---|---|
| `kv_store.go` | `KVStore` 接口 + `Namespace` 别名 / interface + namespace alias | ~50 |
| `kv_json_store.go` | `JSONKVStore`：RWMutex + per-key sync.Map 锁 + 原子重命名持久化 / RWMutex + per-key sync.Map locks + atomic-rename persist | ~250 |
| `main.go` | CLI demo：upsert/get/delete/filter/persist / CLI demo for the same six methods | ~80 |
| `kv_json_store_test.go` | 7 个测试（含 race + 接口契约）/ 7 tests (race + interface contract) | ~210 |

## 关键教学点 / Key teaching points

- **`FilterMissing` 是流水线的「跳过已完成」原语 / `FilterMissing` is the pipeline's "skip-done" primitive**。它在内存里做集合差，O(N) 查 store；输入空切片返回 `[]string{}`（不是 nil）；重复的 missing ID 在结果里只出现一次，匹配 upstream `set(keys) - set(self._data.keys())` 的语义。
- **两层锁互不打架 / Two locks that don't fight each other**：粗粒度 `sync.RWMutex` 守护整张 map（写时 Lock，读时 RLock），保证 Upsert/Persist 的快照一致；细粒度 `sync.Map[string]*sync.Mutex` 让针对不同 ID 的并发 Upsert 真的并行。
- **`Persist()` 用原子重命名 / `Persist()` writes via atomic rename**：先写到 `<file>.tmp`，再 `os.Rename` 覆盖 canonical 文件。POSIX 保证同 FS 上的 rename 是原子的，reader 永远看到老文件或新文件，不会看到半截写入。崩溃后 `<file>.tmp` 残留也无害——`loadFromDisk` 只认 canonical。
- **加载只信 canonical 文件 / Load only trusts the canonical file**：哪怕 dir 里有人手工塞了 `kv_<ns>.json.tmp`（或上次 Persist 崩在 rename 之前），`NewJSONKVStore` 也只从 `kv_<ns>.json` 读。这是 `TestKVAtomicWriteOnCrash` 实测验证的不变量。

完整章节文档：[docs/zh/s05-kv-store.md](../../docs/zh/s05-kv-store.md) · [docs/en/s05-kv-store.md](../../docs/en/s05-kv-store.md)
