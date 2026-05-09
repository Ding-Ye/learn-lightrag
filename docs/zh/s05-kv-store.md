---
title: "s05 · 键值存储与过滤"
chapter: 5
slug: s05-kv-store
est_read_min: 9
---

# s05 · 键值存储与过滤

> 教什么：把 s01 占位的 `sync.Map` 推进成上游 LightRAG 真正在用的 `JsonKVStorage`——一个文件落地、命名空间隔离、原子重命名持久化、per-key 加锁的键值存储；最关键的是引入 `FilterMissing` 原语（上游叫 `filter_keys`），s09 抽取流水线靠它跳过已抽过实体的 chunk，避免一份文档每次 reingest 都重复打 LLM。

---

## Problem / 问题

s01 的 `kv.go` 用 `sync.Map` 做了一个三十行的占位实现，实际撑不住三件事：

1. **重启即归零**：进程退出后 `sync.Map` 里的 chunk → 抽取结果映射全没了。下次 `go run .` 又得对每个 chunk 重新打一次 LLM——抽实体一次、归并 description 一次、缓存最终回答一次，三个 namespace 都重来。OpenAI 抽 100 个 chunk 大概 $1，没有持久化等于把成本翻三倍。
2. **没有批量原语**：s09 想问的是「这 50 个 chunk ID 里，哪些还没在 `llm_response_cache` 里？」——这是一个集合差操作，`sync.Map` 只能 50 次 `Load` 然后自己拼结果。上游 `BaseKVStorage.filter_keys` 一次调用就给答案，下游代码读起来还是「业务语言」而不是「容器操作」。
3. **没有命名空间隔离**：上游同一个 working_dir 下有 `kv_store_full_docs.json`、`kv_store_text_chunks.json`、`kv_store_llm_response_cache.json` 三个文件——三类数据的更新频率、保留策略、调试需求都不一样。`sync.Map` 撑死只能塞一份，多类数据混在一起。

s05 的解法直接对照上游 `lightrag/kg/json_kv_impl.py`：每个 namespace 一个 JSON 文件，磁盘布局映射 `./lightrag-data/kv_<namespace>.json`；in-memory map + RWMutex + per-key 锁；`Persist()` 走原子重命名；最重要的——补齐 `FilterMissing` 这个 s09 会调爆的原语。

## Solution / 解决方案

四件事：

1. **`KVStore` 接口照搬上游 `BaseKVStorage` 的可用子集**：`Get` / `GetByIDs` / `Upsert` / `FilterMissing` / `Delete` / `Persist`，每个方法第一个参数都是 `context.Context`（取代上游 asyncio 的协作让权）。这跟 plan.md `Shared types catalog` 里钉死的签名完全一致，所以 s09 把 store 接进来不用改 caller 代码。
2. **`JSONKVStore` 是一个 in-memory map + 两层锁的并发实现**：`map[string]map[string]any` 是真值，`sync.RWMutex` 保护整张 map（Upsert/Delete/Persist 走 Lock，Get/GetByIDs/FilterMissing 走 RLock），`sync.Map[string]*sync.Mutex` 给每个 key 一把懒分配的细锁，让针对不同 ID 的并发 Upsert 真的并行。两把锁的契合点在 Upsert：先在 per-key 锁下做防御性拷贝（防止 caller 在 Upsert 返回后还能改我们的内存），再在全局写锁下一次性 commit 所有改动，对读者来说要么全见要么全不见。
3. **`Persist()` 是原子重命名**：写到 `<dir>/kv_<namespace>.json.tmp`，`os.Rename` 覆盖 canonical 文件。POSIX 保证同 FS 上 rename 是原子的——reader 永远看到旧文件或新文件，不会看到半截写入。崩溃留下的 `.tmp` 由下次启动时 `loadFromDisk` 主动忽略（它只读 canonical 名）。
4. **`FilterMissing` 是 s09 的「跳过已完成」原语**：内存里 O(N) 集合差。空切片返回 `[]string{}`（不是 nil，调用者直接 `len()` 就行）；输入里有重复的 missing ID 只出现一次。这两个性质 `TestKVFilterMissingPartial` 都断言到了。

## How It Works / 工作原理

```
NewJSONKVStore("./lightrag-data", "demo")
        │
        ▼
   ./lightrag-data/
   ├── kv_demo.json          ← canonical（reader 唯一信任源）
   └── kv_demo.json.tmp      ← Persist 中途存在；rename 后消失

Upsert(ctx, items)             Persist(ctx)
    │                              │
    ├─ for each id:                ├─ snapshot = deep-copy(data) under RLock
    │     keyLock(id).Lock()       ├─ marshal indent
    │     staged[id] = copy(rec)   ├─ WriteFile("kv_<ns>.json.tmp")
    │     keyLock(id).Unlock()     └─ os.Rename(tmp, canonical)
    └─ s.mu.Lock()                       └─ POSIX-atomic
       data[id] = staged[id]                  │
       s.mu.Unlock()                          ▼
                                       reader 永远看到旧文件 OR 新文件
```

载货核心 30 行（节选自 [`agents/s05-kv-store/kv_json_store.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s05-kv-store/kv_json_store.go) 的 `Upsert` 与 `Persist`）：

```go
func (s *JSONKVStore) Upsert(ctx context.Context, items map[string]map[string]any) error {
    if err := ctx.Err(); err != nil { return err }
    if len(items) == 0 { return nil }

    // Stage copies under per-key locks so concurrent Upsert(same-key) is
    // race-free without serializing the whole store.
    staged := make(map[string]map[string]any, len(items))
    for id, rec := range items {
        lk := s.keyLock(id)
        lk.Lock()
        staged[id] = copyMap(rec)
        lk.Unlock()
    }

    s.mu.Lock()
    defer s.mu.Unlock()
    for id, rec := range staged {
        s.data[id] = rec
    }
    return nil
}

func (s *JSONKVStore) Persist(ctx context.Context) error {
    s.mu.RLock()
    snapshot := make(map[string]map[string]any, len(s.data))
    for k, v := range s.data { snapshot[k] = copyMap(v) }
    s.mu.RUnlock()

    buf, err := json.MarshalIndent(snapshot, "", "  ")
    if err != nil { return err }

    canonical := s.filePath()
    tmp := canonical + ".tmp"
    if err := os.WriteFile(tmp, buf, 0o644); err != nil { return err }
    return os.Rename(tmp, canonical)  // POSIX atomic on the same FS
}
```

四个不显眼但很重要的点：

- **加载只读 canonical 名**：`loadFromDisk` 故意不 stat `.tmp`。`TestKVAtomicWriteOnCrash` 故意写一份残缺的 `.tmp` 进去，断言新 store 不会把毒数据 load 进来。崩溃恢复的语义是「丢掉上次没写完的」，不是「尽力还原」。
- **per-key 锁是 lazily 创建**：`sync.Map.LoadOrStore` 保证哪怕首次访问的两个 goroutine 撞上，也只会有一把 `*sync.Mutex` 真的进表。store 生命周期内 key 数和锁数 1:1，绝不重复分配。
- **defensive copy 在两个方向都做**：Upsert 入口拷贝 caller 的 map，Get/GetByIDs 出口也拷贝——这两端任意一端不拷，caller 都能从外面改我们的内存。`TestKVUpsertGetRoundTrip` 显式验证返回值被改不会污染下次 Get。
- **`FilterMissing` 用本地 `seen` 去重**：上游 Python 用 `set(keys) - set(self._data.keys())` 天然去重，Go 没有内建 set，所以我们手撸一个 `map[string]struct{}` 在循环里跳过重复 ID。返回切片永远是 non-nil 的 `[]string{}`，省去调用方 `if missing != nil` 的判断。

## What Changed / 与 s01 的变化

| 维度 / aspect | s01 (`kv.go`) | s05 (`kv_json_store.go`) |
|---|---|---|
| 存储 / backing store | `sync.Map` (内存) | `map[string]map[string]any` + JSON 文件 |
| 命名空间 / namespaces | 无（混在一起） | 一 namespace 一文件 (`kv_<ns>.json`) |
| 持久化 / persistence | 重启即丢 | `Persist()` 原子重命名 + `loadFromDisk` 重启加载 |
| 批量读 / batch read | 多次 `Load` 自拼 | `GetByIDs` 一次返回命中集合 |
| **跳过已完成** / **skip-done** | 调用方手写循环 | `FilterMissing(ids)` 一次集合差 |
| 删除 / delete | `Delete(key)` 单条 | `Delete([]string)` 批量 |
| 并发 / concurrency | `sync.Map` 单层 | RWMutex + per-key sync.Map 双层 |
| 防御性拷贝 / defensive copy | 无 | Upsert 入参 / Get 返回值都拷一次 |

最关键的 diff 是 `FilterMissing`——s09 的伪代码直接从「拿出 50 个 chunk_id，逐个查 cache」一行变成 `missing := cache.FilterMissing(ctx, chunkIDs)`。读起来还是业务语言（「这些里哪些没抽过」），不是容器操作（「Load 50 次然后自己 append」）。s03 的 `DocStatusStore` 之前自己手撸了一份 JSON 持久化代码，s05 的 `JSONKVStore` 上线后，s09/s10/s11 凡是「字符串 → 结构化数据」的存储需求都用同一份接口装，文件名只换 namespace。

## Try It / 动手试一试

```bash
cd agents/s05-kv-store

# 一次插入 5 条 + Delete 2 条 + FilterMissing + Persist
go run .

# 再跑一次：第二次输出里 loaded = 3，证明上一次 Persist 生效
go run .

# 看一眼磁盘上的 JSON——namespace 隔离 + 缩进格式
cat ./lightrag-data/kv_demo.json
ls -la ./lightrag-data/   # 应该看不到 kv_demo.json.tmp

# 7 个测试 + race detector，全离线
go test -race -count=1 -v ./...
```

测试矩阵覆盖六条性质 + 一个接口契约：`TestKVUpsertGetRoundTrip`（往返一致）、`TestKVFilterMissingPartial`（集合差正确，含空入参与重复 ID）、`TestKVPersistAcrossRestart`（落盘后新 store load 回相同数据，且 `.tmp` 不残留）、`TestKVConcurrentUpsertSafe`（100 goroutine × 10 写不打架，最终 1000 条）、`TestKVAtomicWriteOnCrash`（手工塞毒 `.tmp` 不会污染下次 load）、`TestKVDeleteRemovesKey`（删了再 Get miss、删不存在的 key 不报错）、`TestKVStoreInterfaceContract`（编译期接口契约）。

## Upstream Source Reading / 上游源码阅读

下面是 [`lightrag/kg/json_kv_impl.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/json_kv_impl.py) 的核心节选——load / upsert / filter_keys / index_done_callback（即上游的 Persist 入口）。注释指向 s05 对应的 Go 实现：

```python
@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        # → s05: NewJSONKVStore(dir, namespace) 拼出 dir/kv_<ns>.json
        working_dir = self.global_config["working_dir"]
        workspace_dir = (
            os.path.join(working_dir, self.workspace) if self.workspace
            else working_dir
        )
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(
            workspace_dir, f"kv_store_{self.namespace}.json"
        )
        self._data = None
        self._storage_lock = None

    async def initialize(self):
        # → s05: NewJSONKVStore 构造时同步 loadFromDisk();
        #        Go 不用 asyncio 协作锁，直接 sync.RWMutex.
        async with get_data_init_lock():
            need_init = await try_initialize_namespace(self.namespace, ...)
            self._data = await get_namespace_data(self.namespace, ...)
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # → s05: Upsert(ctx, items)；时间戳与 _id 注入这里没翻译,
        #        s09 真用到 cache 时再补.
        if not data: return
        async with self._storage_lock:
            for k, v in data.items():
                if k in self._data:
                    v["update_time"] = current_time
                else:
                    v["create_time"] = current_time
                    v["update_time"] = current_time
                v["_id"] = k
            self._data.update(data)
            await set_all_update_flags(self.namespace, ...)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        # → s05 的核心: FilterMissing(ctx, ids) 在 Go 里返回 []string,
        #    用本地 seen-map 自己去重.
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def index_done_callback(self) -> None:
        # → s05: Persist(ctx) — 上游通过 write_json 实现原子重命名,
        #    我们直接 os.WriteFile + os.Rename, 一致.
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = dict(self._data)
                write_json(data_dict, self._file_name)
                await clear_all_update_flags(self.namespace, ...)
```

**为什么 s05 故意没翻 `_migrate_legacy_cache_structure`**：上游为了向后兼容旧版 cache 文件格式，加了一段 `_cache` namespace 专用的扁平化逻辑。这是「真实生产部署历史包袱」，不是「KV store 的本质能力」——s05 教学目标是 KV store 本身，所以丢掉这段，等 s09 真的引入 LLM cache 时再决定要不要补。

**为什么没用 asyncio 而是 RWMutex**：上游每个 namespace 一把 `NamespaceLock`，配合 `_cooperative_yield` 让长循环能让出 CPU。Go 的 goroutine 调度由 runtime 抢占，不需要应用层主动让权，所以 RWMutex 这层就够了。这是 plan.md 里 risk #6 提到的「async semantics gap」的具体落地。

**完整上游源码 + 注解**：见 [`upstream-readings/s05-kv.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s05-kv.py)，包含 reading-map 指出 s03（doc-status 用了同款 JSON 持久化）、s09（LLM-response cache 直接复用 s05 的 store）和 Phase G（PostgreSQL 后端）三个延伸阅读点。
