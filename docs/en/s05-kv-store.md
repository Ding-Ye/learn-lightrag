---
title: "s05 · KV store with filter_keys"
chapter: 5
slug: s05-kv-store
est_read_min: 9
---

# s05 · KV store with filter_keys

> What this teaches: promote s01's placeholder `sync.Map` into the upstream `JsonKVStorage` shape — a file-backed, namespace-isolated, atomic-rename-persisted, per-key-locked key/value store. The new primitive is `FilterMissing` (upstream's `filter_keys`); s09's extraction pipeline calls it on every chunk batch to skip work that has already been done, so a re-ingest never pays the LLM bill twice.

---

## Problem / 问题

s01's `kv.go` is a 30-line `sync.Map` placeholder that breaks down on three counts:

1. **Restart wipes everything.** The chunk → extraction map evaporates when the process exits. The next `go run .` re-bills every chunk to the LLM (extraction once, description summarization once, final-answer cache once — three namespaces all redone). On a 100-chunk doc that's $3 worth of OpenAI calls every restart, when persistence would make it $1.
2. **No batch primitive.** s09 wants to ask "of these 50 chunk IDs, which are NOT yet in the LLM cache?" — that's a set-difference. With `sync.Map` the caller does 50 `Load` calls and stitches the result by hand. Upstream's `BaseKVStorage.filter_keys` is one call; the call site reads as business intent ("skip the ones that are done") instead of as container plumbing.
3. **No namespace isolation.** Upstream's working dir contains `kv_store_full_docs.json`, `kv_store_text_chunks.json`, `kv_store_llm_response_cache.json` — three flavours of data with different update cadences, retention policies, and debugging needs. A single `sync.Map` jumbles them together; filename-level isolation makes `ls ./rag_storage` self-documenting.

s05 is a near-line-by-line port of `lightrag/kg/json_kv_impl.py`: one JSON file per namespace under `./lightrag-data/kv_<namespace>.json`, in-memory map plus RWMutex plus per-key locks, atomic-rename `Persist()`, and — crucially — the `FilterMissing` primitive that s09 will lean on hard.

## Solution / 解决方案

Four moves:

1. **`KVStore` interface mirrors the usable subset of `BaseKVStorage`.** Six methods: `Get` / `GetByIDs` / `Upsert` / `FilterMissing` / `Delete` / `Persist`, each taking `context.Context` (replacing upstream's asyncio cooperative-yield). The signatures match plan.md's locked types catalog so s09 can plug `JSONKVStore` in without changing any caller.
2. **`JSONKVStore` is an in-memory map guarded by two layers of locks.** The truth is `map[string]map[string]any`. A coarse `sync.RWMutex` guards the whole map (Upsert/Delete/Persist take Lock, Get/GetByIDs/FilterMissing take RLock). A `sync.Map` of `*sync.Mutex` keyed by record id provides per-key fine-grained locks so concurrent Upserts targeting DIFFERENT ids genuinely run in parallel; the per-key mutex is created lazily via `LoadOrStore` so even a first-touch race produces exactly one mutex per key. The two locks meet in `Upsert`: stage defensive copies under per-key locks (so the caller can't mutate our state through their input map), then commit them under one brief global write-lock so readers see all-or-nothing.
3. **`Persist()` writes via atomic rename.** Marshal to `<dir>/kv_<namespace>.json.tmp`, then `os.Rename` over the canonical name. POSIX guarantees a same-FS rename is atomic — readers either see the full old file or the full new file, never a half-written intermediate. A `.tmp` left behind by a crashed run is benign because `loadFromDisk` only ever reads the canonical name.
4. **`FilterMissing` is the s09 "skip-done" primitive.** O(N) set-difference in memory. Empty input returns `[]string{}` (not nil — callers can `len()` immediately). Duplicate input ids appear at most once in the result set, mirroring the upstream `set(keys) - set(self._data.keys())` semantic. `TestKVFilterMissingPartial` asserts both invariants.

## How It Works / 工作原理

```
NewJSONKVStore("./lightrag-data", "demo")
        │
        ▼
   ./lightrag-data/
   ├── kv_demo.json          ← canonical (the only file Load trusts)
   └── kv_demo.json.tmp      ← exists transiently inside Persist; gone after rename

Upsert(ctx, items)             Persist(ctx)
    │                              │
    ├─ for each id:                ├─ snapshot = deep-copy(data) under RLock
    │     keyLock(id).Lock()       ├─ marshal indent
    │     staged[id] = copy(rec)   ├─ WriteFile("kv_<ns>.json.tmp")
    │     keyLock(id).Unlock()     └─ os.Rename(tmp, canonical)
    └─ s.mu.Lock()                       └─ POSIX-atomic
       data[id] = staged[id]                  │
       s.mu.Unlock()                          ▼
                                       readers see old file OR new file
```

Load-bearing 30 lines (excerpted from [`agents/s05-kv-store/kv_json_store.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s05-kv-store/kv_json_store.go), `Upsert` and `Persist`):

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

Four small but load-bearing details:

- **Load only trusts the canonical name.** `loadFromDisk` deliberately never `stat`s `.tmp`. `TestKVAtomicWriteOnCrash` writes a poisoned `.tmp` into the dir and asserts that the new store does NOT pull it in. The crash-recovery contract is "drop the half-written write", not "best-effort recover".
- **Per-key locks are created lazily via `sync.Map.LoadOrStore`.** Even if two goroutines first-touch the same key concurrently, only ONE `*sync.Mutex` ever lands in the map. Lock count = key count over the store's lifetime; no double-allocation.
- **Defensive copies on both edges.** Upsert copies the caller's records on the way in; Get/GetByIDs copies on the way out. Skip either side and the caller can mutate our state from outside. `TestKVUpsertGetRoundTrip` explicitly mutates a returned record and asserts the next Get still sees the original.
- **`FilterMissing` dedups via a local `seen` set.** Upstream Python gets dedup for free from `set(keys)`; Go has no built-in set, so we hand-roll a `map[string]struct{}` inside the loop. The returned slice is always a non-nil `[]string{}`, sparing the caller a `if missing != nil` branch.

## What Changed / 与 s01 的变化

| aspect / 维度 | s01 (`kv.go`) | s05 (`kv_json_store.go`) |
|---|---|---|
| backing store | `sync.Map` (in-memory only) | `map[string]map[string]any` + JSON file |
| namespaces | none (everything in one place) | one namespace per file (`kv_<ns>.json`) |
| persistence | gone on restart | `Persist()` atomic rename + `loadFromDisk` on construct |
| batch read | hand-loop `Load` calls | `GetByIDs` returns the hit map directly |
| **skip-done** | caller writes the loop | `FilterMissing(ids)` returns the missing subset |
| delete | `Delete(key)` single | `Delete([]string)` batch |
| concurrency | single `sync.Map` | RWMutex + per-key sync.Map (two layers) |
| defensive copy | none | both Upsert input and Get output are copied |

The headline diff is `FilterMissing` — s09's pseudocode collapses from "load 50 ids one by one and check the misses" to `missing := cache.FilterMissing(ctx, chunkIDs)`. The call reads as business intent ("which of these still need work") rather than container manipulation. s03's `DocStatusStore` previously hand-rolled its own JSON persistence; once `JSONKVStore` ships, every "string → struct" storage in s09/s10/s11 plugs into the same interface — only the namespace name changes per use site.

## Try It / 动手试一试

```bash
cd agents/s05-kv-store

# 5 Upserts + Delete 2 + FilterMissing + Persist
go run .

# Run again: the second invocation prints `loaded = 3`, proving Persist worked
go run .

# Inspect the on-disk JSON: namespaced filename + indented body
cat ./lightrag-data/kv_demo.json
ls -la ./lightrag-data/   # the .tmp must NOT be there

# 7 tests + race detector, fully offline
go test -race -count=1 -v ./...
```

The test matrix covers six properties + one interface contract: `TestKVUpsertGetRoundTrip` (round-trip equality), `TestKVFilterMissingPartial` (set-difference correctness, empty input, duplicate ids), `TestKVPersistAcrossRestart` (reload via a fresh store recovers all records and `.tmp` does not linger), `TestKVConcurrentUpsertSafe` (100 goroutines × 10 writes don't race; final Len = 1000), `TestKVAtomicWriteOnCrash` (a hand-seeded poisoned `.tmp` does NOT bleed into the next load), `TestKVDeleteRemovesKey` (Get misses after Delete; deleting a missing id is a silent no-op), `TestKVStoreInterfaceContract` (compile-time interface check).

## Upstream Source Reading / 上游源码阅读

The annotated extract below is the load-bearing slice of [`lightrag/kg/json_kv_impl.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/json_kv_impl.py) — load / upsert / filter_keys / index_done_callback (upstream's Persist entry-point). Comments point at the matching Go in s05.

```python
@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        # → s05: NewJSONKVStore(dir, namespace) builds dir/kv_<ns>.json
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
        # → s05: NewJSONKVStore loads synchronously via loadFromDisk;
        #        Go uses sync.RWMutex instead of asyncio cooperative locks.
        async with get_data_init_lock():
            need_init = await try_initialize_namespace(self.namespace, ...)
            self._data = await get_namespace_data(self.namespace, ...)
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # → s05: Upsert(ctx, items). Timestamp + _id injection skipped —
        #        s09 will add it back when it actually wires up cache writes.
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
        # → s05's centerpiece: FilterMissing(ctx, ids) returns []string;
        #    we hand-roll a seen-map for dedup since Go has no set type.
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def index_done_callback(self) -> None:
        # → s05: Persist(ctx). Upstream's write_json does atomic rename
        #    internally; we do os.WriteFile + os.Rename inline. Same shape.
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = dict(self._data)
                write_json(data_dict, self._file_name)
                await clear_all_update_flags(self.namespace, ...)
```

**Why s05 deliberately drops `_migrate_legacy_cache_structure`.** Upstream carries a `_cache`-namespace-only flatten step for backward-compat with older cache file layouts. That's "real-deployment legacy baggage", not "the essence of a KV store" — s05's teaching target is the store itself, so we omit it. s09 can re-introduce migration logic if and when the LLM-response cache actually ships.

**Why we use `sync.RWMutex` instead of asyncio.** Upstream pairs each namespace with a `NamespaceLock` and sprinkles `_cooperative_yield` calls so long loops don't starve other coroutines. Go goroutines are pre-emptively scheduled by the runtime — no application-level yielding required, so an RWMutex covers it. This is the concrete landing-spot for plan.md's risk #6 ("async semantics gap").

**Full upstream source + annotations**: see [`upstream-readings/s05-kv.py`](https://github.com/Ding-Ye/learn-lightrag/blob/main/upstream-readings/s05-kv.py), which includes a reading-map pointing at s03 (doc-status uses the same JSON-persistence pattern), s09 (LLM-response cache plugs into the s05 store directly), and Phase G (a Postgres backend would implement the same interface).
