# Source: https://github.com/HKUDS/LightRAG  (paths below, branch `main`, 2026-05-09)
# License: MIT (Copyright 2025 LightRAG Team)
# Excerpted for learn-lightrag/s05-kv-store — annotated, not for execution.
#
# This file collects the upstream slice that s05 condenses into Go:
#
#   1. lightrag/kg/json_kv_impl.py (full ~150 LOC)
#        — `JsonKVStorage`, the in-memory + JSON file backend that s05
#          mirrors as `JSONKVStore`.
#   2. lightrag/base.py:308–334 (cited only)
#        — `BaseKVStorage` abstract interface; s05 re-declares it as the
#          `KVStore` Go interface.
#
# Annotations call out where s05 keeps upstream behavior verbatim, where it
# simplifies for teaching, and which later session picks up the loose ends.

# ─────────────────────────────────────────────────────────────────────────────
# (1) lightrag/kg/json_kv_impl.py — class header + initialize / upsert
# ─────────────────────────────────────────────────────────────────────────────
# Imports trimmed; this is the load-bearing slice (~50 LOC of method body).

@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""
        os.makedirs(workspace_dir, exist_ok=True)
        # → s05: NewJSONKVStore(dir, namespace) computes
        #        dir/kv_<namespace>.json (note: s05 trims `kv_store_` to `kv_`
        #        so `ls` foregrounds the namespace).
        self._file_name = os.path.join(
            workspace_dir, f"kv_store_{self.namespace}.json"
        )
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        """Initialize storage data"""
        # → s05: combined into NewJSONKVStore — Go callers always pay this
        #        cost at construction time (no separate async warmup phase).
        self._storage_lock = get_namespace_lock(self.namespace, workspace=self.workspace)
        self.storage_updated = await get_update_flag(self.namespace, workspace=self.workspace)
        async with get_data_init_lock():
            need_init = await try_initialize_namespace(self.namespace, workspace=self.workspace)
            self._data = await get_namespace_data(self.namespace, workspace=self.workspace)
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    # NB: s05 deliberately omits the cache-migration branch;
                    # see "Reading map" below.
                    self._data.update(loaded_data)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # → s05: Upsert(ctx, items). The timestamp injection (`create_time`,
        #        `update_time`, `_id`) is skipped in s05 because the s05
        #        teaching target is the *store mechanism*, not the metadata
        #        format. s09 will reinstate timestamps when the LLM cache
        #        actually needs them for staleness checks.
        if not data:
            return
        import time
        current_time = int(time.time())
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")
        async with self._storage_lock:
            for i, (k, v) in enumerate(data.items(), start=1):
                if self.namespace.endswith("text_chunks"):
                    if "llm_cache_list" not in v:
                        v["llm_cache_list"] = []
                if k in self._data:
                    v["update_time"] = current_time
                else:
                    v["create_time"] = current_time
                    v["update_time"] = current_time
                v["_id"] = k
                await _cooperative_yield(i)
            self._data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)


# ─────────────────────────────────────────────────────────────────────────────
# (2) lightrag/kg/json_kv_impl.py — get_by_id / filter_keys / delete / persist
# ─────────────────────────────────────────────────────────────────────────────

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        # → s05: Get(ctx, id) — returns (rec, true, nil) or (nil, false, nil).
        #        Go reports presence via the boolean instead of None.
        async with self._storage_lock:
            result = self._data.get(id)
            if result:
                result = dict(result)  # → s05: copyMap(rec) defensive copy
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                result["_id"] = id
            return result

    async def filter_keys(self, keys: set[str]) -> set[str]:
        # → s05's centerpiece: FilterMissing(ctx, ids) returns []string.
        #    Go has no set type, so we hand-roll a seen-map for dedup
        #    inside the loop. Empty input -> empty (non-nil) slice.
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def delete(self, ids: list[str]) -> None:
        # → s05: Delete(ctx, ids). Missing ids are silently ignored —
        #        upstream's pop(default=None) does the same.
        async with self._storage_lock:
            for doc_id in ids:
                self._data.pop(doc_id, None)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def index_done_callback(self) -> None:
        # → s05: Persist(ctx). Upstream's write_json wraps the same
        #    "tmp + os.replace" pattern that s05 inlines as:
        #       os.WriteFile(canonical+".tmp", buf, 0o644)
        #       os.Rename(canonical+".tmp", canonical)
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = dict(self._data)
                needs_reload = write_json(data_dict, self._file_name)
                if needs_reload:  # sanitization happened — reload cleaned bytes
                    cleaned_data = load_json(self._file_name)
                    if cleaned_data is not None:
                        self._data.clear()
                        self._data.update(cleaned_data)
                await clear_all_update_flags(self.namespace, workspace=self.workspace)


# ─────────────────────────────────────────────────────────────────────────────
# Reading map — where this code re-appears in later sessions
# ─────────────────────────────────────────────────────────────────────────────
#
#   * s03 (doc-status) already implements a near-twin of `index_done_callback`
#     in `doc_status_store.go::Persist`.  s05 generalizes that pattern: any
#     "string -> structured record" persistence in s09/s10/s11 uses the same
#     atomic-rename JSON write.
#
#   * s09 (extraction) is the FIRST consumer of `FilterMissing`.  Its core
#     extraction loop boils down to:
#
#         missing := llmCache.FilterMissing(ctx, chunkIDs)
#         for _, id := range missing { extractEntities(id) }
#
#     Without this primitive, s09 would either re-extract every chunk on
#     each restart (wasting LLM calls) or hand-roll the set-difference at
#     every call site.
#
#   * Phase G (Postgres backend, exercise list in plan.md) drops in a
#     `PostgresKVStore` that satisfies the same `KVStore` interface.  Tests
#     written against `JSONKVStore` should pass against the Postgres impl
#     verbatim — the interface is the contract.
#
#   * `BaseKVStorage` (lightrag/base.py:308) declares the abstract methods
#     `get_by_id`, `get_by_ids`, `upsert`, `filter_keys`, `delete`,
#     `drop`, plus a few cache-specific helpers s05 deliberately skips.
#     The Go `KVStore` interface in `kv_store.go` is a 1:1 (minus the
#     cache helpers) port.
#
# ─────────────────────────────────────────────────────────────────────────────
# What s05 deliberately does NOT do
# ─────────────────────────────────────────────────────────────────────────────
#
#   * No legacy cache migration (`_migrate_legacy_cache_structure`):
#     production-deployment baggage, not core to KV-store mechanics.
#
#   * No timestamp / `_id` injection inside Upsert: the cache-staleness
#     workflow that needs these lives in s09.
#
#   * No `index_done_callback`-style update flag: in upstream, multiple
#     processes share the file via flag coordination.  s05 is single-process
#     teaching code; the explicit `Persist()` call is the only write moment.
#
#   * No `drop()` method: trivially `os.Remove(filePath()) + reset map`,
#     left as a one-liner for the learner to add.
