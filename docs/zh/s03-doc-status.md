---
title: "s03 · 文档状态机"
chapter: 3
slug: s03-doc-status
est_read_min: 9
---

# s03 · 文档状态机

> 教什么：把文档生命周期 PENDING → PROCESSING → PROCESSED/FAILED 抽出来做成一个独立的、可持久化的、能转移校验的 `DocStatusStore`。配合 MD5 内容寻址 docID，「重复插入同一份文档自动 no-op」「一半失败的 run 重启后只补做缺的部分」这两个上游的卖点直接落地——不需要把 chunking / embedding / store 拽进来。

---

## Problem / 问题

s01 的最小闭环里没有文档状态这件事——`Insert(text)` 进去就直接走 chunk → embed → store，没记账。这导致三个具体痛点：

1. **重复插入会重复消费 token**：同一篇 `book.txt` 跑两遍 `go run .`，第二次照样会把 80 个 chunk 全部重新 embed 一次。生产环境用真 OpenAI 一份长文档来回踩几次就是几美金白扔。
2. **没法 resume on failure**：embed 到第 47 个 chunk 时 OpenAI 返回 503，整个 Insert 直接错误退出，已经 embed 的 46 个 chunk 全都白费。重启后没有任何信息能告诉新进程「上次跑到哪了」。
3. **多文档管线没法并发追踪**：用户一次 Insert 一个文件夹，10 篇文档同时在不同阶段（有的 chunking、有的 embedding、有的 done），UI 没法画进度条，CI 没法 assert「该完成的都完成了」。

上游 LightRAG 的解法是给每篇文档配一个独立的状态记录：`DocProcessingStatus`，存在一个 `JsonDocStatusStorage` 里。这是 LightRAG 的 resume-on-failure 卖点的载货实现。s03 的任务就是把这套机制单独抽出来在 ~250 LOC 里讲清楚。

## Solution / 解决方案

四件事：

1. **`DocStatus` 是个 4 值字符串枚举**：`pending` / `processing` / `processed` / `failed`。值是字符串而不是 int 是为了 JSON 持久化时直接可读、对照上游不用查表。
2. **`DocProcessingStatus` 是个普通 struct**：13 个字段，对应上游 dataclass。`DocID` / `ContentSummary` / `ContentLength` / `FilePath` / `Status` / `CreatedAt` / `UpdatedAt` / `TrackID` / `ChunksCount` / `ChunksList` / `ErrorMsg` / `Metadata` 这些。
3. **`IsValidTransition(from, to)` 是状态机的载货代码**：4 条合法边，PROCESSED 是终态，FAILED→PROCESSING 是重试边——这条边就是 resume-on-failure 的核心。任何不在表里的转移会拿到一个 `*transitionError`，可以 `errors.Is(err, ErrInvalidTransition)` 判断。
4. **`DocStatusStore` 是带锁 + JSON 持久化的 store**：`map[string]*DocProcessingStatus` + `sync.RWMutex`。`Persist` 用「写到 `.tmp` 再 `os.Rename`」的原子重命名，reader 永远看到旧文件或新文件。构造时如果 JSON 文件已存在就 `Load` 进来。

`Enqueue` 是 dedup 的入口：第一次调用插入 PENDING 记录返回 `(rec, true, nil)`；第二次同 docID 调用直接返回 `(existing, false, nil)`，**不覆盖**已有记录。这是把上游 `apipeline_enqueue_documents` 里的「filter_keys 过滤一遍 → 只 upsert 新的」两步合并成一步——同样的 dedup 语义，少了一层。

## How It Works / 工作原理

```
       Enqueue(docID, content, filePath)
                │
                ├─ 已存在？ ──┐
                │            ▼
                │    返回 (existing, false, nil)
                │    没有副作用
                │
                ▼ 新插入
        ┌──────────────┐
        │   PENDING    │
        └──────┬───────┘
               │ MarkProcessing
               ▼
        ┌──────────────┐
        │  PROCESSING  │◀─────────┐
        └──┬─────────┬─┘          │ MarkProcessing (FAILED 后重试)
           │         │            │
 MarkProcessed    MarkFailed      │
           │         │            │
           ▼         ▼            │
   ┌───────────┐ ┌────────┐      │
   │ PROCESSED │ │ FAILED │──────┘
   │ (终态)    │ └────────┘
   └───────────┘
```

核心 30 行（节选自 [`agents/s03-doc-status/doc_status_store.go`](https://github.com/Ding-Ye/learn-lightrag/blob/main/agents/s03-doc-status/doc_status_store.go)）：

```go
func (s *DocStatusStore) Enqueue(ctx context.Context, docID, content, filePath string) (*DocProcessingStatus, bool, error) {
    if err := ctx.Err(); err != nil {
        return nil, false, err
    }
    s.mu.Lock()
    defer s.mu.Unlock()

    if existing, ok := s.docs[docID]; ok {
        // 重复插入直接 no-op —— 这是 resume-on-failure 的核心特性。
        return existing, false, nil
    }
    now := s.Clock()
    rec := &DocProcessingStatus{
        DocID:          docID,
        ContentSummary: summarize(content),
        ContentLength:  len(content),
        FilePath:       filePath,
        Status:         DocStatusPending,
        CreatedAt:      now,
        UpdatedAt:      now,
        ChunksList:     []string{}, // 永不为 nil —— 上游 chunks_list 默认 factory
    }
    s.docs[docID] = rec
    return rec, true, nil
}

func (s *DocStatusStore) transition(ctx context.Context, docID string, to DocStatus, mutate func(*DocProcessingStatus)) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    rec, ok := s.docs[docID]
    if !ok {
        return fmt.Errorf("doc-status: %s: not found", docID)
    }
    if !IsValidTransition(rec.Status, to) {
        return &transitionError{DocID: docID, From: rec.Status, To: to}
    }
    rec.Status = to
    rec.UpdatedAt = s.Clock()
    if mutate != nil { mutate(rec) }
    return nil
}
```

**4 个非显然之处**：

1. **`Enqueue` 的 `(rec, fresh, err)` 三元返回是设计要点**：调用方只看 `fresh==true` 决定要不要继续走 chunking + embedding。若 `fresh==false`，说明这份内容已经在 store 里（哪怕状态是 FAILED 也不重置——后面靠 `MarkProcessing(FAILED→PROCESSING)` 重试，记录里 `ChunksList` / `CreatedAt` / `FilePath` 全部保留）。
2. **MD5 而不是 sha256**：上游用 MD5 做内容寻址（`compute_mdhash_id` in lightrag/utils.py）。我们也用 MD5，保持 docID byte-equal 与上游 JSON 文件兼容。这里 MD5 是「内容地址哈希」，不是「安全哈希」——抗碰撞攻击不是相关属性。
3. **`Persist` 走原子重命名**：`os.WriteFile(tmp, ..., 0o644)` + `os.Rename(tmp, target)`。POSIX 保证 rename 是原子的，所以 reader 要么看到完整旧文件、要么看到完整新文件，绝不会看到半写状态。这跟上游 `write_json` 是同款做法。
4. **`ChunksList` 总是被 defensive copy**：`MarkProcessed` / `Get` / `ListByStatus` 都不直接返回内部 slice 引用，而是 `append([]string(nil), src...)` 拷一份。caller 改返回值不会污染 store 内存里的真值。这是「值语义优先」的 Go 习惯，也避免了「caller 拿到 ChunksList 后 sort.Strings 把内存里的也排序了」这种隐蔽 bug。

## What Changed / 与 s01 的变化

```diff
 // s01 pipeline.go: Insert 完全没有 doc-status 的概念
-func (p *Pipeline) Insert(ctx context.Context, text string) error {
-    chunks := p.chunker.Chunk(text)
-    embs, _ := p.embedder.Embed(ctx, chunkContents(chunks))
-    for i, e := range embs {
-        p.vectorStore.Upsert(VectorRecord{ID: fmt.Sprintf("c-%d", i), Vector: e})
-    }
-    return nil
-}

+// s03: Insert 前面挂一层状态机
+func (p *Pipeline) Insert(ctx context.Context, text, filePath string) error {
+    docID := MD5DocID(text)                              // 内容寻址
+    rec, fresh, err := p.docStatus.Enqueue(ctx, docID, text, filePath)
+    if err != nil { return err }
+    if !fresh {
+        // 已经处理过（或正在处理 / 失败待重试）——直接返回，零副作用
+        log.Printf("doc %s already known: status=%s", docID[:8], rec.Status)
+        return nil
+    }
+
+    // 真正开始处理：转 PROCESSING
+    if err := p.docStatus.MarkProcessing(ctx, docID); err != nil { return err }
+
+    chunks := p.chunker.Chunk(text)
+    chunkIDs := make([]string, len(chunks))
+    for i := range chunks { chunkIDs[i] = fmt.Sprintf("%s::chunk-%d", docID, i) }
+
+    embs, err := p.embedder.Embed(ctx, chunkContents(chunks))
+    if err != nil {
+        // 失败后状态保留 + 错误信息留底，下次启动 ListByStatus(FAILED) 能扫到
+        _ = p.docStatus.MarkFailed(ctx, docID, err.Error())
+        return err
+    }
+    for i, e := range embs { p.vectorStore.Upsert(VectorRecord{ID: chunkIDs[i], Vector: e}) }
+
+    // 全部完成：转 PROCESSED + 记账 ChunksList，下次 Enqueue 同内容直接 no-op
+    return p.docStatus.MarkProcessed(ctx, docID, chunkIDs)
+}
```

调用方 `main.go` 也只是多读一次 filePath、多 Persist 一次：`p.docStatus.Persist(ctx)` 在 Insert 完成后调一次，下次启动构造 `DocStatusStore` 时会自动 Load。s03 的 store 不依赖 s05 的 KV 接口；s05 章节会演示如何把这层 store 重写在 KV 之上变成 30 行。

## Try It / 动手试一试

```bash
cd agents/s03-doc-status

# 默认跑 testdata/sample.txt — 跑完会 println 三轮的每一次状态转移
go run .

# 输出会长成这样:
# == round 1: fresh ingest ==
#   Enqueue          (none) -> pending (fresh)  chunks=0
#   MarkProcessing   pending -> processing  chunks=0
#   MarkProcessed    processing -> processed  chunks=3
#
# == round 2: re-ingest same content (dedup) ==
#   Enqueue returned fresh=false (no-op) — content already PROCESSED, no double-chunking.
#
# == round 3: simulate failure on a second doc ==
#   doc 5b8a... → FAILED (err="synthetic: provider rate-limit")
#   attempting FAILED -> PROCESSED rejected as expected: doc-status: ...: cannot transition failed -> processed
#
# == resume-on-failure scan ==
#   ListByStatus(FAILED) found 1 doc(s) needing retry:
#     - 5b8a...  err="synthetic: provider rate-limit"  updated=...

# 跑测试 (5 个，全离线，无外部依赖)
go test -v ./...
# === RUN   TestStatusTransitionsValid          快乐路径全状态走通
# === RUN   TestStatusInvalidTransitionRejected PROCESSED -> PROCESSING 必须返回 ErrInvalidTransition
# === RUN   TestStatusPersistAcrossRestart      3 篇文档 Persist 后重新构造 store 全部 round-trip
# === RUN   TestDuplicateInsertIsNoop           二次 Enqueue 不覆盖 status / FilePath
# === RUN   TestFailedDocPreservesErrorMsg      MarkFailed 后 ErrorMsg 字段留得住
```

测完后 `./lightrag-data/doc_status.json` 里就有持久化的状态文件，下次 `go run .` 会先 Load 进来。删掉这个目录可以从空白重跑。

## Upstream Source Reading / 上游源码阅读

上游的核心 30 行（节选自 `lightrag/kg/json_doc_status_impl.py`，完整注解见 [`upstream-readings/s03-doc-status.py`](../../upstream-readings/s03-doc-status.py)）：

```python
@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        workspace_dir = (
            os.path.join(working_dir, self.workspace) if self.workspace else working_dir
        )
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data: return
        for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
            if "chunks_list" not in doc_data:
                doc_data["chunks_list"] = []          # 默认空 list
            await _cooperative_yield(i)
        async with self._storage_lock:
            self._data.update(data)                   # 内存里先更新
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()              # 末尾刷盘

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                needs_reload = write_json(data_dict, self._file_name)  # 原子重命名
                # ... (省略：sanitize + reload 分支，s03 不涉及)
                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])
```

**对照阅读要点**：

- **`@dataclass` + `@final` vs. Go struct**：上游用 dataclass + final 装饰器锁定字段。Go 里我们就是普通 struct + 不导出的字段（首字母小写）来达到同样的「不可被外部继承」效果。
- **`workspace` / `namespace` prefix vs. s03 单 store**：上游一个进程可以挂多份 doc-status（不同 workspace 隔离），文件名是 `kv_store_doc_status.json` 加前缀。s03 只有一份 store，路径直接是 `./lightrag-data/doc_status.json`——简化后的等价行为。
- **`_cooperative_yield(i)` vs. `sync.RWMutex`**：上游每 N 项让一次出 asyncio 协程，不阻塞别的 task。Go 的 sync.RWMutex 是阻塞锁，但临界区里只有 map 赋值——纳秒级，不会阻塞太久。教学场景这种简化值得；s05 的 KV 章节会用 per-key sync.Map 做更细粒度。
- **`write_json` 原子重命名 vs. `os.Rename`**：上游 `write_json` 内部做 write 到 `.tmp` + `os.replace`，s03 是 `os.WriteFile(tmp,...)` + `os.Rename(tmp,target)`。两边都依赖 POSIX rename 的原子性保证。
- **`upsert` 的 dedup 语义在哪**：注意上游 `upsert` 自己**不**做 dedup——`self._data.update(data)` 是字典 update，会覆盖。dedup 在更上层 `lightrag/lightrag.py:apipeline_enqueue_documents` 里调 `filter_keys` 实现。s03 把这两步合并到 `Enqueue` 里：检查 + 插入是同一把锁下的原子操作，调用方少一次往返。
- **`get_docs_by_status` → `get_docs_by_statuses([status])`**：上游单查询是多查询的特例。s03 直接给 `ListByStatus(status)` 一个方法，多状态查询如果 s09 需要会再加；YAGNI。

**想读更多**：上游 `JsonDocStatusStorage` 还有 `get_status_counts` / `get_docs_paginated` / `get_doc_by_file_path` / `get_docs_by_track_id` 这些 dashboard 用的便利方法，都是 `_data` 字典上的 list/filter。s03 暂时只实现 `ListByStatus`，其余作为练习；s05 章节真正铺开 `KVStore` 接口后会一并补上。s09（实体抽取）会真正用到 `ListByStatus(FAILED) + ListByStatus(PROCESSING)` 这套 resume-on-failure 扫描，到时候这一层会反复被读。

---

**下一节预告**：s04 把 s01 的「按换行切分」换成基于 token 数的 1200/100 滑窗（用 `pkoukk/tiktoken-go`），把 chunking_by_token_size 这个 upstream 工具函数完整 Go 化。chunk_id 命名仍然是 `<docID>::chunk-<index>`——s03 的 docID 也是这同一个 MD5 内容地址，两节天然衔接。
