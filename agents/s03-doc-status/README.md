# s03 · 文档状态机 / Document status state machine

> 把 s01 占位的字符串常量推进成有持久化、转移校验、错误信息保留的 `DocStatusStore`，配合 MD5 内容寻址实现「重复插入直接 no-op」与「失败后从 FAILED 恢复重试」。
> Promote s01's placeholder string constants into a persisted state machine with transition validation and error-msg preservation, plus MD5 content-addressing that makes "re-insert is a no-op" and "resume from FAILED" both fall out for free.

## 跑起来 / Run

```bash
cd agents/s03-doc-status

# 默认 testdata/sample.txt — 跑完会 println 每一次状态转移
# default testdata/sample.txt — prints every state transition
go run .

# 自定义文档
go run . -doc /path/to/your/file.txt

# 跑测试 (5 个，全离线，无外部依赖)
# tests (5 of them, fully offline, no external deps)
go test -v ./...
```

输出里会看到三轮：第一轮 PENDING → PROCESSING → PROCESSED 的快乐路径；第二轮拿同一份内容再 Enqueue 一次，store 直接返回 `(existing, false, nil)` 证明 dedup；第三轮在另一个文档上模拟 FAILED，最后用 `ListByStatus(DocStatusFailed)` 扫出待重试的列表。
You'll see three rounds: round 1 walks the happy path PENDING → PROCESSING → PROCESSED; round 2 re-Enqueues identical content and the store returns `(existing, false, nil)` proving dedup; round 3 fails a synthetic doc and the resume scan via `ListByStatus(DocStatusFailed)` surfaces it.

## 文件 / Files

| 文件 / file | 作用 / role | LOC |
|---|---|---|
| `doc_status.go` | `DocStatus` 常量 + `DocProcessingStatus` 结构 + `IsValidTransition` + `ErrInvalidTransition` / state constants + record + transition table + sentinel error | ~85 |
| `doc_status_store.go` | `DocStatusStore`：sync.RWMutex + JSON 原子重命名持久化 + Enqueue/Mark*/Get/ListByStatus / store: RWMutex + atomic-rename JSON persist + the six methods | ~220 |
| `hash.go` | `MD5DocID(content)`：上游 `compute_mdhash_id` 的 Go 翻版 / Go port of upstream `compute_mdhash_id` | ~25 |
| `main.go` | CLI demo：3 轮跑通整个状态机 / CLI demo: 3 rounds exercising the full state machine | ~115 |
| `doc_status_test.go` | 5 个测试：valid / invalid / persist / dedup / failed-msg / 5 tests covering the same five guarantees | ~210 |
| `testdata/sample.txt` | 约 400 词 Eleanor Hartwell 传记 / ~400 words of Eleanor Hartwell biography prose | — |

## 关键教学点 / Key teaching points

- **状态机是个 2D 表 / The state machine is a 2-D table**：`IsValidTransition(from, to)` 列举 4 条合法边——PENDING→PROCESSING/FAILED、PROCESSING→PROCESSED/FAILED、FAILED→PROCESSING（重试）。PROCESSED 是终态。任何不在表里的转移都返回 `*transitionError` 并 `Unwrap` 到 `ErrInvalidTransition`。
- **MD5 内容寻址 = 重复插入 no-op / MD5 content-addressing = re-insert is a no-op**：`Enqueue` 只在 docID 不存在时插入，已存在则原样返回（不覆盖 status / chunks_list）。这是 upstream resume-on-failure 的核心：进程挂在 PROCESSING，重启后再 Enqueue 同一份文档拿到的还是那个 PROCESSING 记录，不会被重置成 PENDING。
- **JSON 原子重命名 / Atomic-rename JSON persistence**：`Persist` 写到 `<path>.tmp` 再 `os.Rename` 覆盖。reader 永远看到旧文件或新文件，不会看到半写状态。这是 upstream `write_json` 的同款做法。
- **`ListByStatus` 是恢复扫描的唯一入口 / `ListByStatus` is the resume-scan primitive**：worker 启动时跑 `ListByStatus(FAILED)`（甚至 `PROCESSING`，应对中途 kill）拿到所有需要重试的文档。s09 的抽取流水线会直接复用这个扫描——这就是 plan.md 里提到的「resume-on-failure 出口」。
- **defensive copy / 防御性拷贝**：`Get` / `ListByStatus` 返回的指针都是 `cp := *rec` 后的副本，`ChunksList` 也单独 `append` 了一份。caller 改返回值不会污染内存里的真值。

完整章节文档：[docs/zh/s03-doc-status.md](../../docs/zh/s03-doc-status.md) · [docs/en/s03-doc-status.md](../../docs/en/s03-doc-status.md)
