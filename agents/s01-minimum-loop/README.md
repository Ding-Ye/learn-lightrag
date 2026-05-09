# s01 · 最小 RAG 闭环 / Minimum RAG loop

> 一节走完一条端到端的 RAG 管道：chunk → embed → store → retrieve → complete。
> One chapter, one end-to-end RAG pipeline: chunk → embed → store → retrieve → complete.

## 跑起来 / Run

```bash
# 真 OpenAI（需要 OPENAI_API_KEY） / real OpenAI (needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
go run . -q "What did Eleanor patent in 1872?"

# 全离线 mock，CI 用的就是这个 / fully offline mock; what CI runs
go run . -provider mock -q "What did Eleanor patent in 1872?"

# verbose 看检索 / verbose to see retrieval
go run . -provider mock -v -q "Where did Eleanor's papers end up?"

# 测试 / tests
go test -v ./...
```

## 文件 / Files

| 文件 / file | 作用 / role | LOC |
|---|---|---|
| `main.go` | CLI 入口；解析 flag、读 doc、调 pipeline / CLI entry: parse flags, load doc, drive pipeline | ~85 |
| `provider.go` | `Provider` 接口 + OpenAI 实现 / `Provider` interface + OpenAI implementation | ~160 |
| `provider_mock.go` | `MockProvider`：把 system prompt 截断回显，确定性 / `MockProvider`: echoes truncated system prompt, deterministic | ~45 |
| `embedder.go` | `EmbeddingProvider` 接口 + OpenAI 实现 / `EmbeddingProvider` interface + OpenAI implementation | ~110 |
| `embedder_mock.go` | `MockEmbedder`：sha256(text) → 1536d 单位向量 / `MockEmbedder`: sha256(text) → 1536d unit vector | ~70 |
| `chunking.go` | `ChunkByNewlines`：按段落切分 + 长段硬切 / `ChunkByNewlines`: paragraph split + long-paragraph hard cut | ~55 |
| `vectorstore.go` | `VectorStore` 接口 + 切片扫描余弦实现 / `VectorStore` interface + slice-scan cosine implementation | ~135 |
| `kv.go` | `KVStore` 接口 + `sync.Map` 实现 / `KVStore` interface + `sync.Map` implementation | ~75 |
| `pipeline.go` | `Pipeline.Insert/Query` 串起以上五个 / `Pipeline.Insert/Query` glues the five above together | ~165 |
| `lightrag_test.go` | 5 个测试，全部用 mock / 5 tests, all using mock | ~145 |
| `testdata/book.txt` | 虚构的 Eleanor Hartwell 短传，喂给 RAG 试用 / fictional short biography, fed to RAG for demos | — |
| `testdata/expected.txt` | 三段示例运行的形态 / shape of three sample runs | — |

## 关键教学点 / Key teaching points

- **接口先于实现 / Interfaces before impls**：每个 storage/provider 都先声明 interface 再写一个具体实现。后面 10 节把 `sync.Map` 换成 JSON、把切片扫描换成索引、把 mock 换成真 OpenAI——调用方一行不改。
- **`MockProvider` + `MockEmbedder` 让 CI 完全离线 / Mock pair keeps CI offline**：CI 永远不调外网；本地有 key 才走真 OpenAI。
- **`chunkID` 是契约 / `chunkID` is the contract**：`<docID>::chunk-<index>` 这个串既是 KV key 也是 VDB ID 也是 References 输出。后面所有节都保持。
- **代码结构对齐上游 / Code structure mirrors upstream**：`Insert` 对应 `LightRAG.ainsert()` 的最小子集；`Query` 对应 `aquery()` 的 naive mode。每个文件顶部注释指出对应的上游行号。
- **零外部依赖 / Zero external deps**：s01 只用 stdlib + net/http。s04 才引入 `pkoukk/tiktoken-go`。

完整章节文档：[docs/zh/s01-minimum-loop.md](../../docs/zh/s01-minimum-loop.md) · [docs/en/s01-minimum-loop.md](../../docs/en/s01-minimum-loop.md)
