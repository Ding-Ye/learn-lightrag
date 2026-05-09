---
title: "附录 B · 上游源码导读地图"
chapter: appendix-b
slug: appendix-b-upstream-map
est_read_min: 12
---

# 附录 B · 上游源码导读地图

> 把上游 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)（Python，~50K LOC）的关键文件铺成一张可走的地图：每个文件对应到 learn-lightrag 的哪一节，以及把这些文件按"理解上游"的视角排出推荐阅读顺序。读完 11 节再用这张图回看上游，每根指针都不再陌生。

---

## 推荐阅读顺序

跟着这条线一遍走完，等于把 11 节学到的东西再印证一次：

```
1. lightrag/__init__.py             — 看公开 API（LightRAG, QueryParam 是什么）
2. lightrag/base.py:1-700           — 抽象层：Storage / DocStatus / QueryParam / Schema
3. lightrag/operate.py:1-400        — chunking + 一些工具函数
4. lightrag/kg/json_kv_impl.py      — 第一个真存储 impl，看 dataclass + namespace
5. lightrag/kg/nano_vector_db_impl.py — 向量库怎么 wrap nano-vectordb
6. lightrag/kg/networkx_impl.py     — 知识图谱 impl + BFS 子图
7. lightrag/operate.py:400-3200     — 抽取 + summarization 大循环（巨长，可分次读）
8. lightrag/lightrag.py             — 中央 orchestrator，把所有零件接起来
9. lightrag/operate.py:3200-5200    — 4 种 query mode 真实代码
10. lightrag/llm/openai.py          — LLM provider wrap（也是 Phase G 多模型基线）
11. lightrag/api/lightrag_server.py — FastAPI 服务化（可选；扩展练习里有）
```

---

## 上游文件 → 课程章节映射

每个上游文件标注了"从哪个 mini 章节出发回看最有效"。

| 上游文件 | 引用行段 | 课程章节 | 看什么 |
|---|---|---|---|
| `lightrag/base.py:71` | TextChunkSchema | s04 | chunk 的字段约定 |
| `lightrag/base.py:77` | QueryParam | s11 | 查询参数 dataclass |
| `lightrag/base.py:189` | BaseVectorStorage | s07 | 向量库抽象 |
| `lightrag/base.py:308` | BaseKVStorage | s05 | KV 抽象 + filter_keys |
| `lightrag/base.py:333` | BaseGraphStorage | s08 | 图存储抽象 + GetSubgraph |
| `lightrag/base.py:662-697` | DocStatus + DocProcessingStatus | s03 | 状态机数据形 |
| `lightrag/base.py:759` | QueryResult | s11 + s_full | 查询返回形态 |
| `lightrag/operate.py:102-166` | chunking_by_token_size | s04 | token 滑窗切分 |
| `lightrag/operate.py:167-303` | _handle_entity_relation_summary | s10 | 阈值决策 |
| `lightrag/operate.py:304-385` | _summarize_descriptions | s10 | 递归归并 |
| `lightrag/operate.py:560-846` | rebuild_knowledge_from_chunks | (扩展练习) | 增量重建 |
| `lightrag/operate.py:~2883-3163` | extract_entities | s09 | 抽取主流程 + gleaning |
| `lightrag/operate.py:3164-3410` | kg_query | s11 | 4 种 mode 派发 |
| `lightrag/operate.py:3516-4055` | retrieval helpers | s11 | 子图扩展 + chunk 收集 |
| `lightrag/operate.py:4930-5200` | naive_query | s11 | 最简 mode |
| `lightrag/kg/json_kv_impl.py` | full | s05 | KVStore JSON 实现 |
| `lightrag/kg/json_doc_status_impl.py` | full | s03 | 状态机持久化 |
| `lightrag/kg/nano_vector_db_impl.py` | full | s07 | 向量库实现 |
| `lightrag/kg/networkx_impl.py` | full | s08 | 图存储 |
| `lightrag/kg/{neo4j,memgraph,postgres,mongo,faiss,milvus,qdrant,redis,opensearch}_impl.py` | full | (Appendix B 扩展) | 其他后端实现 |
| `lightrag/llm/openai.py` | first 120 lines | s02 + s06 + Phase G | provider 封装 |
| `lightrag/llm/{anthropic,azure_openai,ollama,bedrock,hf,gemini}.py` | full | Phase G 多模型 addendum | 其他 provider 包装 |
| `lightrag/lightrag.py:1-150` | LightRAG @dataclass | s_full | 中央配置 dataclass |
| `lightrag/lightrag.py:1237` | ainsert | s_full step 2 | insert 入口 |
| `lightrag/lightrag.py:2622` | aquery | s_full step 11 | query 入口（包装层） |
| `lightrag/lightrag.py:2884-2970` | aquery_llm | s_full step 12-15 | query 实际工作 |
| `lightrag/prompt.py` | PROMPTS dict | s09 + s10 + Appendix A | 抽取/归并/关键词的 prompt |
| `lightrag/utils.py` | retry / async helpers | s02 + s06 | 工具函数（我们用 stdlib 重写） |
| `lightrag/api/lightrag_server.py` | full | (Appendix B 扩展) | FastAPI 服务化 |

---

## 5 个推荐扩展练习

> 这些不在课程主线里，但每个都是把 mini 升级到"真实生产可用"的常见路径。每个扩展约 100-300 LOC。

### 1. Neo4j 图存储后端

**学到什么**：让 s08 的 `GraphStore` interface 在生产数据库上落地。Neo4j 用 Cypher 表达图查询，比 NetworkX 的 in-memory 邻接表更具扩展性。

**改动范围**：
- 在 s08 同目录新建 `neo4j_graph_store.go`（不动 `adjacency_graph.go`），实现 `GraphStore` 接口。
- 用 `github.com/neo4j/neo4j-go-driver/v5` 连接 Neo4j。
- `UpsertNode` → `MERGE (n:Entity {name: $name}) SET n.description = $desc`。
- `GetSubgraph` → 用 Cypher 的 `MATCH (n:Entity {name: $seed})-[r*1..$depth]-(m) RETURN n, r, m LIMIT $maxNodes`。
- 主测试用 `dockertest` 起临时 Neo4j 容器跑集成测试。

**约 300 LOC**。完成后你能给 s08 的 demo 加 `-backend neo4j` flag。

---

### 2. 在 s11 加流式响应

**学到什么**：把 `Provider.Complete` 的 `Stream=true` 路径打通——LLM 边生成边返回 token，前端可以做 typing effect。

**改动范围**：
- 把 `Provider` 接口改成 `Complete(ctx, req) (<-chan Token, error)`（或单独加一个 `CompleteStream` 方法保持向后兼容）。
- s11 的 `pipeline.go:Query` 也返回 `(chan TokenWithRefs, error)`，先吐 References 再开始吐文字 token。
- main.go 的 demo 改成边收边打印。

**关键点**：Anthropic / OpenAI 的 SSE 格式略不同，Phase G 多模型 addendum 文档里已经讨论过。

**约 200 LOC**。

---

### 3. Reranker 介入

**学到什么**：在 retrieval 和 LLM 之间插一层 cross-encoder reranker，能让 hybrid mode 召回率再涨 5-10%。

**改动范围**：
- 在 s11 加一个 `Reranker` interface：`Score(query string, candidates []string) []float32`。
- 实现 `CohereReranker` 或 `JinaReranker`（都是 HTTP 调用）。
- `Pipeline.Query` 在 retrieval 后、context 拼装前调 `Reranker.Score`，按分数重排 top-K。

**约 150 LOC**。

---

### 4. Postgres KV 后端

**学到什么**：让 s05 的 `KVStore` interface 跑在真 SQL 数据库上。同一接口 → 完全不同的存储引擎，复用所有 caller 代码。

**改动范围**：
- 新建 `postgres_kv_store.go`，用 `github.com/jackc/pgx/v5` 连 Postgres。
- 表 schema：`CREATE TABLE kv (namespace TEXT, id TEXT, data JSONB, updated_at TIMESTAMPTZ, PRIMARY KEY (namespace, id))`。
- `Upsert` → `INSERT ... ON CONFLICT (namespace, id) DO UPDATE`。
- `FilterMissing` → `SELECT id FROM unnest($ids) AS u(id) LEFT JOIN kv ON kv.id = u.id WHERE kv.id IS NULL`。
- 集成测试用 `dockertest` 起临时 Postgres。

**约 250 LOC**。

---

### 5. 给 s11 加一个 HTTP 服务器

**学到什么**：把 mini 包装成可远程调用的服务，对应上游的 `lightrag/api/lightrag_server.py`。

**改动范围**：
- 在 s11 同目录新建 `server.go`，用 `net/http` （**不引入 web 框架**）。
- 暴露 `POST /insert`（body: `{"text": "...", "doc_id": "..."}`）和 `POST /query`（body: `{"q": "...", "mode": "hybrid"}`）。
- JSON 编码 `QueryResult`。
- 加一个 `Authorization: Bearer <token>` 中间件。

**约 150 LOC**。完成后 `curl localhost:8080/query -d '{"q":"..."}'` 就能用。

---

## 一句话总结

> 11 节学完，这张地图让你知道**"下次想读上游某个特性时该从哪一节出发"**。每个上游文件都有"理解它"的最近 mini 锚点。
